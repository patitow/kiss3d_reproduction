#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark pipeline: roda imagem->3D para um conjunto de pares (input_image, gt_mesh)
e calcula métricas (Chamfer/F-score) usando evaluate_mesh_against_gt.

Uso sugerido:
    python scripts/benchmark_pipeline.py --pairs data/raw/gazebo_pairs.csv --output outputs/benchmark_gazebo --pipeline-mode flux --quality-mode balanced

Se --pairs não for fornecido, tenta autogerar pares procurando imagens e .obj/.glb
de mesmo stem em data/raw/gazebo_dataset.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

from kiss3d_utils_local import evaluate_mesh_against_gt, TMP_DIR, OUT_DIR
from kiss3d_wrapper_local import init_wrapper_from_config, run_image_to_3d


def discover_pairs(dataset_root: Path) -> List[Tuple[str, str]]:
    exts_img = {".png", ".jpg", ".jpeg"}
    exts_mesh = {".obj", ".glb"}
    pairs: List[Tuple[str, str]] = []
    for img_path in dataset_root.rglob("*"):
        if img_path.suffix.lower() not in exts_img:
            continue
        stem = img_path.stem
        candidate_mesh = None
        for extm in exts_mesh:
            mpath = img_path.with_name(stem + extm)
            if mpath.exists():
                candidate_mesh = mpath
                break
        if candidate_mesh:
            pairs.append((str(img_path), str(candidate_mesh)))
    return pairs


def load_pairs_from_csv(csv_path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inp = row.get("input") or row.get("image") or row.get("input_image")
            gt = row.get("gt_mesh") or row.get("mesh") or row.get("gt")
            if inp and gt:
                pairs.append((inp, gt))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Benchmark de imagem->3D com métricas")
    parser.add_argument("--pairs", type=str, help="CSV com colunas input,gt_mesh")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/raw/gazebo_dataset",
        help="Raiz do dataset para descoberta automática",
    )
    parser.add_argument("--output", type=str, default="outputs/benchmark", help="Pasta de saída")
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline_config/default.yaml",
        help="Config YAML relativa à raiz Kiss3DGen",
    )
    parser.add_argument("--pipeline-mode", type=str, choices=["flux", "multiview"], default="flux")
    parser.add_argument("--quality-mode", type=str, choices=["fast", "balanced", "high"], default=None)
    parser.add_argument("--use-controlnet", action="store_true", default=True)
    parser.add_argument("--enable-redux", action="store_true", default=True)
    parser.add_argument("--use-mv-rgb", action="store_true", default=True)
    parser.add_argument("--gt-samples", type=int, default=50000)
    parser.add_argument("--disable-metrics-normalization", action="store_true", default=False)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs:
        pairs = load_pairs_from_csv(Path(args.pairs))
    else:
        pairs = discover_pairs(Path(args.dataset_root))

    if not pairs:
        print("[ERRO] Nenhum par (input, gt_mesh) encontrado.")
        return

    print(f"[INFO] Total de pares: {len(pairs)}")

    # Inicializar wrapper
    k3d_wrapper = init_wrapper_from_config(
        args.config,
        fast_mode=False,
        disable_llm=False,
        load_controlnet=args.use_controlnet,
        load_redux=args.enable_redux,
        pipeline_mode=args.pipeline_mode,
        quality_mode=args.quality_mode,
    )

    results: List[Dict] = []

    for idx, (inp, gt) in enumerate(pairs, start=1):
        label = Path(inp).stem
        view_out_dir = output_dir / f"{idx:03d}_{label}"
        view_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(pairs)}] Rodando {inp} -> {view_out_dir}")

        try:
            bundle_path, mesh_path = run_image_to_3d(
                k3d_wrapper,
                inp,
                enable_redux=args.enable_redux,
                use_mv_rgb=args.use_mv_rgb,
                use_controlnet=args.use_controlnet,
                pipeline_mode=args.pipeline_mode,
            )
        except Exception as exc:
            print(f"[ERRO] Falha ao processar {inp}: {exc}")
            results.append(
                {
                    "input": inp,
                    "gt_mesh": gt,
                    "success": False,
                    "error": str(exc),
                }
            )
            continue

        # Copiar artefatos
        bundle_copy = None
        mesh_copy = None
        if bundle_path and os.path.exists(bundle_path):
            bundle_copy = view_out_dir / f"{label}_3d_bundle.png"
            Path(bundle_path).replace(bundle_copy)
        if mesh_path and os.path.exists(mesh_path):
            mesh_copy = view_out_dir / f"{label}.glb"
            Path(mesh_path).replace(mesh_copy)

        metrics = None
        try:
            metrics = evaluate_mesh_against_gt(
                mesh_copy or mesh_path,
                gt,
                num_samples=args.gt_samples,
                normalize=not args.disable_metrics_normalization,
            )
        except Exception as exc:
            print(f"[AVISO] Falha ao calcular métricas para {label}: {exc}")

        results.append(
            {
                "input": inp,
                "gt_mesh": gt,
                "bundle": str(bundle_copy) if bundle_copy else bundle_path,
                "mesh": str(mesh_copy) if mesh_copy else mesh_path,
                "metrics": metrics,
                "success": True,
            }
        )

    # Resumo
    summary = {"count": len(results)}
    chamfer_l1 = [r["metrics"]["chamfer_l1"] for r in results if r.get("metrics")]
    chamfer_l2 = [r["metrics"]["chamfer_l2"] for r in results if r.get("metrics")]
    if chamfer_l1:
        summary["chamfer_l1_mean"] = sum(chamfer_l1) / len(chamfer_l1)
    if chamfer_l2:
        summary["chamfer_l2_mean"] = sum(chamfer_l2) / len(chamfer_l2)

    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Benchmark concluído. Resultados em {output_dir}")


if __name__ == "__main__":
    main()

