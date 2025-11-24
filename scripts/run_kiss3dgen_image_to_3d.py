#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline IMAGE TO 3D baseado no Kiss3DGen
Reproduz exatamente o pipeline do artigo Kiss3DGen
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Adicionar ninja ao PATH antes de qualquer import
project_root = Path(__file__).parent.parent
venv_path = project_root / "mesh3d-generator-py3.11"
possible_ninja_paths = [
    project_root,  # Pode estar na raiz do projeto
    venv_path / "Scripts",
    venv_path / "ninja" / "data" / "bin",
    venv_path / "Lib" / "site-packages" / "ninja" / "data" / "bin",
]

# Também verificar no site-packages do Python
try:
    import site
    site_packages = site.getsitepackages()[0] if site.getsitepackages() else None
    if site_packages:
        possible_ninja_paths.append(Path(site_packages) / "ninja" / "data" / "bin")
except:
    pass

ninja_found = False
for ninja_path in possible_ninja_paths:
    ninja_exe = ninja_path / "ninja.exe"
    if ninja_exe.exists():
        current_path = os.environ.get("PATH", "")
        if str(ninja_path) not in current_path:
            os.environ["PATH"] = str(ninja_path) + os.pathsep + current_path
        print(f"[INFO] Ninja encontrado e adicionado ao PATH: {ninja_path}")
        ninja_found = True
        break

if not ninja_found:
    print("[AVISO] Ninja nao encontrado - pode causar erros ao compilar extensoes C++")
    print("[INFO] Tentando adicionar Scripts ao PATH de qualquer forma...")
    scripts_path = venv_path / "Scripts"
    if scripts_path.exists():
        current_path = os.environ.get("PATH", "")
        if str(scripts_path) not in current_path:
            os.environ["PATH"] = str(scripts_path) + os.pathsep + current_path
            print(f"[INFO] Scripts adicionado ao PATH: {scripts_path}")

# Configurar CUDA_HOME se nao estiver definido
if "CUDA_HOME" not in os.environ and "CUDA_PATH" not in os.environ:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    
    if not cuda_home:
        try:
            import torch
            cuda_version = torch.version.cuda
            if cuda_version:
                cuda_base = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"
                possible_cuda_paths = []
                
                if os.path.exists(cuda_base):
                    for item in os.listdir(cuda_base):
                        cuda_path = os.path.join(cuda_base, item)
                        if os.path.isdir(cuda_path) and item.startswith("v"):
                            possible_cuda_paths.append(cuda_path)
                
                if cuda_version:
                    version_parts = cuda_version.split(".")
                    if len(version_parts) >= 2:
                        version_major_minor = f"{version_parts[0]}.{version_parts[1]}"
                        specific_path = os.path.join(cuda_base, f"v{version_major_minor}")
                        if os.path.exists(specific_path) and specific_path not in possible_cuda_paths:
                            possible_cuda_paths.insert(0, specific_path)
                
                cuda_found = False
                for cuda_path in possible_cuda_paths:
                    if os.path.exists(cuda_path) and os.path.exists(os.path.join(cuda_path, "bin", "nvcc.exe")):
                        os.environ["CUDA_HOME"] = cuda_path
                        os.environ["CUDA_PATH"] = cuda_path
                        print(f"[INFO] CUDA_HOME configurado: {cuda_path}")
                        cuda_found = True
                        break
                
                if not cuda_found:
                    print("[AVISO] CUDA_HOME nao encontrado automaticamente")
                    default_cuda = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1"
                    if os.path.exists(default_cuda):
                        os.environ["CUDA_HOME"] = default_cuda
                        os.environ["CUDA_PATH"] = default_cuda
                        print(f"[INFO] CUDA_HOME configurado para caminho padrao: {default_cuda}")
        except Exception as e:
            print(f"[AVISO] Erro ao configurar CUDA_HOME: {e}")

# Adicionar CUDA bin ao PATH se CUDA_HOME estiver configurado
cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if cuda_home:
    cuda_bin = os.path.join(cuda_home, "bin")
    if os.path.exists(cuda_bin):
        current_path = os.environ.get("PATH", "")
        if cuda_bin not in current_path:
            os.environ["PATH"] = cuda_bin + os.pathsep + current_path
            print(f"[INFO] CUDA bin adicionado ao PATH: {cuda_bin}")

# Adicionar paths - IMPORTANTE: Kiss3DGen precisa estar no path
kiss3dgen_path = project_root / "Kiss3DGen"
if not kiss3dgen_path.exists():
    print(f"[ERRO] Kiss3DGen nao encontrado em: {kiss3dgen_path}")
    print(f"[INFO] Certifique-se de que o diretorio Kiss3DGen existe")
    sys.exit(1)

sys.path.insert(0, str(project_root))

import argparse
import shutil

from kiss3d_utils_local import (
    TMP_DIR,
    OUT_DIR,
    ORIGINAL_WORKDIR,
    ensure_hf_token,
    evaluate_mesh_against_gt,
)
from kiss3d_wrapper_local import init_wrapper_from_config, run_image_to_3d

os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
# expandable_segments não suportado no Windows - removido para evitar warnings

DATASET_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _collect_dataset_views(dataset_root: Path, dataset_item: str) -> Dict[int, Path]:
    """
    Retorna um dicionário {view_index: caminho} para as imagens do dataset.
    """
    image_dir = dataset_root / "images"
    view_map: Dict[int, Path] = {}
    if not image_dir.exists():
        return view_map

    prefix = f"{dataset_item}_"
    for ext in DATASET_IMAGE_EXTS:
        for candidate in image_dir.glob(f"{prefix}*{ext}"):
            suffix = candidate.stem[len(prefix) :]
            if suffix.isdigit():
                view_map[int(suffix)] = candidate.resolve()

    return dict(sorted(view_map.items()))


def _parse_view_argument(raw_value: str, available_views: Dict[int, Path]) -> List[int]:
    """
    Converte a string fornecida em uma lista de índices válidos.
    """
    if not raw_value:
        return []

    normalized = raw_value.strip().lower()
    if normalized == "all":
        return list(available_views.keys())

    selected: List[int] = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError:
            continue
        if idx in available_views and idx not in selected:
            selected.append(idx)

    return selected

def main():
    parser = argparse.ArgumentParser(description="Pipeline IMAGE TO 3D - Kiss3DGen")
    parser.add_argument("--input", type=str, help="Caminho para imagem de input")
    parser.add_argument("--output", type=str, default="data/outputs/kiss3dgen", help="Diretorio de saida")
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline/pipeline_config/default.yaml",
        help="Caminho para config YAML (relativo ao diretorio Kiss3DGen)",
    )
    parser.add_argument("--enable-redux", action="store_true", default=True, help="Habilitar Redux")
    parser.add_argument("--use-mv-rgb", action="store_true", default=True, help="Usar RGB multiview")
    parser.add_argument("--use-controlnet", action="store_true", default=True, help="Usar ControlNet")
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Preset leve: menos passos, sem Redux/ControlNet, libera memoria agressivamente.",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Desabilita refinamento de prompt com LLM (economiza VRAM/RAM).",
    )
    parser.add_argument("--dataset-item", type=str, help="Nome do item em data/raw/gazebo_dataset")
    parser.add_argument("--dataset-root", type=str, default="data/raw/gazebo_dataset", help="Raiz do dataset local")
    parser.add_argument(
        "--dataset-view",
        type=str,
        default="0",
        help="Sufixo(s) da imagem (_{N}.jpg) a usar como input. "
        "Use 'all' para testar todas as vistas ou uma lista separada por vírgula (ex.: '0,2,4').",
    )
    parser.add_argument("--gt-mesh", type=str, help="Caminho para a malha ground-truth (obj/glb)")
    parser.add_argument("--gt-samples", type=int, default=50000, help="Numero de pontos amostrados para as métricas")
    parser.add_argument(
        "--metrics-out",
        type=str,
        help="Arquivo JSON para salvar métricas de reconstrução (default: <output>/<input>_metrics.json)",
    )
    parser.add_argument(
        "--disable-metrics-normalization",
        action="store_true",
        help="Não normaliza as malhas antes de calcular as métricas (por padrão normalizamos).",
    )
    
    args = parser.parse_args()
    
    original_cwd = os.getcwd()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    selected_inputs: List[Dict[str, Path]] = []
    dataset_view_arg = str(args.dataset_view) if args.dataset_view is not None else ""

    if args.dataset_item:
        available_views = _collect_dataset_views(dataset_root, args.dataset_item)
        if not available_views:
            print(f"[ERRO] Nenhuma imagem encontrada para {args.dataset_item} em {dataset_root / 'images'}.")
            sys.exit(1)

        requested_views = _parse_view_argument(dataset_view_arg, available_views)
        if not requested_views:
            first_available = next(iter(available_views.keys()))
            print(
                f"[AVISO] Vistas solicitadas não disponíveis ({dataset_view_arg}). "
                f"Usando vista {first_available}."
            )
            requested_views = [first_available]

        selected_inputs = [
            {
                "label": f"view{view_id}",
                "path": available_views[view_id],
                "display": f"{args.dataset_item}_{view_id}",
            }
            for view_id in requested_views
        ]

        if not args.gt_mesh:
            dataset_gt = dataset_root / "models" / args.dataset_item / "meshes" / "model.obj"
            if dataset_gt.exists():
                args.gt_mesh = str(dataset_gt.resolve())
                print(f"[INFO] Usando ground-truth padrão do dataset: {args.gt_mesh}")
            else:
                print("[AVISO] Ground-truth do dataset não encontrado; métricas serão puladas.")

    if not selected_inputs:
        if not args.input:
            print("[ERRO] Informe --input ou utilize --dataset-item para selecionar uma amostra.")
            sys.exit(1)
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = project_root / input_path
        selected_inputs = [
            {
                "label": "input",
                "path": input_path.resolve(),
                "display": input_path.stem,
            }
        ]

    # Converter caminhos para absolutos
    for job in selected_inputs:
        job["path"] = job["path"].resolve()
    args.input = str(selected_inputs[0]["path"])
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    args.output = str(output_path.resolve())

    if args.gt_mesh:
        gt_path = Path(args.gt_mesh)
        if not gt_path.is_absolute():
            gt_path = project_root / gt_path
        args.gt_mesh = str(gt_path.resolve())
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        if not metrics_path.is_absolute():
            metrics_path = project_root / metrics_path
        args.metrics_out = str(metrics_path.resolve())
    
    # Ajustar caminho do config relativo ao Kiss3DGen
    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = (kiss3dgen_path / config_path).resolve()
        if candidate.exists():
            args.config = str(candidate)
        else:
            default_config = kiss3dgen_path / "pipeline" / "pipeline_config" / "default.yaml"
            if default_config.exists():
                args.config = str(default_config.resolve())
            else:
                print(f"[ERRO] Config nao encontrado: {args.config}")
                sys.exit(1)
    else:
        args.config = str(config_path.resolve())
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    print("=" * 60)
    print("Pipeline IMAGE TO 3D - Kiss3DGen")
    print("=" * 60)
    if len(selected_inputs) == 1:
        print(f"Input: {selected_inputs[0]['path']}")
    else:
        print("Inputs:")
        for job in selected_inputs:
            print(f"  - {job['label']}: {job['path']}")
    print(f"Output: {args.output}")
    print(f"Config: {args.config}")
    print(f"Redux: {args.enable_redux}")
    print(f"Use MV RGB: {args.use_mv_rgb}")
    print(f"Use ControlNet: {args.use_controlnet}")
    if args.dataset_item:
        view_desc = ", ".join(job["label"] for job in selected_inputs)
        print(f"Dataset item: {args.dataset_item} (vistas {view_desc})")
    if args.gt_mesh:
        print(f"GT mesh: {args.gt_mesh}")
    print("=" * 60)

    print("\n[0/4] Validando credenciais HuggingFace...")
    if ensure_hf_token(project_root / ".env"):
        print("[OK] Token HuggingFace carregado.")
    else:
        print("[AVISO] Token HuggingFace nao configurado; repositorios privados podem falhar.")

    if args.fast_mode:
        os.environ["KISS3D_FAST_MODE"] = "1"
        args.enable_redux = False
    
    # Verificar se arquivo existe
    if not os.path.exists(args.input):
        print(f"[ERRO] Arquivo nao encontrado: {args.input}")
        return
    
    # Verificar se config existe (agora relativo ao diretorio Kiss3DGen onde estamos)
    if not os.path.exists(args.config):
        print(f"[ERRO] Config nao encontrado: {args.config}")
        print(f"[INFO] Tentando usar config padrao do Kiss3DGen")
        default_config = Path('pipeline/pipeline_config/default.yaml')
        if default_config.exists():
            args.config = 'pipeline/pipeline_config/default.yaml'
        else:
            print(f"[ERRO] Config padrao tambem nao encontrado!")
            return
    
    # Inicializar wrapper do Kiss3DGen
    print("\n[1/4] Inicializando pipeline Kiss3DGen...")
    try:
        k3d_wrapper = init_wrapper_from_config(
            args.config,
            fast_mode=args.fast_mode,
            disable_llm=args.disable_llm,
            load_controlnet=args.use_controlnet,
            load_redux=args.enable_redux,
        )
        if k3d_wrapper.fast_mode and not args.fast_mode:
            print("[INFO] Fast mode ativado automaticamente (GPU <= 12GB detectada).")
            args.enable_redux = False
        print("[OK] Pipeline inicializado")
    except Exception as e:
        print(f"[ERRO] Falha ao inicializar pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[2/4] Preparando execuções por vista...")
    print(f"Total de vistas a processar: {len(selected_inputs)}")
    views_root = Path(args.output) / "views"
    views_root.mkdir(parents=True, exist_ok=True)
    history: List[Dict] = []
    job_results: List[Dict] = []

    for idx, job in enumerate(selected_inputs, start=1):
        print(f"\n[3/4] Executando vista {job['label']} ({idx}/{len(selected_inputs)})...")
        if os.path.exists(TMP_DIR):
            shutil.rmtree(TMP_DIR)
        os.makedirs(TMP_DIR, exist_ok=True)

        job_record = {
            "label": job["label"],
            "input": str(job["path"]),
            "index": idx,
            "success": False,
            "error": None,
            "metrics": None,
        }

        try:
            gen_save_path, recon_mesh_path = run_image_to_3d(
                k3d_wrapper,
                str(job["path"]),
                enable_redux=args.enable_redux,
                use_mv_rgb=args.use_mv_rgb,
                use_controlnet=args.use_controlnet,
            )
            job_record["success"] = True
            print(f"[OK] Vista {job['label']} concluída.")
        except Exception as exc:
            job_record["error"] = str(exc)
            print(f"[ERRO] Falha na vista {job['label']}: {exc}")
            import traceback

            traceback.print_exc()
            history.append(job_record)
            continue

        view_output_dir = views_root / job["label"]
        view_output_dir.mkdir(parents=True, exist_ok=True)

        bundle_copy = view_output_dir / f"{job['display']}_3d_bundle.png"
        mesh_copy = view_output_dir / f"{job['display']}.glb"
        if os.path.exists(gen_save_path):
            shutil.copy2(gen_save_path, bundle_copy)
        if os.path.exists(recon_mesh_path):
            shutil.copy2(recon_mesh_path, mesh_copy)

        metrics = None
        metrics_file = None
        if args.gt_mesh and os.path.exists(args.gt_mesh):
            try:
                metrics = evaluate_mesh_against_gt(
                    mesh_copy,
                    args.gt_mesh,
                    num_samples=args.gt_samples,
                    normalize=not args.disable_metrics_normalization,
                )
                metrics_file = view_output_dir / f"{job['label']}_metrics.json"
                metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                print(f"[OK] Métricas salvas em: {metrics_file}")
            except Exception as exc:
                print(f"[AVISO] Falha ao calcular métricas da vista {job['label']}: {exc}")

        job_results.append(
            {
                "label": job["label"],
                "display": job["display"],
                "bundle": bundle_copy,
                "mesh": mesh_copy,
                "metrics": metrics,
                "metrics_file": metrics_file,
                "order": idx,
            }
        )
        job_record["metrics"] = metrics
        history.append(job_record)

    history_path = Path(args.output) / "runs_report.json"
    history_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")

    if not job_results:
        print("[ERRO] Nenhuma vista foi processada com sucesso. Consulte runs_report.json para detalhes.")
        return

    def _score(result: Dict) -> Tuple[float, int]:
        if result["metrics"]:
            return (result["metrics"]["chamfer_l1"], result["order"])
        return (float("inf"), result["order"])

    best_result = min(job_results, key=_score)
    best_name = best_result["display"]

    print("\n[4/4] Consolidando melhor resultado...")
    final_bundle = Path(args.output) / f"{best_name}_3d_bundle.png"
    final_mesh = Path(args.output) / f"{best_name}.glb"
    if best_result["bundle"] and best_result["bundle"].exists():
        shutil.copy2(best_result["bundle"], final_bundle)
        print(f"[OK] Bundle selecionado: {final_bundle}")
    if best_result["mesh"] and best_result["mesh"].exists():
        shutil.copy2(best_result["mesh"], final_mesh)
        print(f"[OK] Mesh selecionada: {final_mesh}")

    final_metrics_path = None
    if best_result["metrics_file"] and best_result["metrics_file"].exists():
        target_metrics = Path(args.metrics_out) if args.metrics_out else Path(args.output) / f"{best_name}_metrics.json"
        target_metrics.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_result["metrics_file"], target_metrics)
        final_metrics_path = target_metrics
        print(f"[OK] Métricas consolidadas: {final_metrics_path}")

    summary = {
        "best_view": best_result["label"],
        "best_display": best_name,
        "metrics": best_result["metrics"],
        "all_runs": [
            {
                "label": result["label"],
                "display": result["display"],
                "metrics": result["metrics"],
                "bundle": str(result["bundle"]),
                "mesh": str(result["mesh"]),
            }
            for result in job_results
        ],
    }
    summary_path = Path(args.output) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[OK] Resultados salvos em: {args.output}")
    print(f"Melhor vista: {best_result['label']} -> {best_result['bundle']}")
    if final_metrics_path:
        print(f"Métricas finais: {final_metrics_path}")
    print("\n" + "=" * 60)
    print("Pipeline concluido!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    finally:
        os.chdir(str(ORIGINAL_WORKDIR))
