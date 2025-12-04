#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ruff: noqa: E402
"""
Pipeline IMAGE TO 3D baseado no Kiss3DGen
Reproduz exatamente o pipeline do artigo Kiss3DGen
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import subprocess

# Setup logging completo ANTES de qualquer outro import
from setup_logging import setup_complete_logging

pipeline_logger = logging.getLogger("kiss3dgen_pipeline")

# Adicionar ninja ao PATH antes de qualquer import
project_root = Path(__file__).parent.parent
venv_path = project_root / "mesh3d-generator-py3.11"
torch_ext_cache = project_root / "torch_extensions_cache"
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(torch_ext_cache))
os.makedirs(torch_ext_cache, exist_ok=True)
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
except Exception:
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


def _prepend_env_list(var_name: str, new_paths) -> None:
    if not new_paths:
        return
    existing_entries = [
        entry for entry in os.environ.get(var_name, "").split(os.pathsep) if entry
    ]
    changed = False
    for candidate in new_paths:
        candidate_str = str(candidate)
        if not candidate_str or not os.path.exists(candidate_str):
            continue
        if candidate_str not in existing_entries:
            existing_entries.insert(0, candidate_str)
            changed = True
    if changed and existing_entries:
        os.environ[var_name] = os.pathsep.join(existing_entries)
        print(f"[INFO] Variável {var_name} atualizada com diretórios MSVC/SDK.")


def _ensure_msvc_env(cl_path_str: str | None) -> None:
    if not cl_path_str:
        return
    cl_path = Path(cl_path_str)

    msvc_root = None
    for parent in cl_path.parents:
        candidate = parent
        if (candidate / "include").exists() and (candidate / "lib").exists():
            msvc_root = candidate
            break

    if msvc_root is None:
        print(f"[AVISO] Não foi possível localizar diretórios MSVC a partir de {cl_path}")
        return

    include_paths = [msvc_root / "include"]
    lib_paths = [msvc_root / "lib" / "x64"]

    kits_base = Path("C:/Program Files (x86)/Windows Kits/10")
    include_root = kits_base / "Include"
    lib_root = kits_base / "Lib"
    kit_version = None

    if include_root.exists():
        versions = sorted([p for p in include_root.iterdir() if p.is_dir()], reverse=True)
        if versions:
            kit_version = versions[0]
            include_paths.extend(
                kit_version / subdir for subdir in ("ucrt", "shared", "um", "winrt", "cppwinrt")
            )

    if kit_version and lib_root.exists():
        lib_version = lib_root / kit_version.name
        lib_paths.extend([
            lib_version / "ucrt" / "x64",
            lib_version / "um" / "x64",
        ])

    _prepend_env_list("INCLUDE", include_paths)
    _prepend_env_list("LIB", lib_paths)

# Configurar CUDA_HOME e PATH - CRÍTICO: Deve ser feito ANTES de qualquer import do torch
# Priorizar CUDA 12.1+ para compatibilidade com VS 2019/2022
cuda_base = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"
preferred_versions = ["v12.4", "v12.3", "v12.2", "v12.1", "v12.0"]  # Versões compatíveis com VS 2019/2022

cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
cuda_found = False

# Se CUDA_HOME já está configurado, verificar se é uma versão preferida
if cuda_home and os.path.exists(cuda_home):
    # Verificar se é uma versão preferida
    for version in preferred_versions:
        if version in cuda_home:
            cuda_found = True
            print(f"[INFO] CUDA_HOME já configurado com versão preferida: {cuda_home}")
            break
    
    # Se não é versão preferida, tentar encontrar uma melhor
    if not cuda_found:
        print(f"[AVISO] CUDA_HOME atual ({cuda_home}) não é versão preferida. Procurando versão melhor...")

if not cuda_found:
    # Tentar versões preferidas primeiro (compatíveis com VS 2019/2022)
    for version in preferred_versions:
        cuda_path = os.path.join(cuda_base, version)
        if os.path.exists(cuda_path) and os.path.exists(os.path.join(cuda_path, "bin", "nvcc.exe")):
            os.environ["CUDA_HOME"] = cuda_path
            os.environ["CUDA_PATH"] = cuda_path
            print(f"[INFO] CUDA_HOME configurado para {version} (compatível com VS 2019/2022): {cuda_path}")
            cuda_found = True
            break
    
    # Se ainda não encontrou, procurar qualquer versão instalada
    if not cuda_found and os.path.exists(cuda_base):
        for item in sorted(os.listdir(cuda_base), reverse=True):
            cuda_path = os.path.join(cuda_base, item)
            if os.path.isdir(cuda_path) and item.startswith("v") and os.path.exists(os.path.join(cuda_path, "bin", "nvcc.exe")):
                os.environ["CUDA_HOME"] = cuda_path
                os.environ["CUDA_PATH"] = cuda_path
                print(f"[INFO] CUDA_HOME configurado (versão encontrada): {cuda_path}")
                cuda_found = True
                break

if not cuda_found:
    print("[AVISO] CUDA_HOME nao encontrado. Compilação pode falhar.")
else:
    # IMPORTANTE: Adicionar CUDA bin ao INÍCIO do PATH para garantir que o nvcc correto seja usado
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        cuda_bin = os.path.join(cuda_home, "bin")
        if os.path.exists(cuda_bin):
            current_path = os.environ.get("PATH", "")
            # Remover qualquer outro CUDA do PATH primeiro
            path_parts = [p for p in current_path.split(os.pathsep) if "CUDA" not in p or cuda_bin in p]
            # Adicionar CUDA bin no INÍCIO do PATH
            os.environ["PATH"] = cuda_bin + os.pathsep + os.pathsep.join(path_parts)
            print(f"[INFO] CUDA bin adicionado ao INÍCIO do PATH: {cuda_bin}")
            
            # Verificar qual nvcc está sendo usado
            try:
                import subprocess
                nvcc_result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
                if nvcc_result.returncode == 0:
                    nvcc_output = nvcc_result.stdout
                    if cuda_home in nvcc_output or os.path.basename(cuda_home) in nvcc_output:
                        print(f"[OK] nvcc correto detectado: {nvcc_output.split(chr(10))[0] if chr(10) in nvcc_output else nvcc_output[:100]}")
                    else:
                        print(f"[AVISO] nvcc pode não ser da versão esperada. Output: {nvcc_output[:200]}")
            except Exception as e:
                print(f"[AVISO] Não foi possível verificar versão do nvcc: {e}")

# Configurar Visual Studio - PRIORIZAR VS 2019 (compatível com CUDA 12.1)
# CRÍTICO: Configurar ANTES de qualquer importação do PyTorch/distutils
vs_base_paths = [
    "C:\\Program Files (x86)\\Microsoft Visual Studio",
    "C:\\Program Files\\Microsoft Visual Studio"
]

vs_found = False
vs_preferred = ["2019"]  # VS 2019 é OBRIGATÓRIO para CUDA 12.1 - melhor compatibilidade
vs2019_vcvarsall = None
vs2019_cl_path = None

for vs_base in vs_base_paths:
    if os.path.exists(vs_base):
        for vs_version in vs_preferred:
            vs_path = os.path.join(vs_base, vs_version)
            if os.path.exists(vs_path):
                # Procurar por vcvarsall.bat primeiro (melhor método)
                for edition in ["BuildTools", "Community", "Professional", "Enterprise"]:
                    vcvarsall_path = os.path.join(vs_path, edition, "VC", "Auxiliary", "Build", "vcvarsall.bat")
                    if os.path.exists(vcvarsall_path):
                        vs2019_vcvarsall = vcvarsall_path
                        print(f"[INFO] VS 2019 vcvarsall.bat encontrado: {vcvarsall_path}")
                        break
                
                # Procurar por cl.exe no caminho típico
                for edition in ["BuildTools", "Community", "Professional", "Enterprise"]:
                    vc_tools = os.path.join(vs_path, edition, "VC", "Tools", "MSVC")
                    if os.path.exists(vc_tools):
                        # Procurar versão mais recente do MSVC
                        try:
                            msvc_versions = sorted([d for d in os.listdir(vc_tools) if os.path.isdir(os.path.join(vc_tools, d))], reverse=True)
                            if msvc_versions:
                                cl_path = os.path.join(vc_tools, msvc_versions[0], "bin", "Hostx64", "x64", "cl.exe")
                                if os.path.exists(cl_path):
                                    cl_dir = os.path.dirname(cl_path)
                                    vs2019_cl_path = cl_dir
                                    current_path = os.environ.get("PATH", "")
                                    # Remover qualquer VS 2022 do PATH primeiro
                                    path_parts = [p for p in current_path.split(os.pathsep) if "Visual Studio" not in p or "2019" in p]
                                    # Adicionar VS 2019 no INÍCIO do PATH
                                    if cl_dir not in path_parts:
                                        os.environ["PATH"] = cl_dir + os.pathsep + os.pathsep.join(path_parts)
                                    else:
                                        # Mover para o início
                                        path_parts = [p for p in path_parts if p != cl_dir]
                                        os.environ["PATH"] = cl_dir + os.pathsep + os.pathsep.join(path_parts)
                                    print(f"[INFO] VS 2019 cl.exe encontrado e adicionado ao INÍCIO do PATH: {cl_dir}")
                                    vs_found = True
                                    break
                        except Exception as e:
                            print(f"[AVISO] Erro ao procurar MSVC em {vc_tools}: {e}")
                if vs_found:
                    break
        if vs_found:
            break

if vs_found:
    # Configurar variáveis de ambiente do distutils para forçar VS 2019
    os.environ["DISTUTILS_USE_SDK"] = "1"
    # Configurar versão do toolset (14.29 = VS 2019)
    if vs2019_cl_path and "14.29" in vs2019_cl_path:
        os.environ["CMAKE_GENERATOR_TOOLSET_VERSION"] = "14.29"
    elif vs2019_cl_path and "14.27" in vs2019_cl_path:
        os.environ["CMAKE_GENERATOR_TOOLSET_VERSION"] = "14.27"
    else:
        os.environ["CMAKE_GENERATOR_TOOLSET_VERSION"] = "14.29"  # Default VS 2019
    
    # Se vcvarsall.bat foi encontrado, executar via subprocess para configurar ambiente
    if vs2019_vcvarsall:
        try:
            import subprocess
            # Executar vcvarsall.bat e capturar variáveis de ambiente
            # Usar cmd /c para executar o batch e exportar variáveis
            cmd = f'cmd /c "{vs2019_vcvarsall}" x64 && set'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Parsear variáveis de ambiente do output
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if '=' in line and not line.startswith('_'):
                        key, value = line.split('=', 1)
                        key_upper = key.upper()
                        # Atualizar PATH e outras variáveis importantes (case-insensitive)
                        if key_upper == 'PATH':
                            os.environ['PATH'] = value
                        elif key_upper in {"INCLUDE", "LIB", "LIBPATH"}:
                            os.environ[key_upper] = value
                print("[INFO] VS 2019 ambiente configurado via vcvarsall.bat")
                print(f"[DEBUG] INCLUDE={os.environ.get('INCLUDE', '<none>')}")
                print(f"[DEBUG] LIB={os.environ.get('LIB', '<none>')}")
        except Exception as e:
            print(f"[AVISO] Não foi possível executar vcvarsall.bat: {e}")
            print("[INFO] Continuando com configuração manual do PATH")

    _ensure_msvc_env(vs2019_cl_path)
    print(f"[DEBUG] INCLUDE(after fix)={os.environ.get('INCLUDE', '<none>')}")
    print(f"[DEBUG] LIB(after fix)={os.environ.get('LIB', '<none>')}")

    print("[OK] VS 2019 configurado corretamente")
else:
    print("[ERRO] VS 2019 não encontrado! Compilação vai falhar.")
    print("[INFO] Instale Visual Studio 2019 Build Tools:")
    print("  https://visualstudio.microsoft.com/vs/older-downloads/")
    print("  Selecione: 'Desktop development with C++'")

# Configurar TORCH_CUDA_ARCH_LIST - deixar vazio para auto-detectar, mas garantir que não tenha valores antigos
# O PyTorch vai detectar automaticamente a arquitetura da GPU disponível
if "TORCH_CUDA_ARCH_LIST" in os.environ:
    old_arch = os.environ["TORCH_CUDA_ARCH_LIST"]
    if old_arch and old_arch.strip():
        print(f"[INFO] TORCH_CUDA_ARCH_LIST estava configurado como '{old_arch}', limpando para auto-detecção")
    os.environ["TORCH_CUDA_ARCH_LIST"] = ""  # PyTorch vai auto-detectar
else:
    os.environ["TORCH_CUDA_ARCH_LIST"] = ""  # PyTorch vai auto-detectar
    print("[INFO] TORCH_CUDA_ARCH_LIST configurado para auto-detecção")

# Adicionar paths - IMPORTANTE: Kiss3DGen precisa estar no path
kiss3dgen_path = project_root / "Kiss3DGen"
if not kiss3dgen_path.exists():
    print(f"[ERRO] Kiss3DGen nao encontrado em: {kiss3dgen_path}")
    print("[INFO] Certifique-se de que o diretorio Kiss3DGen existe")
    sys.exit(1)

sys.path.insert(0, str(project_root))

import argparse
import shutil

from kiss3d_utils_local import (
    TMP_DIR,
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

def _resolve_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _run_dataset_plan(args) -> bool:
    plan_path = _resolve_path(args.dataset_plan)
    if not plan_path.exists():
        print(f"[ERRO] Arquivo de dataset plan não encontrado: {plan_path}")
        return False

    try:
        plan_data = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERRO] Falha ao ler dataset plan {plan_path}: {exc}")
        return False

    if isinstance(plan_data, dict):
        objects = plan_data.get("objects", [])
    elif isinstance(plan_data, list):
        objects = plan_data
    else:
        objects = []

    if not objects:
        print(f"[ERRO] Dataset plan {plan_path} não contém objetos.")
        return False

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    base_output = Path(args.output or "data/outputs/kiss3dgen_plan")
    if not base_output.is_absolute():
        base_output = project_root / base_output
    base_output.mkdir(parents=True, exist_ok=True)

    if args.metrics_out:
        print("[AVISO] '--metrics-out' será ignorado durante a execução do dataset plan; cada objeto salvará métricas localmente.")

    config_path = _resolve_path(args.config)
    print(f"[PLAN] Carregando modelos uma única vez a partir de {config_path} ...")
    try:
        k3d_wrapper = init_wrapper_from_config(
            str(config_path),
            fast_mode=args.fast_mode,
            disable_llm=args.disable_llm,
            load_controlnet=args.use_controlnet,
            load_redux=args.enable_redux,
        )
    except Exception as exc:
        print(f"[ERRO] Falha ao inicializar modelos para o dataset plan: {exc}")
        return False

    success_all = True

    for idx, entry in enumerate(objects, 1):
        name = entry.get("name")
        if not name:
            print(f"[AVISO] Objeto #{idx} no dataset plan não possui campo 'name'. Ignorando.")
            continue

        entry_mode = entry.get("pipeline_mode") or args.pipeline_mode
        if entry_mode not in {"flux", "multiview"}:
            entry_mode = args.pipeline_mode

        view_spec = entry.get("view", args.dataset_view if args.dataset_view is not None else 0)
        if isinstance(view_spec, (list, tuple)):
            view_arg = ",".join(str(v) for v in view_spec)
        else:
            view_arg = str(view_spec)

        custom_output = base_output / name
        custom_output.mkdir(parents=True, exist_ok=True)
        views_root = custom_output / "views"
        views_root.mkdir(parents=True, exist_ok=True)

        history_path = custom_output / "runs_report.json"
        summary_path = custom_output / "summary.json"
        log_file = custom_output / "plan_run.log"

        print(f"\n[PLAN] ({idx}/{len(objects)}) Executando {name} (views {view_arg}) modo={entry_mode}...")
        available_views = _collect_dataset_views(dataset_root, name)
        if not available_views:
            print(f"[ERRO] Nenhuma imagem encontrada para {name} em {dataset_root / 'images'}.")
            success_all = False
            continue

        requested_views = _parse_view_argument(view_arg, available_views)
        if not requested_views:
            first_available = next(iter(available_views.keys()))
            print(f"[AVISO] Vistas solicitadas não disponíveis ({view_arg}). Usando vista {first_available}.")
            requested_views = [first_available]

        selected_inputs = [
            {
                "label": f"view{view_id}",
                "path": available_views[view_id],
                "display": f"{name}_{view_id}",
            }
            for view_id in requested_views
        ]

        gt_mesh_path = entry.get("gt_mesh") or args.gt_mesh
        if gt_mesh_path:
            gt_mesh_path = str(_resolve_path(gt_mesh_path))
            if not Path(gt_mesh_path).exists():
                print(f"[AVISO] GT mesh informada não encontrada ({gt_mesh_path}). Ignorando.")
                gt_mesh_path = None
        if not gt_mesh_path:
            dataset_gt = dataset_root / "models" / name / "meshes" / "model.obj"
            if dataset_gt.exists():
                gt_mesh_path = str(dataset_gt.resolve())

        history_path.write_text(json.dumps([], indent=2), encoding="utf-8")
        history: List[Dict] = []
        job_results: List[Dict] = []
        plan_success = False

        for view_idx, job in enumerate(selected_inputs, start=1):
            print(f"\n[PLAN] -> Vista {job['label']} ({view_idx}/{len(selected_inputs)})")
            # Não remover a pasta TMP para preservar todos os intermediários.
            os.makedirs(TMP_DIR, exist_ok=True)

            job_record = {
                "label": job["label"],
                "input": str(job["path"]),
                "index": view_idx,
                "success": False,
                "error": None,
                "metrics": None,
                "pipeline_mode": entry_mode,
            }

            try:
                gen_save_path, recon_mesh_path = run_image_to_3d(
                    k3d_wrapper,
                    str(job["path"]),
                    enable_redux=args.enable_redux,
                    use_mv_rgb=args.use_mv_rgb,
                    use_controlnet=args.use_controlnet,
                    pipeline_mode=entry_mode,
                )
                job_record["success"] = True
                plan_success = True
                print(f"[OK] Vista {job['label']} concluída.")
            except Exception as exc:
                job_record["error"] = str(exc)
                history.append(job_record)
                print(f"[ERRO] Falha na vista {job['label']}: {exc}")
                pipeline_logger.error(f"[PLAN] Falha em {name} {job['label']}: {exc}")
                success_all = False
                continue

            view_output_dir = views_root / job["label"]
            view_output_dir.mkdir(parents=True, exist_ok=True)

            bundle_copy = None
            mesh_copy = view_output_dir / f"{job['display']}.glb"

            if gen_save_path and os.path.exists(gen_save_path):
                bundle_copy = view_output_dir / f"{job['display']}_3d_bundle.png"
                shutil.copy2(gen_save_path, bundle_copy)
            if recon_mesh_path and os.path.exists(recon_mesh_path):
                shutil.copy2(recon_mesh_path, mesh_copy)
            else:
                mesh_copy = None

            metrics = None
            metrics_file = None
            if gt_mesh_path and mesh_copy and os.path.exists(gt_mesh_path):
                try:
                    metrics = evaluate_mesh_against_gt(
                        mesh_copy,
                        gt_mesh_path,
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
                    "bundle": str(bundle_copy) if bundle_copy else None,
                    "mesh": str(mesh_copy) if mesh_copy else None,
                    "metrics": metrics,
                    "metrics_file": str(metrics_file) if metrics_file else None,
                    "order": view_idx,
                    "pipeline_mode": entry_mode,
                }
            )
            job_record["metrics"] = metrics
            history.append(job_record)

            try:
                history_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
                pipeline_logger.debug(f"[PLAN] Histórico {name}: {len(history)} registros")
            except Exception as exc:
                pipeline_logger.error(f"[PLAN] Erro ao salvar histórico parcial ({name}): {exc}")

        if not plan_success:
            print(f"[PLAN] {name} não gerou resultados válidos. Veja {history_path}.")
            continue

        try:
            history_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            print(f"[ERRO] Falha ao salvar histórico final de {name}: {exc}")

        def _score(result: Dict) -> Tuple[float, int]:
            if result["metrics"]:
                return (result["metrics"]["chamfer_l1"], result["order"])
            return (float("inf"), result["order"])

        best_result = min(job_results, key=_score)
        best_name = best_result["display"]
        final_bundle = custom_output / f"{best_name}_3d_bundle.png"
        final_mesh = custom_output / f"{best_name}.glb"

        best_bundle_path = Path(best_result["bundle"]) if best_result["bundle"] else None
        best_mesh_path = Path(best_result["mesh"]) if best_result["mesh"] else None
        best_metrics_path = Path(best_result["metrics_file"]) if best_result["metrics_file"] else None

        if best_bundle_path and best_bundle_path.exists():
            shutil.copy2(best_bundle_path, final_bundle)
            print(f"[PLAN] Bundle selecionado: {final_bundle}")
        if best_mesh_path and best_mesh_path.exists():
            shutil.copy2(best_mesh_path, final_mesh)
            print(f"[PLAN] Mesh selecionada: {final_mesh}")

        final_metrics_path = None
        if best_metrics_path and best_metrics_path.exists():
            target_metrics = custom_output / f"{best_name}_metrics.json"
            shutil.copy2(best_metrics_path, target_metrics)
            final_metrics_path = target_metrics
            print(f"[PLAN] Métricas consolidadas: {final_metrics_path}")

        summary = {
            "dataset_item": name,
            "pipeline_mode": entry_mode,
            "best_view": best_result["label"],
            "best_display": best_name,
            "metrics": best_result["metrics"],
            "all_runs": job_results,
        }
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

        print(f"[PLAN] {name} concluído. Resultados em {custom_output}")
        print(f"        Histórico: {history_path}")
        print(f"        Resumo: {summary_path}")
        print(f"        Logs: {log_file}")

    return success_all


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
    parser.add_argument("--dataset-item", type=str, default="OXO_Cookie_Spatula", help="Nome do item em data/raw/gazebo_dataset (padrão: OXO_Cookie_Spatula)")
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
        "--dataset-plan",
        type=str,
        help="Arquivo YAML descrevendo uma lista de objetos (name/view) para processar sequencialmente",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=["flux", "multiview"],
        default="flux",
        help="Escolhe entre reconstrução baseada em Flux (bundle) ou Zero123++ multiview",
    )
    parser.add_argument(
        "--precompile-nvdiffrast",
        action="store_true",
        help="Pré-compila o nvdiffrast antes de executar o pipeline (recomendado na primeira execução)",
    )
    parser.add_argument(
        "--disable-metrics-normalization",
        action="store_true",
        help="Não normaliza as malhas antes de calcular as métricas (por padrão normalizamos).",
    )
    
    args = parser.parse_args()
    
    if args.dataset_plan:
        success = _run_dataset_plan(args)
        sys.exit(0 if success else 1)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    selected_inputs: List[Dict[str, Path]] = []
    dataset_view_arg = str(args.dataset_view) if args.dataset_view is not None else ""

    if args.input:
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
    elif args.dataset_item:
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
    else:
        print("[ERRO] Informe --input ou utilize --dataset-item para selecionar uma amostra.")
        sys.exit(1)

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
    
    # Setup logging completo
    log_dir = Path(args.output) / "logs"
    global pipeline_logger
    pipeline_logger, log_file = setup_complete_logging(
        log_dir=log_dir,
        log_level=logging.DEBUG,
        capture_warnings=True,
        capture_stdout=True
    )
    
    # Mostrar claramente onde o log está sendo salvo
    print("=" * 80)
    print(f"LOGS: {log_file}")
    print("=" * 80)
    pipeline_logger.info(f"Logs sendo salvos em: {log_file}")
    
    # Redirecionar print para logger também
    import builtins
    original_print = builtins.print
    def logged_print(*args, **kwargs):
        original_print(*args, **kwargs)
        pipeline_logger.info(" ".join(str(arg) for arg in args))
    builtins.print = logged_print
    
    print("=" * 60)
    print("Pipeline IMAGE TO 3D - Kiss3DGen")
    print("=" * 60)
    pipeline_logger.info("=" * 60)
    pipeline_logger.info("Pipeline IMAGE TO 3D - Kiss3DGen")
    pipeline_logger.info("=" * 60)
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
    
    # Pré-compilar nvdiffrast se solicitado
    if args.precompile_nvdiffrast:
        print("\n[0.5/4] Pré-compilando nvdiffrast...")
        try:
            precompile_script = project_root / "scripts" / "precompile_nvdiffrast.py"
            if precompile_script.exists():
                result = subprocess.run(
                    [sys.executable, str(precompile_script)],
                    cwd=str(project_root),
                    check=False,
                )
                if result.returncode != 0:
                    print("[AVISO] Pré-compilação falhou ou foi cancelada.")
                    print("[INFO] O pipeline tentará compilar durante a execução se necessário.")
                else:
                    print("[OK] Pré-compilação concluída com sucesso!")
            else:
                print(f"[AVISO] Script de pré-compilação não encontrado: {precompile_script}")
        except Exception as e:
            print(f"[AVISO] Erro ao executar pré-compilação: {e}")
            print("[INFO] O pipeline tentará compilar durante a execução se necessário.")

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
        print("[INFO] Tentando usar config padrao do Kiss3DGen")
        default_config = Path('pipeline/pipeline_config/default.yaml')
        if default_config.exists():
            args.config = 'pipeline/pipeline_config/default.yaml'
        else:
            print("[ERRO] Config padrao tambem nao encontrado!")
            return
    
    # Inicializar wrapper do Kiss3DGen
    print("\n[1/4] Inicializando pipeline Kiss3DGen...")
    k3d_wrapper = None
    try:
        k3d_wrapper = init_wrapper_from_config(
            args.config,
            fast_mode=args.fast_mode,
            disable_llm=args.disable_llm,
            load_controlnet=args.use_controlnet,
            load_redux=args.enable_redux,
            pipeline_mode=args.pipeline_mode,  # CORRIGIDO: passar pipeline_mode
        )
        if k3d_wrapper.fast_mode and not args.fast_mode:
            print("[INFO] Fast mode ativado automaticamente (GPU <= 12GB detectada).")
            args.enable_redux = False
        print("[OK] Pipeline inicializado")
    except Exception as e:
        error_msg = f"Falha ao inicializar pipeline: {e}"
        print(f"[ERRO] {error_msg}")
        pipeline_logger.error(error_msg, exc_info=True)
        import traceback
        traceback.print_exc()
        # Salvar erro no histórico antes de retornar
        error_record = {
            "label": "initialization",
            "input": str(args.input) if args.input else "N/A",
            "index": 0,
            "success": False,
            "error": str(e),
            "metrics": None,
        }
        history_path = Path(args.output) / "runs_report.json"
        try:
            history_path.write_text(json.dumps([error_record], indent=2, default=str), encoding="utf-8")
            print(f"[INFO] Erro salvo em: {history_path}")
        except Exception:
            pass
        return
    
    print("\n[2/4] Preparando execuções por vista...")
    print(f"Total de vistas a processar: {len(selected_inputs)}")
    views_root = Path(args.output) / "views"
    views_root.mkdir(parents=True, exist_ok=True)
    history: List[Dict] = []
    job_results: List[Dict] = []
    
    # Garantir que o diretório de output existe
    history_path = Path(args.output) / "runs_report.json"
    summary_path = Path(args.output) / "summary.json"
    
    # Salvar arquivo inicial vazio para garantir que existe
    history_path.write_text(json.dumps([], indent=2), encoding="utf-8")
    print(f"[INFO] Arquivo de histórico criado: {history_path}")

    for idx, job in enumerate(selected_inputs, start=1):
        print(f"\n[3/4] Executando vista {job['label']} ({idx}/{len(selected_inputs)})...")
        # Preservar a pasta TMP entre execuções para manter todos os artefatos.
        os.makedirs(TMP_DIR, exist_ok=True)

        job_record = {
            "label": job["label"],
            "input": str(job["path"]),
            "index": idx,
            "success": False,
            "error": None,
            "metrics": None,
            "pipeline_mode": args.pipeline_mode,
        }

        try:
            gen_save_path, recon_mesh_path = run_image_to_3d(
                k3d_wrapper,
                str(job["path"]),
                enable_redux=args.enable_redux,
                use_mv_rgb=args.use_mv_rgb,
                use_controlnet=args.use_controlnet,
                pipeline_mode=args.pipeline_mode,
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

        bundle_copy = None
        mesh_copy = None

        if gen_save_path and os.path.exists(gen_save_path):
            bundle_copy = view_output_dir / f"{job['display']}_3d_bundle.png"
            shutil.copy2(gen_save_path, bundle_copy)
        if recon_mesh_path and os.path.exists(recon_mesh_path):
            mesh_copy = view_output_dir / f"{job['display']}.glb"
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
                "pipeline_mode": args.pipeline_mode,
            }
        )
        job_record["metrics"] = metrics
        history.append(job_record)
        
        # Salvar histórico incrementalmente após cada job
        try:
            history_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
            pipeline_logger.debug(f"Histórico atualizado: {len(history)} registros")
        except Exception as e:
            pipeline_logger.error(f"Erro ao salvar histórico: {e}")

    # Salvar histórico final
    try:
        history_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
        print(f"[OK] Histórico completo salvo em: {history_path}")
        pipeline_logger.info(f"Histórico completo salvo: {history_path}")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar histórico: {e}")
        pipeline_logger.error(f"Falha ao salvar histórico: {e}")

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
    bundle_source = best_result["bundle"]
    mesh_source = best_result["mesh"]
    metrics_source = best_result["metrics_file"]

    if bundle_source and Path(bundle_source).exists():
        shutil.copy2(bundle_source, final_bundle)
        print(f"[OK] Bundle selecionado: {final_bundle}")
    if mesh_source and Path(mesh_source).exists():
        shutil.copy2(mesh_source, final_mesh)
        print(f"[OK] Mesh selecionada: {final_mesh}")

    final_metrics_path = None
    if metrics_source and Path(metrics_source).exists():
        target_metrics = Path(args.metrics_out) if args.metrics_out else Path(args.output) / f"{best_name}_metrics.json"
        target_metrics.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metrics_source, target_metrics)
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
                "bundle": str(result["bundle"]) if result["bundle"] else None,
                "mesh": str(result["mesh"]) if result["mesh"] else None,
            }
            for result in job_results
        ],
    }
    summary_path = Path(args.output) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[OK] Resultados salvos em: {args.output}")
    print(f"  - Histórico: {history_path}")
    print(f"  - Resumo: {summary_path}")
    print(f"  - Logs: {log_file}")
    print(f"Melhor vista: {best_result['label']} -> {best_result['bundle']}")
    if final_metrics_path:
        print(f"Métricas finais: {final_metrics_path}")
    print("\n" + "=" * 60)
    print("Pipeline concluido!")
    print("=" * 60)
    pipeline_logger.info(f"Pipeline concluído. Resultados em: {args.output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        os.chdir(str(ORIGINAL_WORKDIR))
