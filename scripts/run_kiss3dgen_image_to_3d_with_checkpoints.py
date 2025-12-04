#!/usr/bin/env python3
"""
Pipeline IMAGE TO 3D - Kiss3DGen com Sistema de Checkpoints
Salva resultados intermediários em cada etapa para diagnóstico
"""

import os
import sys
from pathlib import Path

# CRÍTICO: Configurar logging IMEDIATAMENTE para capturar tudo
import logging
import warnings

# Suprimir warnings comuns e não críticos ANTES de qualquer importação
warnings.filterwarnings('ignore', message='.*Some weights of.*were not used.*')
warnings.filterwarnings('ignore', message='.*text_projection.*')
warnings.filterwarnings('ignore', message='.*add_prefix_space.*')
warnings.filterwarnings('ignore', message='.*The tokenizer.*needs to be converted.*')
warnings.filterwarnings('ignore', message='.*TRANSFORMERS_CACHE.*')
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')
warnings.filterwarnings('ignore', message='.*_get_vc_env is private.*')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Saída imediata para stdout
    ]
)
logger = logging.getLogger(__name__)
logger.info("="*80)
logger.info("INICIANDO PIPELINE - Kiss3DGen Image to 3D")
logger.info("="*80)

# CRÍTICO: Configurar VS 2019 e CUDA ANTES de qualquer importação do PyTorch
project_root = Path(__file__).parent.parent
logger.info(f"[INIT] Project root: {project_root}")

# Configurar CUDA_HOME - PRIORIZAR CUDA 12.1 (compatível com PyTorch 2.5.1+cu121)
cuda_base = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"
cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

# Verificar se CUDA_HOME atual é compatível com PyTorch (12.1+)
preferred_versions = ["v12.1", "v12.2", "v12.3", "v12.4", "v12.0"]
cuda_found = False

# Se CUDA_HOME já está configurado e é versão preferida, usar
if cuda_home and os.path.exists(cuda_home):
    for version in preferred_versions:
        if version in cuda_home:
            cuda_found = True
            logger.info(f"[CUDA] CUDA_HOME já configurado com versão preferida: {cuda_home}")
            break

# Se não encontrou versão preferida, procurar
if not cuda_found and os.path.exists(cuda_base):
    # Primeiro tentar versões preferidas (12.x)
    for version in preferred_versions:
        cuda_path = os.path.join(cuda_base, version)
        if os.path.exists(cuda_path) and os.path.exists(os.path.join(cuda_path, "bin", "nvcc.exe")):
            os.environ["CUDA_HOME"] = cuda_path
            os.environ["CUDA_PATH"] = cuda_path
            logger.info(f"[CUDA] CUDA_HOME configurado para {version} (compatível com PyTorch): {cuda_path}")
            cuda_found = True
            break
    
    # Se não encontrou 12.x, usar qualquer CUDA disponível
    if not cuda_found:
        for item in sorted(os.listdir(cuda_base), reverse=True):
            cuda_path = os.path.join(cuda_base, item)
            if os.path.isdir(cuda_path) and item.startswith("v") and os.path.exists(os.path.join(cuda_path, "bin", "nvcc.exe")):
                os.environ["CUDA_HOME"] = cuda_path
                os.environ["CUDA_PATH"] = cuda_path
                logger.warning(f"[CUDA] CUDA_HOME configurado para {item} (pode não ser compatível com PyTorch): {cuda_path}")
                logger.warning(f"[CUDA] PyTorch foi compilado com CUDA 12.1 - considere instalar CUDA 12.1")
                break

# Adicionar CUDA bin ao PATH
cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if cuda_home:
    cuda_bin = os.path.join(cuda_home, "bin")
    if os.path.exists(cuda_bin):
        current_path = os.environ.get("PATH", "")
        path_parts = [p for p in current_path.split(os.pathsep) if "CUDA" not in p or cuda_bin in p]
        os.environ["PATH"] = cuda_bin + os.pathsep + os.pathsep.join(path_parts)
        logger.info(f"[CUDA] CUDA bin adicionado ao PATH: {cuda_bin}")

# Configurar VS 2019 ANTES de qualquer importação - USAR VCVARSALL.BAT
vs_base_paths = [
    "C:\\Program Files (x86)\\Microsoft Visual Studio",
    "C:\\Program Files\\Microsoft Visual Studio"
]

vs_found = False
vs_preferred = ["2019"]
vs2019_vcvarsall = None

# Primeiro tentar encontrar vcvarsall.bat
for vs_base in vs_base_paths:
    if os.path.exists(vs_base):
        for vs_version in vs_preferred:
            vs_path = os.path.join(vs_base, vs_version)
            if os.path.exists(vs_path):
                for edition in ["BuildTools", "Community", "Professional", "Enterprise"]:
                    vcvarsall_path = os.path.join(vs_path, edition, "VC", "Auxiliary", "Build", "vcvarsall.bat")
                    if os.path.exists(vcvarsall_path):
                        vs2019_vcvarsall = vcvarsall_path
                        print(f"[INFO] VS 2019 vcvarsall.bat encontrado: {vcvarsall_path}")
                        vs_found = True
                        break
                if vs_found:
                    break
        if vs_found:
            break

# Se encontrou vcvarsall, executar para configurar ambiente completo
if vs2019_vcvarsall:
    try:
        import subprocess
        import tempfile
        
        # Criar script batch temporário para executar vcvarsall e exportar variáveis
        # IMPORTANTE: Usar encoding='utf-8' e capturar TODAS as variáveis
        temp_bat = tempfile.NamedTemporaryFile(mode='w', suffix='.bat', delete=False, encoding='utf-8')
        temp_bat.write('@echo off\n')
        temp_bat.write(f'call "{vs2019_vcvarsall}" x64 >nul 2>&1\n')
        temp_bat.write('set\n')  # Exportar todas as variáveis
        temp_bat.close()
        
        # Executar o batch e capturar output
        result = subprocess.run(
            ['cmd', '/c', temp_bat.name],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=20,
            cwd=os.path.expanduser('~')
        )
        
        # Limpar arquivo temporário
        try:
            os.unlink(temp_bat.name)
        except:
            pass
        
        if result.returncode == 0:
            # Parsear variáveis de ambiente do output - CRÍTICO para compilação
            include_paths = []
            lib_paths = []
            path_paths = []
            
            for line in result.stdout.split('\n'):
                line = line.strip()
                if '=' in line and not line.startswith('_') and not line.startswith('PROMPT') and not line.startswith('_='):
                    try:
                        key, value = line.split('=', 1)
                        # Capturar variáveis críticas
                        if key == 'PATH':
                            path_paths = [p.strip() for p in value.split(';') if p.strip()]
                        elif key == 'INCLUDE':
                            include_paths = [p.strip() for p in value.split(';') if p.strip()]
                        elif key == 'LIB':
                            lib_paths = [p.strip() for p in value.split(';') if p.strip()]
                        elif key == 'LIBPATH':
                            lib_paths.extend([p.strip() for p in value.split(';') if p.strip()])
                    except:
                        pass
            
            # Configurar variáveis de ambiente - CRÍTICO para encontrar headers C++
            if include_paths:
                current_include = os.environ.get("INCLUDE", "")
                current_include_list = [p.strip() for p in current_include.split(';') if p.strip()] if current_include else []
                # REMOVER paths do VS 2022 (incompatível com CUDA 12.1)
                current_include_list = [p for p in current_include_list if "Visual Studio\\2022" not in p and "MSVC\\14.4" not in p]
                # Adicionar paths do VS 2019 no início
                for inc_path in include_paths:
                    if inc_path not in current_include_list and os.path.exists(inc_path):
                        current_include_list.insert(0, inc_path)
                os.environ["INCLUDE"] = ';'.join(current_include_list)
                print(f"[INFO] INCLUDE configurado com {len(include_paths)} diretórios (VS 2022 removido)")
            
            if lib_paths:
                current_lib = os.environ.get("LIB", "")
                current_lib_list = [p.strip() for p in current_lib.split(';') if p.strip()] if current_lib else []
                # REMOVER paths do VS 2022 (incompatível com CUDA 12.1)
                current_lib_list = [p for p in current_lib_list if "Visual Studio\\2022" not in p and "MSVC\\14.4" not in p]
                # Adicionar paths do VS 2019 no início
                for lib_path in lib_paths:
                    if lib_path not in current_lib_list and os.path.exists(lib_path):
                        current_lib_list.insert(0, lib_path)
                os.environ["LIB"] = ';'.join(current_lib_list)
                print(f"[INFO] LIB configurado com {len(lib_paths)} diretórios (VS 2022 removido)")
            
            if path_paths:
                current_path = os.environ.get("PATH", "")
                current_path_list = [p.strip() for p in current_path.split(os.pathsep) if p.strip()]
                # REMOVER paths do VS 2022 (incompatível com CUDA 12.1)
                current_path_list = [p for p in current_path_list if "Visual Studio\\2022" not in p and "MSVC\\14.4" not in p]
                # Adicionar paths do VS 2019 no início (remover duplicatas)
                for path_item in path_paths:
                    if path_item not in current_path_list:
                        current_path_list.insert(0, path_item)
                os.environ["PATH"] = os.pathsep.join(current_path_list)
                print(f"[INFO] PATH atualizado com {len(path_paths)} diretórios do VS 2019 (VS 2022 removido)")
            
            # Variáveis adicionais críticas
            os.environ["DISTUTILS_USE_SDK"] = "1"
            os.environ["VSCMD_SKIP_SENDTELEMETRY"] = "1"
            
            # CRÍTICO: Garantir que cl.exe do VS 2019 está no PATH antes do VS 2022
            # nvcc procura cl.exe no PATH, então precisamos garantir ordem correta
            vs2019_cl_path = None
            for path_item in path_paths:
                cl_exe = os.path.join(path_item, "cl.exe")
                if os.path.exists(cl_exe) and "2019" in path_item:
                    vs2019_cl_path = path_item
                    break
            
            if vs2019_cl_path:
                # Garantir que VS 2019 cl.exe está ANTES de qualquer VS 2022 no PATH
                current_path = os.environ.get("PATH", "")
                path_list = [p.strip() for p in current_path.split(os.pathsep) if p.strip()]
                # Remover VS 2022
                path_list = [p for p in path_list if "Visual Studio\\2022" not in p and "MSVC\\14.4" not in p]
                # Garantir que VS 2019 está no início
                if vs2019_cl_path in path_list:
                    path_list.remove(vs2019_cl_path)
                path_list.insert(0, vs2019_cl_path)
                os.environ["PATH"] = os.pathsep.join(path_list)
                print(f"[INFO] cl.exe do VS 2019 priorizado no PATH: {vs2019_cl_path}")
            
            # Verificar qual cl.exe será usado
            try:
                import subprocess
                which_cl = subprocess.run(["where", "cl"], capture_output=True, text=True, shell=True, timeout=5)
                if which_cl.returncode == 0 and which_cl.stdout:
                    cl_path = which_cl.stdout.strip().split('\n')[0]
                    if "2019" in cl_path:
                        print(f"[OK] cl.exe do VS 2019 será usado: {cl_path}")
                    elif "2022" in cl_path:
                        print(f"[ERRO] cl.exe do VS 2022 ainda está sendo usado: {cl_path}")
                        print(f"[INFO] Tentando forçar remoção do VS 2022...")
                        # Remover VS 2022 do PATH novamente
                        path_list = [p for p in os.environ.get("PATH", "").split(os.pathsep) if "Visual Studio\\2022" not in p and "MSVC\\14.4" not in p]
                        os.environ["PATH"] = os.pathsep.join(path_list)
                    else:
                        print(f"[AVISO] cl.exe encontrado em local inesperado: {cl_path}")
            except Exception as e:
                print(f"[AVISO] Não foi possível verificar qual cl.exe será usado: {e}")
            
            # Verificar se headers padrão estão acessíveis
            test_header = None
            for inc_path in include_paths[:3]:  # Verificar primeiros 3 paths
                test_header_path = os.path.join(inc_path, "cstddef")
                if os.path.exists(test_header_path):
                    test_header = test_header_path
                    break
            
            if test_header:
                print(f"[OK] Headers C++ padrão encontrados: {test_header}")
            else:
                print(f"[AVISO] Headers C++ padrão não encontrados nos primeiros paths")
                print(f"[INFO] Verificando manualmente...")
                # Tentar encontrar manualmente
                for inc_path in include_paths:
                    if "MSVC" in inc_path and os.path.exists(inc_path):
                        test_files = ["cstddef", "limits.h", "iostream"]
                        found = sum(1 for f in test_files if os.path.exists(os.path.join(inc_path, f)))
                        if found > 0:
                            print(f"[OK] Encontrados {found}/{len(test_files)} headers em {inc_path}")
            
            print(f"[OK] VS 2019 ambiente configurado via vcvarsall.bat")
        else:
            print(f"[AVISO] vcvarsall.bat retornou código {result.returncode}")
            print(f"[INFO] Tentando configuração manual do PATH")
            vs_found = False
            # Fallback: configuração manual
            for vs_base in vs_base_paths:
                if os.path.exists(vs_base):
                    for vs_version in vs_preferred:
                        vs_path = os.path.join(vs_base, vs_version)
                        if os.path.exists(vs_path):
                            for edition in ["BuildTools", "Community", "Professional", "Enterprise"]:
                                vc_tools = os.path.join(vs_path, edition, "VC", "Tools", "MSVC")
                                if os.path.exists(vc_tools):
                                    try:
                                        msvc_versions = sorted([d for d in os.listdir(vc_tools) if os.path.isdir(os.path.join(vc_tools, d))], reverse=True)
                                        if msvc_versions:
                                            cl_path = os.path.join(vc_tools, msvc_versions[0], "bin", "Hostx64", "x64", "cl.exe")
                                            if os.path.exists(cl_path):
                                                cl_dir = os.path.dirname(cl_path)
                                                include_dir = os.path.join(vc_tools, msvc_versions[0], "include")
                                                lib_dir = os.path.join(vc_tools, msvc_versions[0], "lib", "x64")
                                                
                                                current_path = os.environ.get("PATH", "")
                                                path_parts = [p for p in current_path.split(os.pathsep) if "Visual Studio" not in p or "2019" in p]
                                                if cl_dir not in path_parts:
                                                    os.environ["PATH"] = cl_dir + os.pathsep + os.pathsep.join(path_parts)
                                                
                                                current_include = os.environ.get("INCLUDE", "")
                                                if include_dir and os.path.exists(include_dir):
                                                    if include_dir not in current_include:
                                                        os.environ["INCLUDE"] = include_dir + os.pathsep + current_include if current_include else include_dir
                                                
                                                current_lib = os.environ.get("LIB", "")
                                                if lib_dir and os.path.exists(lib_dir):
                                                    if lib_dir not in current_lib:
                                                        os.environ["LIB"] = lib_dir + os.pathsep + current_lib if current_lib else lib_dir
                                                
                                                print(f"[INFO] VS 2019 configurado manualmente: {cl_dir}")
                                                os.environ["DISTUTILS_USE_SDK"] = "1"
                                                vs_found = True
                                                break
                                    except Exception as e2:
                                        pass
                            if vs_found:
                                break
                    if vs_found:
                        break
    except Exception as e:
        print(f"[AVISO] Erro ao configurar VS 2019: {e}")
        print(f"[INFO] Tentando configuração manual do PATH")
        vs_found = False
        # Fallback: configuração manual
        for vs_base in vs_base_paths:
            if os.path.exists(vs_base):
                for vs_version in vs_preferred:
                    vs_path = os.path.join(vs_base, vs_version)
                    if os.path.exists(vs_path):
                        for edition in ["BuildTools", "Community", "Professional", "Enterprise"]:
                            vc_tools = os.path.join(vs_path, edition, "VC", "Tools", "MSVC")
                            if os.path.exists(vc_tools):
                                try:
                                    msvc_versions = sorted([d for d in os.listdir(vc_tools) if os.path.isdir(os.path.join(vc_tools, d))], reverse=True)
                                    if msvc_versions:
                                        cl_path = os.path.join(vc_tools, msvc_versions[0], "bin", "Hostx64", "x64", "cl.exe")
                                        if os.path.exists(cl_path):
                                            cl_dir = os.path.dirname(cl_path)
                                            include_dir = os.path.join(vc_tools, msvc_versions[0], "include")
                                            lib_dir = os.path.join(vc_tools, msvc_versions[0], "lib", "x64")
                                            
                                            current_path = os.environ.get("PATH", "")
                                            path_parts = [p for p in current_path.split(os.pathsep) if "Visual Studio" not in p or "2019" in p]
                                            if cl_dir not in path_parts:
                                                os.environ["PATH"] = cl_dir + os.pathsep + os.pathsep.join(path_parts)
                                            
                                            current_include = os.environ.get("INCLUDE", "")
                                            if include_dir and os.path.exists(include_dir):
                                                if include_dir not in current_include:
                                                    os.environ["INCLUDE"] = include_dir + os.pathsep + current_include if current_include else include_dir
                                            
                                            current_lib = os.environ.get("LIB", "")
                                            if lib_dir and os.path.exists(lib_dir):
                                                if lib_dir not in current_lib:
                                                    os.environ["LIB"] = lib_dir + os.pathsep + current_lib if current_lib else lib_dir
                                            
                                            print(f"[INFO] VS 2019 configurado manualmente: {cl_dir}")
                                            os.environ["DISTUTILS_USE_SDK"] = "1"
                                            vs_found = True
                                            break
                                except Exception as e2:
                                    pass
                        if vs_found:
                            break
                if vs_found:
                    break

if not vs_found:
    logger.error("[VS2019] VS 2019 não encontrado. Compilação vai falhar.")
else:
    logger.info("[VS2019] Visual Studio 2019 configurado com sucesso")

# Configurar TORCH_CUDA_ARCH_LIST com todas as arquiteturas CUDA comuns
# Isso evita o warning e garante que todas as arquiteturas sejam compiladas
# Arquiteturas CUDA: 6.0, 6.1, 6.2 (Pascal), 7.0, 7.5 (Volta/Turing), 8.0, 8.6, 8.9 (Ampere/Ada), 9.0, 9.0a (Hopper)
cuda_archs = [
    "6.0",   # Pascal (GTX 1080, Titan X)
    "6.1",   # Pascal (GTX 1080 Ti, Titan Xp)
    "6.2",   # Pascal (Tesla P100)
    "7.0",   # Volta (V100)
    "7.5",   # Turing (RTX 2080, RTX 2070, GTX 1660)
    "8.0",   # Ampere (A100, RTX 3090, RTX 3080)
    "8.6",   # Ampere (RTX 3060, RTX 3050)
    "8.9",   # Ada Lovelace (RTX 4090, RTX 4080)
    "9.0",   # Hopper (H100)
    "9.0a",  # Hopper (H100 with additional features)
]
# PyTorch espera TORCH_CUDA_ARCH_LIST como string separada por espaços
os.environ["TORCH_CUDA_ARCH_LIST"] = " ".join(cuda_archs)
logger.info(f"[CUDA] TORCH_CUDA_ARCH_LIST configurado com {len(cuda_archs)} arquiteturas CUDA")
logger.info(f"[CUDA] Arquiteturas: {os.environ['TORCH_CUDA_ARCH_LIST']}")

# Agora importar o resto
import json
import shutil
import argparse
from argparse import BooleanOptionalAction
import traceback
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger.info("[INIT] Importando PyTorch...")
import torch
logger.info(f"[INIT] PyTorch {torch.__version__} importado")
logger.info(f"[INIT] CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"[INIT] GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"[INIT] CUDA Version: {torch.version.cuda}")

from PIL import Image

# Adicionar Kiss3DGen ao path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Kiss3DGen"))
logger.info(f"[INIT] Paths configurados: {sys.path[:3]}")

logger.info("[INIT] Importando módulos Kiss3DGen...")
from scripts.kiss3d_wrapper_local import init_wrapper_from_config
from scripts.kiss3d_utils_local import TMP_DIR
logger.info("[INIT] Módulos Kiss3DGen importados com sucesso")


class PipelineCheckpoint:
    """Sistema de checkpointing para salvar resultados intermediários"""
    
    def __init__(self, output_dir: Path, job_name: str):
        self.output_dir = Path(output_dir)
        self.job_name = job_name
        self.checkpoint_dir = self.output_dir / "checkpoints" / job_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.status_file = self.checkpoint_dir / "status.json"
        self.status = self._load_status()
        
    def _load_status(self) -> Dict[str, Any]:
        """Carrega status anterior se existir"""
        if self.status_file.exists():
            try:
                return json.loads(self.status_file.read_text(encoding='utf-8'))
            except:
                pass
        return {
            "job_name": self.job_name,
            "started_at": datetime.now().isoformat(),
            "stages": {
                "initialization": {"status": "pending", "timestamp": None, "error": None},
                "caption": {"status": "pending", "timestamp": None, "error": None, "result": None},
                "multiview": {"status": "pending", "timestamp": None, "error": None, "result": None},
                "bundle_3d": {"status": "pending", "timestamp": None, "error": None, "result": None},
                "reconstruction": {"status": "pending", "timestamp": None, "error": None, "result": None},
            },
            "final_status": "in_progress"
        }
    
    def save_status(self):
        """Salva status atual"""
        self.status_file.write_text(json.dumps(self.status, indent=2, ensure_ascii=False), encoding='utf-8')
    
    def mark_stage_start(self, stage: str):
        """Marca início de uma etapa"""
        if stage in self.status["stages"]:
            self.status["stages"][stage]["status"] = "in_progress"
            self.status["stages"][stage]["timestamp"] = datetime.now().isoformat()
            self.save_status()
    
    def mark_stage_success(self, stage: str, result_path: Optional[str] = None, metadata: Optional[Dict] = None):
        """Marca sucesso de uma etapa"""
        if stage in self.status["stages"]:
            self.status["stages"][stage]["status"] = "completed"
            self.status["stages"][stage]["timestamp"] = datetime.now().isoformat()
            if result_path:
                self.status["stages"][stage]["result"] = str(result_path)
            if metadata:
                self.status["stages"][stage]["metadata"] = metadata
            self.save_status()
            logger.info(f"[CHECKPOINT] Etapa '{stage}' concluída com sucesso")
    
    def mark_stage_error(self, stage: str, error: str, traceback_str: Optional[str] = None):
        """Marca erro em uma etapa"""
        if stage in self.status["stages"]:
            self.status["stages"][stage]["status"] = "failed"
            self.status["stages"][stage]["timestamp"] = datetime.now().isoformat()
            self.status["stages"][stage]["error"] = str(error)
            if traceback_str:
                self.status["stages"][stage]["traceback"] = traceback_str
            self.save_status()
            logger.error(f"[CHECKPOINT] Etapa '{stage}' falhou: {error}")
    
    def save_file(self, stage: str, filename: str, source_path: Path) -> Path:
        """Salva arquivo de uma etapa"""
        stage_dir = self.checkpoint_dir / stage
        stage_dir.mkdir(exist_ok=True)
        dest_path = stage_dir / filename
        if source_path.exists():
            if source_path.is_file():
                shutil.copy2(source_path, dest_path)
            else:
                shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
            logger.info(f"[CHECKPOINT] Arquivo salvo: {dest_path}")
            return dest_path
        else:
            logger.warning(f"[CHECKPOINT] Arquivo não encontrado: {source_path}")
            return None
    
    def get_failed_stages(self) -> list:
        """Retorna lista de etapas que falharam"""
        return [stage for stage, info in self.status["stages"].items() if info["status"] == "failed"]
    
    def get_completed_stages(self) -> list:
        """Retorna lista de etapas completadas"""
        return [stage for stage, info in self.status["stages"].items() if info["status"] == "completed"]


def run_image_to_3d_with_checkpoints(
    k3d_wrapper,
    input_image_path: str,
    checkpoint: PipelineCheckpoint,
    enable_redux: bool = True,
    use_mv_rgb: bool = True,
    use_controlnet: bool = True,
    pipeline_mode: str = "flux",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Executa pipeline com sistema de checkpoints
    Retorna: (bundle_path, mesh_path)
    """
    from scripts.kiss3d_utils_local import preprocess_input_image
    from PIL import Image

    input_image_path = Path(input_image_path)
    if not input_image_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {input_image_path}")

    input_image = preprocess_input_image(Image.open(input_image_path))
    input_image.save(os.path.join(TMP_DIR, f"{k3d_wrapper.uuid}_input_image.png"))

    # Captions agora independem do multiview
    checkpoint.mark_stage_start("caption")
    try:
        logger.info("[ETAPA 1/3] Gerando caption diretamente da imagem...")
        caption = k3d_wrapper.get_image_caption(input_image)
        logger.info(f"[OK] Caption gerado: {caption[:160]}...")

        caption_file = checkpoint.checkpoint_dir / "caption" / "caption.txt"
        caption_file.parent.mkdir(parents=True, exist_ok=True)
        caption_file.write_text(caption, encoding="utf-8")
        checkpoint.mark_stage_success(
            "caption",
            str(caption_file),
            {"caption_preview": caption[:256], "pipeline_mode": pipeline_mode},
        )
        k3d_wrapper.release_text_models()
    except Exception as exc:
        error_msg = f"Erro ao gerar caption: {exc}"
        checkpoint.mark_stage_error("caption", error_msg, traceback.format_exc())
        raise

    recon_stage = "reconstruction"
    bundle_stage = "bundle_3d"
    bundle_save_path = None
    mesh_save_path = None

    if pipeline_mode == "multiview":
        checkpoint.mark_stage_start("multiview")
        try:
            logger.info("[ETAPA 2/3] Executando pipeline baseado em multiview (Zero123++ + LRM + ISOMER)...")
            (
                mesh_glb,
                mesh_obj,
                multiview_png,
            ) = k3d_wrapper.run_multiview_pipeline(
                input_image,
                reconstruction_stage2_steps=k3d_wrapper.get_reconstruction_stage2_steps(),
                save_intermediate_results=True,
                use_mv_rgb=use_mv_rgb,
            )

            if multiview_png and Path(multiview_png).exists():
                mv_file = checkpoint.save_file("multiview", Path(multiview_png).name, Path(multiview_png))
                checkpoint.mark_stage_success("multiview", str(mv_file))
            else:
                checkpoint.mark_stage_success("multiview", None)

            # Não há bundle intermediário nessa rota
            checkpoint.mark_stage_success(
                bundle_stage,
                None,
                {"skipped": True, "pipeline_mode": pipeline_mode},
            )

            mesh_target = mesh_glb or mesh_obj
            if mesh_target and Path(mesh_target).exists():
                mesh_file = checkpoint.save_file(recon_stage, Path(mesh_target).name, Path(mesh_target))
                checkpoint.mark_stage_success(recon_stage, str(mesh_file))
                mesh_save_path = str(mesh_file)
            else:
                checkpoint.mark_stage_success(recon_stage, None)
                mesh_save_path = mesh_target
        except Exception as exc:
            error_msg = f"Erro na reconstrução multiview: {exc}"
            checkpoint.mark_stage_error("multiview", error_msg, traceback.format_exc())
            checkpoint.mark_stage_error(recon_stage, error_msg, traceback.format_exc())
            raise
    else:
        checkpoint.mark_stage_success("multiview", None, {"skipped": True, "pipeline_mode": pipeline_mode})
        checkpoint.mark_stage_start(bundle_stage)
        try:
            logger.info("[ETAPA 2/3] Gerando bundle 2x4 condicionado por Flux/ControlNet...")
            bundle_tensor, bundle_path = k3d_wrapper.generate_flux_bundle(
                input_image=input_image,
                caption=caption,
                enable_redux=enable_redux,
                use_controlnet=use_controlnet,
            )

            if bundle_path and Path(bundle_path).exists():
                bundle_file = checkpoint.save_file(bundle_stage, Path(bundle_path).name, Path(bundle_path))
                checkpoint.mark_stage_success(bundle_stage, str(bundle_file))
                bundle_save_path = str(bundle_file)
            else:
                checkpoint.mark_stage_success(bundle_stage, None)
                bundle_save_path = bundle_path
        except Exception as exc:
            error_msg = f"Erro ao gerar bundle com Flux: {exc}"
            checkpoint.mark_stage_error(bundle_stage, error_msg, traceback.format_exc())
            raise

        checkpoint.mark_stage_start(recon_stage)
        try:
            logger.info("[ETAPA 3/3] Reconstruindo mesh a partir do bundle Flux...")
            recon_mesh_path = k3d_wrapper.reconstruct_3d_bundle_image(
                bundle_tensor,
                reconstruction_stage2_steps=k3d_wrapper.get_reconstruction_stage2_steps(),
                save_intermediate_results=True,
            )
            if recon_mesh_path and Path(recon_mesh_path).exists():
                mesh_file = checkpoint.save_file(recon_stage, Path(recon_mesh_path).name, Path(recon_mesh_path))
                checkpoint.mark_stage_success(recon_stage, str(mesh_file))
                mesh_save_path = str(mesh_file)
            else:
                checkpoint.mark_stage_success(recon_stage, None)
                mesh_save_path = recon_mesh_path
        except Exception as exc:
            error_msg = f"Erro ao reconstruir mesh do bundle Flux: {exc}"
            checkpoint.mark_stage_error(recon_stage, error_msg, traceback.format_exc())
            raise

    return bundle_save_path, mesh_save_path


def main():
    logger.info("[MAIN] Iniciando função main()")
    parser = argparse.ArgumentParser(description="Pipeline IMAGE TO 3D - Kiss3DGen com Checkpoints")
    parser.add_argument("--input", type=str, required=True, help="Caminho da imagem de entrada")
    parser.add_argument("--output", type=str, required=True, help="Diretório de saída")
    parser.add_argument("--config", type=str, default="Kiss3DGen/pipeline/pipeline_config/default.yaml")
    parser.add_argument(
        "--quality-mode",
        type=str,
        default=None,
        choices=["fast", "balanced", "high"],
        help="Perfil de qualidade (override do config).",
    )
    parser.add_argument("--fast-mode", action="store_true", help="Modo rápido")
    parser.add_argument("--disable-llm", action="store_true", help="Desabilitar LLM")
    parser.add_argument(
        "--use-controlnet",
        action=BooleanOptionalAction,
        default=True,
        help="Usar ControlNet (use --no-use-controlnet para desativar)",
    )
    parser.add_argument(
        "--enable-redux",
        action=BooleanOptionalAction,
        default=True,
        help="Habilitar Redux (use --no-enable-redux para desativar)",
    )
    parser.add_argument(
        "--use-mv-rgb",
        action=BooleanOptionalAction,
        default=True,
        help="Usar RGB multiview (use --no-use-mv-rgb para desativar)",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=["multiview", "flux"],
        default="flux",
        help="Escolhe entre reconstrução baseada em multiview (Zero123++ + LRM) ou bundle gerado por Flux",
    )
    
    args = parser.parse_args()
    logger.info(
        f"[MAIN] Argumentos: input={args.input}, output={args.output}, fast_mode={args.fast_mode}, "
        f"use_controlnet={args.use_controlnet}, enable_redux={args.enable_redux}, pipeline_mode={args.pipeline_mode}"
    )
    
    # Configurar paths
    project_root = Path(__file__).parent.parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    
    logger.info(f"[MAIN] Input path: {input_path}")
    logger.info(f"[MAIN] Output path: {output_path}")
    
    if not input_path.exists():
        logger.error(f"[MAIN] ERRO: Imagem de entrada não encontrada: {input_path}")
        sys.exit(1)
    
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"[MAIN] Diretório de saída criado/verificado: {output_path}")
    
    # Nome do job baseado na imagem
    job_name = input_path.stem
    
    # Criar checkpoint
    checkpoint = PipelineCheckpoint(output_path, job_name)
    
    logger.info("=" * 80)
    logger.info("Pipeline IMAGE TO 3D - Kiss3DGen com Sistema de Checkpoints")
    logger.info("=" * 80)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Checkpoint dir: {checkpoint.checkpoint_dir}")
    logger.info("=" * 80)
    
    # Etapa 0: Inicialização
    checkpoint.mark_stage_start("initialization")
    try:
        logger.info("[ETAPA 0/4] Inicializando pipeline...")
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root / config_path
        
        logger.info(f"[INIT] Carregando configuração de: {config_path}")
        logger.info("[INIT] Chamando init_wrapper_from_config...")
        logger.info("[INIT] Isso pode levar vários minutos (carregamento de modelos)...")
        
        # Adicionar flush para garantir que logs sejam escritos imediatamente
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        k3d_wrapper = init_wrapper_from_config(
            str(config_path),
            fast_mode=args.fast_mode,
            disable_llm=getattr(args, 'disable_llm', False),  # Padrão: False (LLM habilitado)
            load_controlnet=args.use_controlnet,
            load_redux=args.enable_redux,
            quality_mode=args.quality_mode,
        )
        logger.info("[OK] Pipeline inicializado")
        checkpoint.mark_stage_success("initialization")
    except KeyboardInterrupt:
        logger.error("[ERRO] Pipeline interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        error_msg = f"Erro ao inicializar pipeline: {e}"
        checkpoint.mark_stage_error("initialization", error_msg, traceback.format_exc())
        logger.error(error_msg)
        logger.error(f"[ERRO] Traceback completo:\n{traceback.format_exc()}")
        sys.exit(1)
    
    # Executar pipeline
    try:
        gen_save_path, recon_mesh_path = run_image_to_3d_with_checkpoints(
            k3d_wrapper,
            str(input_path),
            checkpoint,
            enable_redux=args.enable_redux,
            use_mv_rgb=args.use_mv_rgb,
            use_controlnet=args.use_controlnet,
            pipeline_mode=args.pipeline_mode,
        )
        
        # Atualizar status final
        checkpoint.status["final_status"] = "completed"
        checkpoint.status["completed_at"] = datetime.now().isoformat()
        checkpoint.save_status()
        
        logger.info("=" * 80)
        logger.info("Pipeline concluído com sucesso!")
        logger.info(f"Bundle: {gen_save_path}")
        logger.info(f"Mesh: {recon_mesh_path}")
        logger.info(f"Checkpoints salvos em: {checkpoint.checkpoint_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        checkpoint.status["final_status"] = "failed"
        checkpoint.status["failed_at"] = datetime.now().isoformat()
        checkpoint.status["final_error"] = str(e)
        checkpoint.save_status()
        
        failed_stages = checkpoint.get_failed_stages()
        completed_stages = checkpoint.get_completed_stages()
        
        logger.error("=" * 80)
        logger.error("Pipeline falhou!")
        logger.error(f"Erro: {e}")
        logger.error(f"Etapas completadas: {', '.join(completed_stages) if completed_stages else 'Nenhuma'}")
        logger.error(f"Etapas falhadas: {', '.join(failed_stages) if failed_stages else 'Nenhuma'}")
        logger.error(f"Checkpoints salvos em: {checkpoint.checkpoint_dir}")
        logger.error("=" * 80)
        
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

