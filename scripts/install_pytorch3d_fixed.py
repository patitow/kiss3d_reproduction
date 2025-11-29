#!/usr/bin/env python3
"""
Script para instalar PyTorch3D com CUDA de forma robusta
Tenta diferentes métodos até conseguir
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Executa um comando e retorna o resultado"""
    print(f"\n[EXEC] {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Comando falhou com código {result.returncode}")
    return result

def setup_vs2019_env():
    """Configura o ambiente do VS 2019"""
    vs2019_paths = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
    ]
    
    for vcvars in vs2019_paths:
        if os.path.exists(vcvars):
            print(f"[OK] VS 2019 encontrado: {vcvars}")
            # O vcvars64.bat precisa ser chamado em um shell cmd
            # Vamos retornar o caminho para ser usado no batch
            return vcvars
    
    print("[ERRO] VS 2019 não encontrado!")
    return None

def check_cuda():
    """Verifica se CUDA está disponível"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] CUDA disponível: {torch.version.cuda}")
            return True
        else:
            print("[ERRO] CUDA não disponível no PyTorch")
            return False
    except ImportError:
        print("[ERRO] PyTorch não encontrado")
        return False

def install_pytorch3d_method1():
    """Método 1: Instalar via pip do repositório git"""
    print("\n" + "="*60)
    print("MÉTODO 1: Instalação via pip do git")
    print("="*60)
    
    try:
        # Desinstalar versões anteriores
        run_command(f"{sys.executable} -m pip uninstall pytorch3d -y", check=False)
        
        # Instalar do git
        cmd = f'{sys.executable} -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-cache-dir'
        run_command(cmd)
        
        # Verificar instalação
        import pytorch3d
        from pytorch3d import _C
        print(f"[OK] PyTorch3D instalado: {pytorch3d.__version__}")
        print("[OK] Módulo _C importado com sucesso!")
        return True
    except Exception as e:
        print(f"[ERRO] Método 1 falhou: {e}")
        return False

def install_pytorch3d_method2():
    """Método 2: Compilar do código-fonte local"""
    print("\n" + "="*60)
    print("MÉTODO 2: Compilação do código-fonte local")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    temp_pytorch3d = project_root / "temp_pytorch3d"
    
    # Clonar se não existir
    if not temp_pytorch3d.exists():
        print("[INFO] Clonando repositório PyTorch3D...")
        run_command(
            "git clone https://github.com/facebookresearch/pytorch3d.git temp_pytorch3d",
            cwd=project_root
        )
    
    # Fazer checkout da branch stable
    print("[INFO] Fazendo checkout da branch stable...")
    run_command("git checkout stable", cwd=temp_pytorch3d)
    
    # Instalar dependências
    print("[INFO] Instalando dependências...")
    run_command(f"{sys.executable} -m pip install iopath fvcore", check=False)
    
    # Configurar variáveis de ambiente
    os.environ["FORCE_CUDA"] = "1"
    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MAX_JOBS"] = "1"
    # Tentar desabilitar Ninja se houver problemas
    os.environ["PYTORCH3D_NO_NINJA"] = "0"  # Tentar com Ninja primeiro
    
    # Configurar CUDA
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
    ]
    
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            os.environ["CUDA_HOME"] = cuda_path
            os.environ["CUDA_PATH"] = cuda_path
            print(f"[OK] CUDA configurado: {cuda_path}")
            break
    
    # Configurar CUB se existir
    cub_path = r"C:\cub"
    if os.path.exists(cub_path):
        os.environ["CUB_HOME"] = cub_path
        print(f"[OK] CUB configurado: {cub_path}")
    
    # Compilar e instalar
    print("[INFO] Compilando PyTorch3D (isso pode demorar 20-40 minutos)...")
    try:
        run_command(
            f"{sys.executable} setup.py install",
            cwd=temp_pytorch3d
        )
        
        # Verificar instalação
        import pytorch3d
        from pytorch3d import _C
        print(f"[OK] PyTorch3D instalado: {pytorch3d.__version__}")
        print("[OK] Módulo _C importado com sucesso!")
        return True
    except Exception as e:
        print(f"[ERRO] Método 2 falhou: {e}")
        return False

def install_pytorch3d_method3():
    """Método 3: Usar wheel pré-compilado se disponível"""
    print("\n" + "="*60)
    print("MÉTODO 3: Tentando wheel pré-compilado")
    print("="*60)
    
    try:
        import torch
        pytorch_version = torch.__version__.split('+')[0]
        cuda_version = torch.version.cuda.replace('.', '') if torch.version.cuda else 'cpu'
        
        print(f"[INFO] Procurando wheel para PyTorch {pytorch_version}, CUDA {cuda_version}")
        
        # Tentar instalar diretamente
        run_command(f"{sys.executable} -m pip install pytorch3d --no-cache-dir", check=False)
        
        # Verificar instalação
        import pytorch3d
        from pytorch3d import _C
        print(f"[OK] PyTorch3D instalado: {pytorch3d.__version__}")
        print("[OK] Módulo _C importado com sucesso!")
        return True
    except Exception as e:
        print(f"[ERRO] Método 3 falhou: {e}")
        return False

def install_pytorch3d_method4():
    """Método 4: Compilar sem Ninja (mais lento mas mais compatível)"""
    print("\n" + "="*60)
    print("MÉTODO 4: Compilação sem Ninja")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    temp_pytorch3d = project_root / "temp_pytorch3d"
    
    # Garantir que o código-fonte existe
    if not temp_pytorch3d.exists():
        print("[INFO] Clonando repositório PyTorch3D...")
        run_command(
            "git clone https://github.com/facebookresearch/pytorch3d.git temp_pytorch3d",
            cwd=project_root
        )
    
    run_command("git checkout stable", cwd=temp_pytorch3d)
    
    # Instalar dependências
    run_command(f"{sys.executable} -m pip install iopath fvcore", check=False)
    
    # Configurar variáveis de ambiente (sem Ninja)
    os.environ["FORCE_CUDA"] = "1"
    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MAX_JOBS"] = "1"
    os.environ["PYTORCH3D_NO_NINJA"] = "1"  # Desabilitar Ninja
    
    # Configurar CUDA
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
    ]
    
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            os.environ["CUDA_HOME"] = cuda_path
            os.environ["CUDA_PATH"] = cuda_path
            print(f"[OK] CUDA configurado: {cuda_path}")
            break
    
    # Configurar CUB
    cub_path = r"C:\cub"
    if os.path.exists(cub_path):
        os.environ["CUB_HOME"] = cub_path
        print(f"[OK] CUB configurado: {cub_path}")
    
    print("[INFO] Compilando PyTorch3D sem Ninja (pode demorar mais)...")
    try:
        run_command(
            f"{sys.executable} setup.py install",
            cwd=temp_pytorch3d
        )
        
        # Verificar instalação
        import pytorch3d
        from pytorch3d import _C
        print(f"[OK] PyTorch3D instalado: {pytorch3d.__version__}")
        print("[OK] Módulo _C importado com sucesso!")
        return True
    except Exception as e:
        print(f"[ERRO] Método 4 falhou: {e}")
        return False

def main():
    print("="*60)
    print("INSTALAÇÃO PyTorch3D - MÉTODO ROBUSTO")
    print("="*60)
    
    # Verificar pré-requisitos
    if not check_cuda():
        print("[ERRO] CUDA não disponível. Instale PyTorch com suporte CUDA primeiro.")
        return 1
    
    vs2019 = setup_vs2019_env()
    if not vs2019:
        print("[AVISO] VS 2019 não encontrado. A compilação pode falhar.")
        print("[INFO] Continuando mesmo assim...")
    
    # Tentar métodos em ordem
    methods = [
        ("Wheel pré-compilado", install_pytorch3d_method3),
        ("Git + pip", install_pytorch3d_method1),
        ("Compilação local (com Ninja)", install_pytorch3d_method2),
        ("Compilação local (sem Ninja)", install_pytorch3d_method4),
    ]
    
    for method_name, method_func in methods:
        print(f"\n{'='*60}")
        print(f"Tentando: {method_name}")
        print(f"{'='*60}")
        try:
            if method_func():
                print(f"\n[SUCESSO] PyTorch3D instalado usando: {method_name}")
                return 0
        except Exception as e:
            print(f"[ERRO] {method_name} falhou: {e}")
            continue
    
    print("\n[ERRO] Todos os métodos falharam!")
    print("[INFO] Verifique os erros acima para mais detalhes.")
    return 1

if __name__ == "__main__":
    sys.exit(main())

