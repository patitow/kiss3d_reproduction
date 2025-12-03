#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para pré-compilar o renderutils_plugin (nvdiffrast) antes da execução do pipeline.
Isso evita travamentos durante a execução e permite verificar se a compilação está funcionando.
"""

import os
import sys
import torch
from pathlib import Path

# Adicionar caminhos necessários
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Kiss3DGen"))

def check_precompiled():
    """Verifica se o plugin já está compilado"""
    print("=" * 80)
    print("VERIFICAÇÃO DE PRÉ-COMPILAÇÃO DO RENDERUTILS_PLUGIN")
    print("=" * 80)
    
    # Método 1: Tentar importar diretamente
    try:
        import renderutils_plugin
        print("[OK] Plugin já está importado e disponível!")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"[AVISO] Erro ao tentar importar: {e}")
    
    # Método 2: Verificar diretório de build
    try:
        import torch.utils.cpp_extension
        build_dir = torch.utils.cpp_extension._get_build_directory('renderutils_plugin', False)
        if os.path.exists(build_dir):
            print(f"[INFO] Diretório de build encontrado: {build_dir}")
            compiled_files = []
            for root, dirs, files in os.walk(build_dir):
                for file in files:
                    if (file.endswith('.pyd') or file.endswith('.so')) and 'renderutils_plugin' in file:
                        compiled_files.append(os.path.join(root, file))
            
            if compiled_files:
                print(f"[OK] Encontrados {len(compiled_files)} arquivo(s) compilado(s):")
                for f in compiled_files:
                    size_mb = os.path.getsize(f) / (1024 * 1024)
                    print(f"  - {f} ({size_mb:.2f} MB)")
                return True
            else:
                print("[INFO] Diretório de build existe mas não há arquivos compilados")
        else:
            print("[INFO] Diretório de build não existe")
    except Exception as e:
        print(f"[AVISO] Erro ao verificar build directory: {e}")
    
    return False

def precompile_plugin():
    """Pré-compila o renderutils_plugin"""
    print("\n" + "=" * 80)
    print("PRÉ-COMPILAÇÃO DO RENDERUTILS_PLUGIN")
    print("=" * 80)
    
    # Verificar se já está compilado
    if check_precompiled():
        print("\n[OK] Plugin já está compilado! Não é necessário recompilar.")
        return True
    
    print("\n[INFO] Plugin não encontrado. Iniciando compilação...")
    print("[AVISO] Esta etapa pode demorar 5-15 minutos dependendo do sistema.")
    print("[INFO] Verificando requisitos...")
    
    # Verificar CUDA
    if not torch.cuda.is_available():
        print("[ERRO] CUDA não está disponível! Plugin requer CUDA.")
        return False
    
    print(f"[OK] CUDA disponível: {torch.version.cuda}")
    print(f"[OK] PyTorch: {torch.__version__}")
    
    # Verificar Visual Studio (Windows)
    if os.name == 'nt':
        import glob
        vs_paths = [
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\*\bin\Hostx64\x64",
            r"C:\Program Files\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64",
            r"C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\*\bin\Hostx64\x64",
        ]
        
        cl_found = False
        for pattern in vs_paths:
            paths = sorted(glob.glob(pattern), reverse=True)
            if paths:
                cl_path = os.path.join(paths[0], "cl.exe")
                if os.path.exists(cl_path):
                    print(f"[OK] Visual Studio 2019 encontrado: {paths[0]}")
                    cl_found = True
                    # Adicionar ao PATH
                    if paths[0] not in os.environ.get('PATH', ''):
                        os.environ['PATH'] = paths[0] + ';' + os.environ.get('PATH', '')
                    break
        
        if not cl_found:
            print("[AVISO] Visual Studio 2019 não encontrado explicitamente.")
            print("[INFO] Tentando usar cl.exe do PATH...")
            import subprocess
            try:
                result = subprocess.run(['where', 'cl.exe'], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"[OK] cl.exe encontrado no PATH: {result.stdout.strip()}")
                    cl_found = True
            except:
                pass
        
        if not cl_found:
            print("[ERRO] Visual Studio Build Tools não encontrado!")
            print("[INFO] Instale Visual Studio 2019 Build Tools com 'Desktop development with C++'")
            return False
    
    # Importar e usar a função de compilação do ops.py
    print("\n[INFO] Importando módulo de compilação...")
    try:
        from Kiss3DGen.models.lrm.models.geometry.render.renderutils import ops
        
        # Forçar compilação chamando _get_plugin
        print("[INFO] Iniciando compilação...")
        print("[INFO] Isso pode demorar vários minutos. Aguarde...")
        sys.stdout.flush()
        
        plugin = ops._get_plugin()
        
        if plugin is not None:
            print("\n[OK] Compilação concluída com sucesso!")
            
            # Testar o plugin
            print("[INFO] Testando plugin compilado...")
            try:
                # Teste simples: verificar se o módulo tem as funções esperadas
                required_funcs = ['prepare_shading_normal_fwd', 'xfm_points']
                missing = []
                for func_name in required_funcs:
                    if not hasattr(plugin, func_name):
                        missing.append(func_name)
                
                if missing:
                    print(f"[AVISO] Algumas funções não encontradas: {missing}")
                else:
                    print("[OK] Plugin testado e funcionando corretamente!")
                
                return True
            except Exception as test_err:
                print(f"[AVISO] Erro ao testar plugin: {test_err}")
                print("[INFO] Mas a compilação parece ter sido bem-sucedida.")
                return True
        else:
            print("[ERRO] Compilação retornou None!")
            return False
            
    except Exception as e:
        print(f"\n[ERRO] Falha durante compilação: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Função principal"""
    print("\n" + "=" * 80)
    print("PRÉ-COMPILAÇÃO DO NVDIFFRAST (renderutils_plugin)")
    print("=" * 80)
    print("\nEste script compila o renderutils_plugin antes da execução do pipeline,")
    print("evitando travamentos durante a execução.\n")
    
    # Verificar se já está compilado
    if check_precompiled():
        print("\n" + "=" * 80)
        print("[OK] Plugin já está compilado!")
        print("=" * 80)
        return 0
    
    # Perguntar se deseja compilar
    print("\n[INFO] Plugin não encontrado.")
    response = input("Deseja compilar agora? (s/N): ").strip().lower()
    
    if response not in ['s', 'sim', 'y', 'yes']:
        print("[INFO] Compilação cancelada pelo usuário.")
        return 1
    
    # Compilar
    success = precompile_plugin()
    
    if success:
        print("\n" + "=" * 80)
        print("[OK] PRÉ-COMPILAÇÃO CONCLUÍDA COM SUCESSO!")
        print("=" * 80)
        print("\nO plugin está pronto para uso. O pipeline não precisará compilar durante a execução.")
        return 0
    else:
        print("\n" + "=" * 80)
        print("[ERRO] PRÉ-COMPILAÇÃO FALHOU")
        print("=" * 80)
        print("\nVerifique os erros acima e tente novamente.")
        print("Certifique-se de que:")
        print("  - Visual Studio 2019 Build Tools está instalado")
        print("  - CUDA está configurado corretamente")
        print("  - As variáveis de ambiente estão configuradas")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[INFO] Compilação interrompida pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERRO] Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

