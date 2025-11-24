#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para configurar autenticação do HuggingFace
Necessário para baixar modelos restritos como FLUX.1-dev
"""

import os
import sys
from huggingface_hub import login, whoami

def main():
    print("="*60)
    print("CONFIGURAÇÃO DE AUTENTICAÇÃO HUGGINGFACE")
    print("="*60)
    
    print("\n[INFO] Modelos restritos (gated) requerem autenticação:")
    print("   - black-forest-labs/FLUX.1-dev")
    print("   - black-forest-labs/FLUX.1-Redux-dev")
    
    print("\n[INFO] Para baixar esses modelos, você precisa:")
    print("   1. Criar conta no HuggingFace: https://huggingface.co/join")
    print("   2. Aceitar os termos de uso do modelo:")
    print("      https://huggingface.co/black-forest-labs/FLUX.1-dev")
    print("   3. Gerar um token de acesso:")
    print("      https://huggingface.co/settings/tokens")
    print("   4. Autenticar usando este script ou:")
    print("      huggingface-cli login")
    
    # Verificar se já está autenticado
    try:
        user = whoami()
        print(f"\n✅ Já autenticado como: {user.get('name', 'Unknown')}")
        print(f"   Email: {user.get('email', 'Unknown')}")
        
        response = input("\nDeseja fazer logout e autenticar novamente? (s/N): ")
        if response.lower() != 's':
            print("✅ Mantendo autenticação atual")
            return True
    except Exception:
        print("\n⚠️  Não autenticado")
    
    # Solicitar token
    print("\n[INFO] Cole seu token do HuggingFace abaixo")
    print("   (ou pressione Enter para pular)")
    token = input("Token: ").strip()
    
    if not token:
        print("⚠️  Token não fornecido. Pulando autenticação.")
        print("\n💡 Para autenticar depois, execute:")
        print("   huggingface-cli login")
        print("   ou")
        print("   python scripts/setup_huggingface_auth.py")
        return False
    
    try:
        login(token=token)
        user = whoami()
        print(f"\n✅ Autenticado com sucesso!")
        print(f"   Usuário: {user.get('name', 'Unknown')}")
        return True
    except Exception as e:
        print(f"\n❌ Erro na autenticação: {e}")
        print("\n💡 Verifique se o token está correto")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

