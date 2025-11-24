#!/usr/bin/env python3
from huggingface_hub import whoami

try:
    user = whoami()
    print(f"✅ Autenticado como: {user.get('name', 'Unknown')}")
    print(f"   Email: {user.get('email', 'Unknown')}")
except Exception:
    print("❌ Não autenticado no HuggingFace")
    print("\n💡 Para autenticar:")
    print("   1. Acesse: https://huggingface.co/settings/tokens")
    print("   2. Crie um token (read)")
    print("   3. Execute: huggingface-cli login")
    print("   4. Ou: python scripts/setup_huggingface_auth.py")

