import ninja
import os
from pathlib import Path

ninja_path = Path(ninja.__file__).parent
print(f"Ninja package path: {ninja_path}")

# Tentar diferentes caminhos
possible_paths = [
    ninja_path / "data" / "bin",
    ninja_path.parent / "ninja" / "data" / "bin",
    Path("mesh3d-generator-py3.11") / "ninja" / "data" / "bin",
]

for path in possible_paths:
    ninja_exe = path / "ninja.exe"
    print(f"Checking: {ninja_exe}")
    if ninja_exe.exists():
        print(f"FOUND: {ninja_exe}")
        print(f"Absolute: {ninja_exe.resolve()}")

