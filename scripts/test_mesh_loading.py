#!/usr/bin/env python3
"""Teste de carregamento de mesh original"""

import sys
from pathlib import Path
import trimesh
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

model_path = Path("data/raw/gazebo_dataset/models/Gigabyte_GA78LMTUSB3_50_Motherboard_Micro_ATX_Socket_AM3/meshes/model.obj")

print(f"Carregando: {model_path}")
print(f"Existe: {model_path.exists()}")

if model_path.exists():
    mesh = trimesh.load(str(model_path))
    
    print(f"\nTipo: {type(mesh)}")
    print(f"É Scene: {isinstance(mesh, trimesh.Scene)}")
    print(f"É Mesh: {isinstance(mesh, trimesh.Trimesh)}")
    
    if isinstance(mesh, trimesh.Scene):
        print(f"\nGeometry keys: {list(mesh.geometry.keys())}")
        for key, geom in mesh.geometry.items():
            print(f"  {key}: {type(geom)}")
            if isinstance(geom, trimesh.Trimesh):
                print(f"    Vertices: {len(geom.vertices)}")
                print(f"    Faces: {len(geom.faces)}")
                print(f"    Has vertex colors: {hasattr(geom.visual, 'vertex_colors')}")
                print(f"    Has material: {hasattr(geom.visual, 'material')}")
    elif isinstance(mesh, trimesh.Trimesh):
        print(f"\nVertices: {len(mesh.vertices)}")
        print(f"Faces: {len(mesh.faces)}")
        print(f"Has vertex colors: {hasattr(mesh.visual, 'vertex_colors')}")
        print(f"Has material: {hasattr(mesh.visual, 'material')}")
        print(f"Bounds: {mesh.bounds}")
        print(f"Volume: {mesh.volume}")
        
        # Verificar se tem MTL
        mtl_path = model_path.parent / "model.mtl"
        if mtl_path.exists():
            print(f"\nMTL encontrado: {mtl_path}")
            with open(mtl_path, 'r') as f:
                mtl_content = f.read()
                print(f"MTL size: {len(mtl_content)} bytes")
                print(f"MTL lines: {len(mtl_content.splitlines())}")


