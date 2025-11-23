#!/usr/bin/env python3
"""Debug script para verificar mesh original"""

import sys
from pathlib import Path
import trimesh
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

mesh_path = Path("data/raw/gazebo_dataset/models/Gigabyte_GA78LMTUSB3_50_Motherboard_Micro_ATX_Socket_AM3/meshes/model.obj")

print(f"Carregando: {mesh_path}")
mesh = trimesh.load(str(mesh_path))

print(f"\n=== Informações da Mesh ===")
print(f"Tipo: {type(mesh)}")
print(f"É Scene: {isinstance(mesh, trimesh.Scene)}")
print(f"É Mesh: {isinstance(mesh, trimesh.Trimesh)}")

if isinstance(mesh, trimesh.Trimesh):
    print(f"\nVertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"É watertight: {mesh.is_watertight}")
    print(f"Volume: {mesh.volume}")
    print(f"Área de superfície: {mesh.area}")
    print(f"Bounds: {mesh.bounds}")
    print(f"Centroid: {mesh.centroid}")
    
    # Verificar componentes conectados
    print(f"\n=== Componentes Conectados ===")
    components = mesh.split(only_watertight=False)
    print(f"Número de componentes: {len(components)}")
    for i, comp in enumerate(components):
        print(f"  Componente {i}: {len(comp.vertices)} vertices, {len(comp.faces)} faces")
    
    # Verificar se há faces degeneradas
    print(f"\n=== Qualidade da Mesh ===")
    print(f"Faces degeneradas: {mesh.faces.shape[0] - len(mesh.nondegenerate_faces())}")
    print(f"Vértices não referenciados: {len(mesh.vertices) - len(mesh.referenced_vertices)}")
    
    # Verificar orientação das faces
    print(f"\n=== Orientação ===")
    print(f"Faces com orientação consistente: {mesh.is_winding_consistent}")
    
    # Verificar se há buracos
    print(f"\n=== Topologia ===")
    edges = mesh.edges
    print(f"Edges: {len(edges)}")
    
    # Verificar se mesh tem visual/material
    print(f"\n=== Visual ===")
    print(f"Tem visual: {mesh.visual is not None}")
    if mesh.visual is not None:
        print(f"Tem vertex colors: {hasattr(mesh.visual, 'vertex_colors')}")
        print(f"Tem material: {hasattr(mesh.visual, 'material')}")
        if hasattr(mesh.visual, 'material'):
            print(f"Material: {mesh.visual.material}")

