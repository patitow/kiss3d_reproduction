import sys
from pathlib import Path

import numpy as np
import trimesh

if len(sys.argv) < 2:
    sys.exit("Usage: python tmp_color_stats.py <mesh_path>")

mesh_path = Path(sys.argv[1])
mesh = trimesh.load(mesh_path, process=False)
if isinstance(mesh, trimesh.Scene):
    if not mesh.geometry:
        raise RuntimeError(f"Scene in {mesh_path} has no geometries")
    mesh = trimesh.util.concatenate(
        [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
    )

visual = mesh.visual
if hasattr(visual, "vertex_colors") and visual.vertex_colors is not None:
    colors = visual.vertex_colors
else:
    colors = visual.to_color().vertex_colors
print(f"path: {mesh_path}")
print("shape:", colors.shape, "dtype:", colors.dtype)
print("min:", colors.min(axis=0))
print("max:", colors.max(axis=0))
print("mean:", colors.mean(axis=0))
print("std:", colors.std(axis=0))

