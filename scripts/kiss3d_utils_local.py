import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
from PIL import Image
import rembg
import trimesh
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KISS3D_ROOT = PROJECT_ROOT / "Kiss3DGen"

if str(KISS3D_ROOT) not in sys.path:
    sys.path.insert(0, str(KISS3D_ROOT))

ORIGINAL_WORKDIR = Path.cwd()
if ORIGINAL_WORKDIR != KISS3D_ROOT:
    os.chdir(str(KISS3D_ROOT))

from models.lrm.online_render.render_single import load_mipmap, render_mesh
from models.lrm.utils.camera_util import (
    get_zero123plus_input_cameras,
    get_custom_zero123plus_input_cameras,
    get_flux_input_cameras,
)
from models.lrm.utils.render_utils import rotate_x, rotate_y
from models.lrm.utils.mesh_util import save_obj, save_obj_with_mtl
from models.lrm.utils.infer_util import remove_background, resize_foreground

from models.ISOMER.reconstruction_func import reconstruction
from models.ISOMER.projection_func import projection

# Importar função original do ISOMER para monkey-patch
from models.ISOMER.scripts.utils import save_py3dmesh_with_trimesh_fast as _original_save_py3dmesh

from utils.tool import (
    NormalTransfer,
    get_render_cameras_frames,
    get_background,
    get_render_cameras_video,
    render_frames,
    mask_fix,
)

logger = logging.getLogger("kiss3d_wrapper_local")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

ENV_FILE_PATH = PROJECT_ROOT / ".env"


def _load_env_file(env_path: Path):
    env_data = {}
    if not env_path or not env_path.exists():
        return env_data
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env_data[key.strip()] = value.strip().strip('"').strip("'")
    return env_data


def ensure_hf_token(env_path: Path | None = None) -> bool:
    """
    Garante que o token do Hugging Face esteja disponível.
    1. Carrega variáveis do arquivo .env (se existir).
    2. Usa HUGGINGFACE_TOKEN/HF_TOKEN/HUGGINGFACE_HUB_TOKEN do ambiente.
    3. Se existir token, executa login programaticamente para liberar os repositórios privados.
    """
    try:
        from huggingface_hub import login, whoami
    except ImportError:
        logger.warning("huggingface_hub não está instalado; não é possível configurar o token automaticamente.")
        return False

    # Já autenticado?
    try:
        whoami()
        return True
    except Exception:
        pass

    env_path = env_path or ENV_FILE_PATH
    env_values = _load_env_file(env_path)
    for key, value in env_values.items():
        os.environ.setdefault(key, value)

    token = (
        os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or env_values.get("HUGGINGFACE_TOKEN")
        or env_values.get("HF_TOKEN")
    )

    if not token:
        logger.warning(
            "Token do Hugging Face não informado. "
            "Crie um arquivo .env na raiz com HF_TOKEN=<seu_token> ou exporte HUGGINGFACE_TOKEN."
        )
        return False

    # Propagar token para todas as variáveis esperadas
    os.environ.setdefault("HUGGINGFACE_TOKEN", token)
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)

    try:
        login(token=token, add_to_git_credential=False)
        whoami()
        logger.info("Hugging Face autenticado via token do .env/variáveis de ambiente.")
        return True
    except Exception as exc:
        logger.error("Falha ao autenticar Hugging Face: %s", exc)
    return False

OUT_DIR_PATH = PROJECT_ROOT / "outputs"
TMP_DIR_PATH = OUT_DIR_PATH / "tmp"
OUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
TMP_DIR_PATH.mkdir(parents=True, exist_ok=True)

OUT_DIR = str(OUT_DIR_PATH)
TMP_DIR = str(TMP_DIR_PATH)

rembg_session = rembg.new_session("isnet-general-use")
normal_transfer = NormalTransfer()


def evaluate_mesh_against_gt(
    pred_mesh_path: str | Path,
    gt_mesh_path: str | Path,
    num_samples: int = 50000,
    normalize: bool = True,
    thresholds: tuple[float, ...] = (0.01, 0.02),
) -> Dict[str, Any]:
    """
    Compara a malha reconstruída com um ground-truth usando amostragem de superfície
    e distância de Chamfer.
    """

    pred_mesh = trimesh.load_mesh(pred_mesh_path, force="mesh")
    gt_mesh = trimesh.load_mesh(gt_mesh_path, force="mesh")

    def _sample_points(mesh):
        pts, _ = trimesh.sample.sample_surface(mesh, num_samples)
        return pts

    def _normalize_points(points):
        if not normalize:
            return points
        center = points.mean(axis=0, keepdims=True)
        pts = points - center
        scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        if scale > 0:
            pts = pts / scale
        return pts

    pred_pts = _normalize_points(_sample_points(pred_mesh))
    gt_pts = _normalize_points(_sample_points(gt_mesh))

    pred_tree = cKDTree(pred_pts)
    gt_tree = cKDTree(gt_pts)

    dist_pred_to_gt, _ = pred_tree.query(gt_pts, k=1)
    dist_gt_to_pred, _ = gt_tree.query(pred_pts, k=1)

    chamfer_l1 = float(dist_pred_to_gt.mean() + dist_gt_to_pred.mean())
    chamfer_l2 = float((dist_pred_to_gt ** 2).mean() + (dist_gt_to_pred ** 2).mean())

    f_scores: Dict[str, Dict[str, float]] = {}
    for thr in thresholds:
        thr_val = float(thr)
        recall = float((dist_pred_to_gt < thr_val).mean())
        precision = float((dist_gt_to_pred < thr_val).mean())
        denom = precision + recall
        f1 = float(2 * precision * recall / denom) if denom > 0 else 0.0
        f_scores[f"{thr_val:.4f}"] = {
            "precision": precision,
            "recall": recall,
            "fscore": f1,
        }

    metrics = {
        "pred_mesh": str(pred_mesh_path),
        "gt_mesh": str(gt_mesh_path),
        "num_samples": num_samples,
        "normalized": normalize,
        "chamfer_l1": chamfer_l1,
        "chamfer_l2": chamfer_l2,
        "f_scores": f_scores,
    }

    return metrics


def lrm_reconstruct(
    model,
    infer_config,
    images,
    name="",
    export_texmap=False,
    input_camera_type="zero123",
    render_3d_bundle_image=True,
    render_azimuths=[270, 0, 90, 180],
    render_elevations=[5, 5, 5, 5],
    render_radius=4.15,
):
    mesh_path_idx = os.path.join(TMP_DIR, f"{name}_recon_from_{input_camera_type}.obj")

    device = images.device
    if input_camera_type == "zero123":
        input_cameras = get_custom_zero123plus_input_cameras(batch_size=1, radius=3.5, fov=30).to(device)
    elif input_camera_type == "kiss3d":
        input_cameras = get_flux_input_cameras(batch_size=1, radius=3.5, fov=30).to(device)
    else:
        raise NotImplementedError(f"Unexpected input camera type: {input_camera_type}")

    images = v2.functional.resize(images, 512, interpolation=3, antialias=True).clamp(0, 1)

    logger.info("==> Runing LRM reconstruction ...")
    planes = model.forward_planes(images, input_cameras)
    mesh_out = model.extract_mesh(planes, use_texture_map=export_texmap, **infer_config)
    if export_texmap:
        vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
        save_obj_with_mtl(
            vertices.data.cpu().numpy(),
            uvs.data.cpu().numpy(),
            faces.data.cpu().numpy(),
            mesh_tex_idx.data.cpu().numpy(),
            tex_map.permute(1, 2, 0).data.cpu().numpy(),
            mesh_path_idx,
        )
    else:
        vertices, faces, vertex_colors = mesh_out
        save_obj(vertices, faces, vertex_colors, mesh_path_idx)
    logger.info(f"Mesh saved to {mesh_path_idx}")

    if render_3d_bundle_image:
        assert render_azimuths is not None and render_elevations is not None and render_radius is not None
        render_azimuths = torch.Tensor(render_azimuths).to(device)
        render_elevations = torch.Tensor(render_elevations).to(device)

        render_size = infer_config.render_resolution
        env = load_mipmap("models/lrm/env_mipmap/6")
        materials = (0.0, 0.9)
        all_mv, all_mvp, all_campos, identity_mv = get_render_cameras_frames(
            batch_size=1,
            radius=render_radius,
            azimuths=render_azimuths,
            elevations=render_elevations,
            fov=30,
        )
        frames, albedos, pbr_spec_lights, pbr_diffuse_lights, normals, alphas = render_frames(
            model,
            planes,
            render_cameras=all_mvp,
            camera_pos=all_campos,
            env=env,
            materials=materials,
            render_size=render_size,
            render_mv=all_mv,
            local_normal=True,
            identity_mv=identity_mv,
        )
    else:
        normals = None
        frames = None
        albedos = None

    if isinstance(vertices, torch.Tensor):
        vertices = vertices.to(device)
    else:
        vertices = torch.from_numpy(vertices).to(device)
    if isinstance(faces, torch.Tensor):
        faces = faces.to(device)
    else:
        faces = torch.from_numpy(faces).to(device)
    vertices = vertices @ rotate_x(np.pi / 2, device=device)[:3, :3]
    vertices = vertices @ rotate_y(np.pi / 2, device=device)[:3, :3]

    return vertices.cpu(), faces.cpu(), normals, frames, albedos


def local_normal_global_transform(local_normal_images, azimuths_deg, elevations_deg):
    if local_normal_images.min() >= 0:
        local_normal = local_normal_images.float() * 2 - 1
    else:
        local_normal = local_normal_images.float()
    global_normal = normal_transfer.trans_local_2_global(
        local_normal, azimuths_deg, elevations_deg, radius=4.5, for_lotus=False
    )
    global_normal[..., 0] *= -1
    global_normal = (global_normal + 1) / 2
    global_normal = global_normal.permute(0, 3, 1, 2)
    return global_normal


def isomer_reconstruct(
    rgb_multi_view,
    normal_multi_view,
    multi_view_mask,
    vertices,
    faces,
    save_paths=None,
    azimuths=[0, 90, 180, 270],
    elevations=[5, 5, 5, 5],
    geo_weights=[1.0, 0.95, 1.0, 0.95],  # Valores padrão melhorados
    color_weights=[1.0, 0.7, 1.0, 0.7],  # Valores padrão melhorados
    reconstruction_stage1_steps=15,  # Aumentado de 10 para 15
    reconstruction_stage2_steps=50,
    radius=4.5,
):
    end = time.time()
    device = rgb_multi_view.device
    
    # Detect if pytorch3d has GPU support, fallback to CPU if not
    try:
        from pytorch3d.structures import Meshes
        test_mesh = Meshes(verts=[torch.zeros(3, 3)], faces=[torch.zeros(1, 3).long()])
        if device.type == 'cuda':
            test_mesh = test_mesh.to(device)
            _ = test_mesh.faces_normals_packed()  # This will fail if no GPU support
        py3d_device = device
    except RuntimeError as e:
        if "Not compiled with GPU support" in str(e) or "GPU" in str(e):
            logger.warning("pytorch3d não tem suporte GPU, forçando CPU para operações ISOMER")
            py3d_device = torch.device('cpu')
            # Move tensors to CPU for pytorch3d operations
            vertices = vertices.cpu() if isinstance(vertices, torch.Tensor) else vertices
            faces = faces.cpu() if isinstance(faces, torch.Tensor) else faces
        else:
            raise
    
    to_tensor_ = lambda x: torch.Tensor(x).float().to(device)

    global_normal = local_normal_global_transform(
        normal_multi_view.permute(0, 2, 3, 1).cpu(),
        to_tensor_(azimuths),
        to_tensor_(elevations),
    ).to(device)
    global_normal = global_normal * multi_view_mask + (1 - multi_view_mask)

    global_normal = global_normal.permute(0, 2, 3, 1)
    multi_view_mask = multi_view_mask.squeeze(1)
    rgb_multi_view = rgb_multi_view.permute(0, 2, 3, 1)

    logger.info("==> Runing ISOMER reconstruction ...")
    # Use py3d_device for pytorch3d operations, but keep original device for other ops
    meshes = reconstruction(
        normal_pils=global_normal,
        masks=multi_view_mask,
        weights=to_tensor_(geo_weights),
        fov=30,
        radius=radius,
        camera_angles_azi=to_tensor_(azimuths),
        camera_angles_ele=to_tensor_(elevations),
        expansion_weight_stage1=0.1,
        init_type="file",
        init_verts=vertices,
        init_faces=faces,
        stage1_steps=reconstruction_stage1_steps,
        stage2_steps=reconstruction_stage2_steps,
        # Parâmetros de qualidade melhorados para mesh mais detalhada
        start_edge_len_stage1=0.08,  # Reduzido de 0.1 para mais detalhes
        end_edge_len_stage1=0.015,   # Reduzido de 0.02 para mais detalhes
        start_edge_len_stage2=0.015, # Reduzido de 0.02 para mais detalhes
        end_edge_len_stage2=0.003,   # Reduzido de 0.005 para mais detalhes
    )

    multi_view_mask_proj = mask_fix(multi_view_mask, erode_dilate=-10, blur=5)

    logger.info("==> Runing ISOMER projection ...")
    
    # Monkey-patch: substituir função de salvamento pela nossa versão melhorada
    import models.ISOMER.scripts.utils as isomer_utils
    original_save = isomer_utils.save_py3dmesh_with_trimesh_fast
    isomer_utils.save_py3dmesh_with_trimesh_fast = lambda m, p, **kwargs: save_py3dmesh_with_trimesh_fast_local(
        m, p, apply_sRGB_to_LinearRGB=True, use_uv_texture=True, texture_resolution=2048
    )
    
    try:
        save_glb_addr = projection(
            meshes,
            masks=multi_view_mask_proj.to(device),
            images=rgb_multi_view.to(device),
            azimuths=to_tensor_(azimuths),
            elevations=to_tensor_(elevations),
            weights=to_tensor_(color_weights),
            fov=30,
            radius=radius,
            save_dir=TMP_DIR,
            save_addrs=save_paths,
        )
    finally:
        # Restaurar função original
        isomer_utils.save_py3dmesh_with_trimesh_fast = original_save

    logger.info(f"==> Save mesh to {save_paths} ...")
    print(f"ISMOER time: {time.time() - end:.2f}s")
    return save_glb_addr


def to_rgb_image(maybe_rgba):
    assert isinstance(maybe_rgba, Image.Image)
    if maybe_rgba.mode == "RGB":
        return maybe_rgba, None
    elif maybe_rgba.mode == "RGBA":
        rgba = maybe_rgba
        img = np.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = Image.fromarray(img, "RGB")
        img.paste(rgba, mask=rgba.getchannel("A"))
        return img, rgba.getchannel("A")
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


def preprocess_input_image(input_image):
    image = remove_background(to_rgb_image(input_image)[0], rembg_session, bgcolor=(255, 255, 255, 255))
    image = resize_foreground(image, ratio=0.85, pad_value=255)
    return to_rgb_image(image)[0]


def render_3d_bundle_image_from_mesh(mesh_path):
    renderings = render_mesh(mesh_path, save_dir=None)

    rgbs = renderings["rgb"][..., :3].cpu().permute(0, 3, 1, 2)
    normals = renderings["normal"][..., :3].cpu()
    alphas = renderings["alpha"][..., 0].cpu()

    local_normal = local_normal_global_transform(
        normals.cpu(),
        azimuths_deg=np.array([0, 90, 180, 270]),
        elevations_deg=np.array([5, 5, 5, 5]),
    )
    local_normal = local_normal * alphas[:, None, ...] + (1 - alphas[:, None, ...])

    bundle_image = torchvision.utils.make_grid(torch.cat([rgbs, local_normal], dim=0), nrow=4, padding=0)

    return bundle_image


# ============================================================================
# FUNÇÕES COPIADAS E MELHORADAS DO KISS3DGen (NÃO MODIFICAR ORIGINAIS)
# ============================================================================

def srgb_to_linear(c_srgb):
    """Convert sRGB to linear RGB."""
    c_linear = np.where(c_srgb <= 0.04045, c_srgb / 12.92, ((c_srgb + 0.055) / 1.055) ** 2.4)
    return c_linear.clip(0, 1.)


def fix_vert_color_glb(mesh_path):
    """Fix vertex color GLB file to have proper material."""
    try:
        from pygltflib import GLTF2, Material, PbrMetallicRoughness
        obj1 = GLTF2().load(mesh_path)
        if len(obj1.meshes) > 0 and len(obj1.meshes[0].primitives) > 0:
            obj1.meshes[0].primitives[0].material = 0
            obj1.materials.append(Material(
                pbrMetallicRoughness=PbrMetallicRoughness(
                    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                ),
                emissiveFactor=[0.0, 0.0, 0.0],
                doubleSided=True,
            ))
            obj1.save(mesh_path)
    except Exception as e:
        logger.warning(f"Failed to fix GLB material: {e}")


def save_py3dmesh_with_trimesh_fast_local(
    meshes, 
    save_glb_path, 
    apply_sRGB_to_LinearRGB=True,
    use_uv_texture=True,
    texture_resolution=2048
):
    """
    Versão LOCAL melhorada de save_py3dmesh_with_trimesh_fast.
    Converte vertex colors para UV textures quando possível para melhor qualidade.
    
    Args:
        meshes: pytorch3d.structures.Meshes object
        save_glb_path: caminho de saída
        apply_sRGB_to_LinearRGB: converter sRGB para linear RGB
        use_uv_texture: se True, tenta criar textura UV (melhor qualidade)
        texture_resolution: resolução da textura se use_uv_texture=True
    """
    from pytorch3d.structures import Meshes
    
    # Converter de pytorch3d para numpy
    vertices = meshes.verts_packed().cpu().float().numpy()
    triangles = meshes.faces_packed().cpu().long().numpy()
    np_color = meshes.textures.verts_features_packed().cpu().float().numpy()
    
    if save_glb_path.endswith(".glb"):
        # Rotacionar 180 graus ao longo do eixo Y
        vertices[:, [0, 2]] = -vertices[:, [0, 2]]

    if apply_sRGB_to_LinearRGB:
        np_color = srgb_to_linear(np_color)
    
    assert vertices.shape[0] == np_color.shape[0], f"Vertices ({vertices.shape[0]}) != Colors ({np_color.shape[0]})"
    assert np_color.shape[1] == 3, f"Colors must be RGB, got shape {np_color.shape}"
    assert 0 <= np_color.min() and np_color.max() <= 1, f"Colors out of range: min={np_color.min()}, max={np_color.max()}"
    
    # Clamp colors
    np_color = np.clip(np_color, 0, 1)
    
    # Tentar criar textura UV se solicitado e for GLB
    if use_uv_texture and save_glb_path.endswith(".glb"):
        try:
            # Criar mesh base
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, vertex_colors=np_color)
            mesh.remove_unreferenced_vertices()
            
            # Tentar criar textura UV usando unwrapping do trimesh
            # Isso cria uma textura real ao invés de apenas vertex colors
            try:
                # Gerar UV coordinates usando unwrapping
                # O trimesh tem funcionalidade para isso, mas pode falhar em meshes complexas
                # Por enquanto, vamos usar uma abordagem mais simples mas funcional
                
                # Criar textura a partir das cores dos vértices
                # Usar projeção simples para criar UV mapping
                texture_img, uv_coords = _create_texture_from_vertex_colors(
                    vertices, triangles, np_color, texture_resolution
                )
                
                # Criar material com textura
                from trimesh.visual import TextureVisuals
                from trimesh.visual.material import SimpleMaterial
                
                material = SimpleMaterial(image=texture_img)
                visual = TextureVisuals(uv=uv_coords, material=material)
                mesh.visual = visual
                
                logger.info(f"Exportando mesh com textura UV ({texture_resolution}x{texture_resolution})")
                mesh.export(save_glb_path)
                
            except Exception as e:
                logger.warning(f"Falha ao criar textura UV, usando vertex colors: {e}")
                # Fallback para vertex colors
                mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, vertex_colors=np_color)
                mesh.remove_unreferenced_vertices()
                mesh.export(save_glb_path)
                
        except Exception as e:
            logger.warning(f"Erro ao processar mesh com textura, usando método básico: {e}")
            # Fallback completo
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, vertex_colors=np_color)
            mesh.remove_unreferenced_vertices()
            mesh.export(save_glb_path)
    else:
        # Usar vertex colors (comportamento original)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, vertex_colors=np_color)
        mesh.remove_unreferenced_vertices()
        mesh.export(save_glb_path)
    
    # Fix material para GLB
    if save_glb_path.endswith(".glb"):
        fix_vert_color_glb(save_glb_path)


def _create_texture_from_vertex_colors(vertices, faces, vertex_colors, resolution=2048):
    """
    Cria textura e UV coordinates a partir de vertex colors.
    Usa projeção melhorada e interpolação para qualidade superior.
    """
    # Gerar UV coordinates usando projeção esférica melhorada
    # Usa coordenadas esféricas para melhor distribuição
    uv_coords = np.zeros((len(vertices), 2))
    
    # Calcular coordenadas esféricas
    r = np.linalg.norm(vertices, axis=1)
    theta = np.arctan2(vertices[:, 1], vertices[:, 0])  # Azimuth
    phi = np.arccos(np.clip(vertices[:, 2] / (r + 1e-8), -1, 1))  # Elevation
    
    # Normalizar para [0, 1]
    uv_coords[:, 0] = (theta + np.pi) / (2 * np.pi)  # Azimuth -> U
    uv_coords[:, 1] = phi / np.pi  # Elevation -> V
    
    # Criar grid de UV
    u_grid = np.linspace(0, 1, resolution)
    v_grid = np.linspace(0, 1, resolution)
    U, V = np.meshgrid(u_grid, v_grid)
    
    # Interpolar cores usando interpolação bilinear
    from scipy.interpolate import griddata
    
    # Criar pontos de amostragem
    uv_points = np.column_stack([U.ravel(), V.ravel()])
    
    # Interpolar cores nos pontos do grid
    colors_interp = griddata(
        uv_coords,
        vertex_colors,
        uv_points,
        method='linear',
        fill_value=np.mean(vertex_colors, axis=0)  # Preencher com média se fora do range
    )
    
    # Reshape para imagem
    texture = (np.clip(colors_interp, 0, 1) * 255).astype(np.uint8).reshape(resolution, resolution, 3)
    
    # Aplicar suavização para melhor qualidade
    try:
        from scipy.ndimage import gaussian_filter
        texture = gaussian_filter(texture.astype(float), sigma=0.5).astype(np.uint8)
    except:
        pass
    
    texture_img = Image.fromarray(texture)
    
    return texture_img, uv_coords
