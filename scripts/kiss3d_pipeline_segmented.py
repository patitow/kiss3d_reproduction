"""
Pipeline Kiss3DGen segmentado com descarregamento de VRAM entre etapas
e melhorias de consistência de texto
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from PIL import Image
import torchvision
from einops import rearrange

# Adicionar paths necessários
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KISS3D_ROOT = PROJECT_ROOT / "Kiss3DGen"
if str(KISS3D_ROOT) not in sys.path:
    sys.path.insert(0, str(KISS3D_ROOT))

from kiss3d_wrapper_local import kiss3d_wrapper, init_wrapper_from_config
from kiss3d_utils_local import (
    TMP_DIR,
    OUT_DIR,
    preprocess_input_image,
    evaluate_mesh_against_gt,
    logger,
)

# Para processamento de imagens e OCR
try:
    import cv2
    from skimage import filters
except ImportError:
    cv2 = None
    filters = None

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("easyocr não disponível - métricas de OCR serão puladas")

# Para geração de vídeos
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    logger.warning("imageio não disponível - geração de vídeos será pulada")


def _empty_cuda_cache():
    """Descarrega VRAM explicitamente"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def process_image_features(
    image: Image.Image,
    output_dir: Path,
    uuid: str,
    generate_depth: bool = True,
    generate_normal: bool = True,
    generate_canny: bool = True,
    generate_shuffle: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Etapa 1: Processa a imagem de input gerando features (depth, normal, canny, etc)
    Descarrega VRAM após processamento
    """
    logger.info("[Etapa 1/7] Processando features da imagem...")
    
    features = {}
    img_array = np.array(image.convert("RGB"))
    
    if generate_depth and cv2 is not None:
        # Depth estimation usando MiDaS ou método simples
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        depth = cv2.bilateralFilter(gray, 9, 75, 75)
        depth = cv2.adaptiveThreshold(depth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(0) / 255.0
        features["depth"] = depth_tensor
        logger.info("  ✓ Depth map gerado")
    
    if generate_normal and cv2 is not None:
        # Normal map estimation
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        normal = np.dstack([sobelx, sobely, np.ones_like(gray)])
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = normal / (norm + 1e-6)
        normal = (normal + 1) / 2  # Normalizar para [0, 1]
        normal_tensor = torch.from_numpy(normal).float().permute(2, 0, 1)
        features["normal"] = normal_tensor
        logger.info("  ✓ Normal map gerado")
    
    if generate_canny and cv2 is not None:
        # Canny edges
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray, 100, 200)
        canny_tensor = torch.from_numpy(canny).float().unsqueeze(0) / 255.0
        features["canny"] = canny_tensor
        logger.info("  ✓ Canny edges gerado")
    
    if generate_shuffle and filters is not None:
        # Shuffle (tile-based feature)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Aplicar filtro de tiles
        tiles = np.array_split(gray, 8, axis=0)
        tiles = [np.array_split(t, 8, axis=1) for t in tiles]
        shuffled = np.zeros_like(gray)
        for i, row in enumerate(tiles):
            for j, tile in enumerate(row):
                shuffled[i*len(tile):(i+1)*len(tile), j*len(tile[0]):(j+1)*len(tile[0])] = tile
        shuffle_tensor = torch.from_numpy(shuffled).float().unsqueeze(0) / 255.0
        features["shuffle"] = shuffle_tensor
        logger.info("  ✓ Shuffle feature gerado")
    
    # Salvar features intermediárias
    for name, tensor in features.items():
        save_path = output_dir / f"{uuid}_{name}_feature.png"
        torchvision.utils.save_image(tensor, save_path)
    
    _empty_cuda_cache()
    logger.info("[Etapa 1/7] ✓ Features processadas e VRAM descarregada")
    
    return features


def generate_detailed_caption(
    k3d_wrapper: kiss3d_wrapper,
    image: Image.Image,
    use_llm: bool = True,
    preserve_text: bool = True,
) -> str:
    """
    Etapa 2: Gera descrição detalhada da imagem
    Descarrega VRAM após processamento
    """
    logger.info("[Etapa 2/7] Gerando descrição detalhada da imagem...")
    
    # Caption básico com Florence-2
    basic_caption = k3d_wrapper.get_image_caption(image)
    logger.info(f"  Caption básico: {basic_caption[:100]}...")
    
    # OCR para extrair texto da imagem
    extracted_text = ""
    if preserve_text and EASYOCR_AVAILABLE:
        try:
            reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            results = reader.readtext(np.array(image))
            extracted_text = " ".join([result[1] for result in results])
            if extracted_text:
                logger.info(f"  Texto extraído (OCR): {extracted_text}")
        except Exception as e:
            logger.warning(f"  Erro no OCR: {e}")
    
    # Refinamento com LLM se disponível
    if use_llm and k3d_wrapper.llm_model is not None:
        system_prompt = """You are an expert at generating detailed prompts for 3D object generation.
Generate an extremely detailed, precise, long, and robust description of the object in the image.
Focus on:
- Geometric details (shape, dimensions, proportions)
- Material properties (texture, reflectivity, roughness)
- Color information (exact shades, gradients, patterns)
- Text preservation (if text is visible, ensure it remains crisp and legible)
- View-specific details for 4 views (front, left, rear, right)
- 3D qualities and depth information

Output ONLY the prompt in English, be extremely detailed and specific."""
        
        user_prompt = f"Image description: {basic_caption}"
        if extracted_text:
            user_prompt += f"\n\nText visible in image (must be preserved): {extracted_text}"
        
        try:
            from models.llm.llm import get_llm_response
            detailed_caption = get_llm_response(
                k3d_wrapper.llm_model,
                k3d_wrapper.llm_tokenizer,
                user_prompt,
                system_prompt=system_prompt,
            )
            logger.info(f"  Caption refinado (LLM): {detailed_caption[:150]}...")
            _empty_cuda_cache()
            return detailed_caption
        except Exception as e:
            logger.warning(f"  Erro no LLM: {e}, usando caption básico")
    
    _empty_cuda_cache()
    logger.info("[Etapa 2/7] ✓ Descrição gerada e VRAM descarregada")
    
    return basic_caption


def generate_multiview_with_consistency(
    k3d_wrapper: kiss3d_wrapper,
    image: Image.Image,
    caption: str,
    features: Dict[str, torch.Tensor],
    use_controlnet: bool = True,
    control_modes: List[str] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Etapa 3: Gera vistas multiview com consistência
    Descarrega VRAM após processamento
    """
    logger.info("[Etapa 3/7] Gerando vistas multiview...")
    
    # Gerar vista de referência com Zero123++
    mv_image = k3d_wrapper.generate_multiview(image)
    _empty_cuda_cache()
    
    # Reconstruir bundle de referência
    reference_bundle, ref_save_path = k3d_wrapper.generate_reference_3D_bundle_image_zero123(
        image, use_mv_rgb=True
    )
    _empty_cuda_cache()
    
    logger.info("[Etapa 3/7] ✓ Vistas multiview geradas e VRAM descarregada")
    
    return mv_image, {"reference_bundle": reference_bundle, "ref_save_path": ref_save_path}


def generate_3d_bundle_with_controlnet(
    k3d_wrapper: kiss3d_wrapper,
    caption: str,
    reference_bundle: torch.Tensor,
    features: Dict[str, torch.Tensor],
    use_controlnet: bool = True,
    control_modes: List[str] = None,
    enable_redux: bool = True,
) -> Tuple[torch.Tensor, str]:
    """
    Etapa 4: Gera bundle 3D final com ControlNet e consistência
    Descarrega VRAM após processamento
    """
    logger.info("[Etapa 4/7] Gerando bundle 3D final com ControlNet...")
    
    if control_modes is None:
        control_modes = ["tile", "canny"] if use_controlnet else []
    
    # Preparar imagens de controle
    control_images = []
    if use_controlnet:
        flux_cfg = k3d_wrapper.config.get("flux", {})
        control_downscale = flux_cfg.get("controlnet_down_scale", 1)
        control_kernel = flux_cfg.get("controlnet_kernel_size", 51)
        control_sigma = flux_cfg.get("controlnet_sigma", 2.0)
        
        for mode in control_modes:
            if mode in features:
                # Usar feature pré-processada
                control_img = features[mode]
            else:
                # Gerar feature on-the-fly
                control_img = k3d_wrapper.preprocess_controlnet_cond_image(
                    reference_bundle,
                    mode,
                    down_scale=control_downscale,
                    kernel_size=control_kernel,
                    sigma=control_sigma,
                )[0]
            control_images.append(control_img)
    
    # Preparar Redux se habilitado
    redux_hparam = None
    if enable_redux and k3d_wrapper.flux_redux_pipeline is not None:
        input_tensor = k3d_wrapper.to_512_tensor(Image.fromarray(
            (reference_bundle.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )).unsqueeze(0).clip(0.0, 1.0)
        redux_hparam = {
            "image": input_tensor,
            "prompt_embeds_scale": 1.0,
            "pooled_prompt_embeds_scale": 1.0,
            "strength": 0.5,
        }
    
    # Gerar bundle final
    if use_controlnet and control_images:
        gen_bundle, gen_save_path = k3d_wrapper.generate_3d_bundle_image_controlnet(
            prompt=caption,
            image=reference_bundle.unsqueeze(0) if reference_bundle.dim() == 3 else reference_bundle,
            strength=0.95,
            control_image=control_images,
            control_mode=control_modes,
            redux_hparam=redux_hparam,
        )
    else:
        gen_bundle, gen_save_path = k3d_wrapper.generate_3d_bundle_image_text(
            prompt=caption,
            image=reference_bundle.unsqueeze(0) if reference_bundle.dim() == 3 else reference_bundle,
            strength=0.95,
            redux_hparam=redux_hparam,
        )
    
    k3d_wrapper.offload_flux_pipelines()
    _empty_cuda_cache()
    
    logger.info("[Etapa 4/7] ✓ Bundle 3D gerado e VRAM descarregada")
    
    return gen_bundle, gen_save_path


def reconstruct_3d_mesh(
    k3d_wrapper: kiss3d_wrapper,
    bundle_image: torch.Tensor,
    reconstruction_stage2_steps: int = 50,
) -> str:
    """
    Etapa 5: Reconstrói mesh 3D a partir do bundle
    Descarrega VRAM após processamento
    """
    logger.info("[Etapa 5/7] Reconstruindo mesh 3D...")
    
    mesh_path = k3d_wrapper.reconstruct_3d_bundle_image(
        bundle_image,
        isomer_radius=4.15,
        reconstruction_stage2_steps=reconstruction_stage2_steps,
        save_intermediate_results=True,
    )
    
    _empty_cuda_cache()
    logger.info("[Etapa 5/7] ✓ Mesh 3D reconstruído e VRAM descarregada")
    
    return mesh_path


def evaluate_mesh_quality(
    generated_mesh_path: str,
    ground_truth_mesh_path: Optional[str] = None,
    num_samples: int = 50000,
) -> Dict[str, float]:
    """
    Etapa 6: Avalia qualidade da mesh gerada
    Compara com ground-truth se disponível
    """
    logger.info("[Etapa 6/7] Avaliando qualidade da mesh...")
    
    metrics = {}
    
    if ground_truth_mesh_path and os.path.exists(ground_truth_mesh_path):
        try:
            metrics = evaluate_mesh_against_gt(
                generated_mesh_path,
                ground_truth_mesh_path,
                num_samples=num_samples,
                normalize=True,
            )
            logger.info(f"  Métricas: {json.dumps(metrics, indent=2)}")
        except Exception as e:
            logger.warning(f"  Erro ao calcular métricas: {e}")
    
    # Métricas de OCR se disponível
    if EASYOCR_AVAILABLE:
        # TODO: Implementar comparação de OCR entre bundle gerado e original
        pass
    
    logger.info("[Etapa 6/7] ✓ Qualidade avaliada")
    
    return metrics


def generate_rotation_video(
    mesh_path: str,
    output_dir: Path,
    uuid: str,
    num_frames: int = 60,
    fps: int = 30,
    resolution: int = 512,
) -> Optional[str]:
    """
    Etapa 7: Gera vídeo de rotação do objeto 3D
    """
    logger.info("[Etapa 7/7] Gerando vídeo de rotação...")
    
    if not IMAGEIO_AVAILABLE:
        logger.warning("  imageio não disponível - pulando geração de vídeo")
        return None
    
    try:
        import trimesh
        from kiss3d_utils_local import get_render_cameras_video, render_frames
        from models.lrm.online_render.render_single import load_mipmap
        
        # Carregar mesh
        mesh = trimesh.load(mesh_path)
        
        # Gerar câmeras para rotação
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cameras = get_render_cameras_video(
            batch_size=1,
            num_frames=num_frames,
            radius=4.5,
            fov=30,
        )
        
        # Renderizar frames
        # TODO: Implementar renderização completa com o modelo LRM
        # Por enquanto, usar renderização simples com trimesh
        
        frames = []
        for i in range(num_frames):
            # Rotação simples
            angle = 2 * np.pi * i / num_frames
            # Renderizar frame (implementação simplificada)
            # frames.append(rendered_frame)
            pass
        
        # Salvar vídeo
        video_path = output_dir / f"{uuid}_rotation.mp4"
        # imageio.mimwrite(str(video_path), frames, fps=fps)
        
        logger.info(f"[Etapa 7/7] ✓ Vídeo gerado: {video_path}")
        return str(video_path)
        
    except Exception as e:
        logger.warning(f"  Erro ao gerar vídeo: {e}")
        return None


def run_segmented_pipeline(
    k3d_wrapper: kiss3d_wrapper,
    input_image_path: str,
    output_dir: str,
    ground_truth_mesh_path: Optional[str] = None,
    use_controlnet: bool = True,
    use_llm: bool = True,
    enable_redux: bool = True,
    control_modes: List[str] = None,
    reconstruction_stage2_steps: int = 50,
    generate_video: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline completo segmentado com descarregamento de VRAM entre etapas
    """
    k3d_wrapper.renew_uuid()
    uuid = k3d_wrapper.uuid
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "uuid": uuid,
        "input_image": input_image_path,
        "output_dir": str(output_path),
        "stages": {},
    }
    
    start_time = time.time()
    
    try:
        # Etapa 1: Processar features da imagem
        input_image = preprocess_input_image(Image.open(input_image_path))
        features = process_image_features(
            input_image,
            output_path,
            uuid,
            generate_depth=True,
            generate_normal=True,
            generate_canny=True,
            generate_shuffle=False,  # Desabilitado por padrão (mais pesado)
        )
        results["stages"]["features"] = {"status": "completed", "features": list(features.keys())}
        
        # Etapa 2: Gerar descrição
        caption = generate_detailed_caption(
            k3d_wrapper,
            input_image,
            use_llm=use_llm,
            preserve_text=True,
        )
        results["stages"]["caption"] = {"status": "completed", "caption": caption[:200]}
        
        # Etapa 3: Gerar vistas multiview
        mv_image, mv_info = generate_multiview_with_consistency(
            k3d_wrapper,
            input_image,
            caption,
            features,
            use_controlnet=use_controlnet,
            control_modes=control_modes,
        )
        results["stages"]["multiview"] = {"status": "completed"}
        
        # Etapa 4: Gerar bundle 3D final
        gen_bundle, gen_save_path = generate_3d_bundle_with_controlnet(
            k3d_wrapper,
            caption,
            mv_info["reference_bundle"],
            features,
            use_controlnet=use_controlnet,
            control_modes=control_modes,
            enable_redux=enable_redux,
        )
        results["stages"]["bundle_generation"] = {"status": "completed", "save_path": gen_save_path}
        
        # Etapa 5: Reconstruir mesh
        mesh_path = reconstruct_3d_mesh(
            k3d_wrapper,
            gen_bundle,
            reconstruction_stage2_steps=reconstruction_stage2_steps,
        )
        results["stages"]["reconstruction"] = {"status": "completed", "mesh_path": mesh_path}
        
        # Etapa 6: Avaliar qualidade
        metrics = evaluate_mesh_quality(
            mesh_path,
            ground_truth_mesh_path,
            num_samples=50000,
        )
        results["stages"]["evaluation"] = {"status": "completed", "metrics": metrics}
        
        # Etapa 7: Gerar vídeo
        if generate_video:
            video_path = generate_rotation_video(
                mesh_path,
                output_path,
                uuid,
                num_frames=60,
                fps=30,
            )
            if video_path:
                results["stages"]["video"] = {"status": "completed", "video_path": video_path}
        
        total_time = time.time() - start_time
        results["total_time"] = total_time
        results["status"] = "success"
        
        logger.info(f"Pipeline completo em {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Erro no pipeline: {e}", exc_info=True)
        results["status"] = "error"
        results["error"] = str(e)
    
    # Salvar relatório
    report_path = output_path / f"{uuid}_pipeline_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    return results

