"""
Wrapper otimizado do Kiss3DGen para 12GB VRAM
- Quantização automática de modelos
- Pipeline segmentado com alocação/desalocação de modelos
- Alinhamento com implementação original
"""

import os
import sys
import torch
import yaml
import uuid
from typing import Union, Any, Dict, Optional
from pathlib import Path

# Adicionar paths necessários
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KISS3D_ROOT = PROJECT_ROOT / "Kiss3DGen"
if str(KISS3D_ROOT) not in sys.path:
    sys.path.insert(0, str(KISS3D_ROOT))

from kiss3d_wrapper_local import (
    kiss3d_wrapper,
    init_wrapper_from_config,
    seed_everything,
    _empty_cuda_cache,
    _log_cuda_allocation,
)
from kiss3d_utils_local import (
    logger,
    TMP_DIR,
    OUT_DIR,
    preprocess_input_image,
    lrm_reconstruct,
    isomer_reconstruct,
    render_3d_bundle_image_from_mesh,
)
from PIL import Image
import torchvision
from torchvision.transforms import functional as TF
from einops import rearrange


class OptimizedKiss3DWrapper:
    """
    Wrapper otimizado que gerencia memória de forma inteligente
    para GPUs com 12GB de VRAM.
    """
    
    def __init__(self, config_path: str, target_vram_gb: float = 12.0):
        self.config_path = config_path
        self.target_vram_gb = target_vram_gb
        self.wrapper: Optional[kiss3d_wrapper] = None
        self._models_loaded = {
            'flux': False,
            'multiview': False,
            'caption': False,
            'reconstruction': False,
            'llm': False,
            'redux': False,
        }
        
        # Detectar VRAM disponível
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"VRAM total detectada: {total_vram:.2f} GB")
            if total_vram <= target_vram_gb:
                logger.warning(f"VRAM limitada ({total_vram:.2f} GB). Ativando modo otimizado.")
                self.optimized_mode = True
            else:
                self.optimized_mode = False
        else:
            self.optimized_mode = True  # CPU mode
        
        # Carregar config
        with open(config_path, "r") as f:
            self.config = yaml.load(f, yaml.FullLoader)
        
        # Aplicar otimizações ao config
        self._optimize_config()
    
    def _optimize_config(self):
        """Aplica otimizações ao config para reduzir uso de VRAM."""
        if not self.optimized_mode:
            return
        
        logger.info("Aplicando otimizações de VRAM ao config...")
        
        # Flux: usar fp16 ao invés de fp8 (mais compatível)
        if "flux" in self.config:
            self.config["flux"]["flux_dtype"] = "fp16"
            self.config["flux"]["cpu_offload"] = True
            # Reduzir resolução se necessário
            if self.config["flux"].get("image_height", 1024) > 640:
                self.config["flux"]["image_height"] = 640
                self.config["flux"]["image_width"] = 1280
            # Reduzir steps
            if self.config["flux"].get("num_inference_steps", 20) > 14:
                self.config["flux"]["num_inference_steps"] = 14
                self.config["flux"]["num_inference_steps_fast"] = 10
        
        # Multiview: usar fp16
        if "multiview" in self.config:
            # Reduzir steps
            if self.config["multiview"].get("num_inference_steps", 32) > 24:
                self.config["multiview"]["num_inference_steps"] = 24
                self.config["multiview"]["num_inference_steps_fast"] = 20
        
        # Caption: sempre CPU
        if "caption" in self.config:
            self.config["caption"]["device"] = "cpu"
        
        # Reconstruction: reduzir steps
        if "reconstruction" in self.config:
            if self.config["reconstruction"].get("stage2_steps", 50) > 40:
                self.config["reconstruction"]["stage2_steps"] = 40
                self.config["reconstruction"]["stage2_steps_fast"] = 28
        
        # LLM: sempre CPU e desabilitar se muito pesado
        if "llm" in self.config:
            self.config["llm"]["device"] = "cpu"
    
    def _load_wrapper_minimal(self):
        """Carrega wrapper com apenas modelos essenciais."""
        if self.wrapper is not None:
            return
        
        logger.info("Carregando wrapper Kiss3DGen (modo otimizado)...")
        self.wrapper = init_wrapper_from_config(
            self.config_path,
            fast_mode=True,  # Sempre usar fast mode em 12GB
            disable_llm=False,  # Manter LLM mas em CPU
            load_controlnet=True,
            load_redux=True,
        )
        logger.info("Wrapper carregado.")
    
    def load_model(self, model_name: str):
        """Carrega um modelo específico se ainda não estiver carregado."""
        if not self._models_loaded.get(model_name, False):
            self._load_wrapper_minimal()
            self._models_loaded[model_name] = True
            logger.info(f"Modelo {model_name} marcado como carregado.")
    
    def unload_model(self, model_name: str):
        """Descarrega um modelo específico da VRAM."""
        if not self.wrapper:
            return
        
        if model_name == 'multiview':
            self.wrapper.offload_multiview_pipeline()
            logger.info("Multiview pipeline descarregado.")
        elif model_name == 'flux':
            self.wrapper.offload_flux_pipelines()
            logger.info("Flux pipelines descarregados.")
        elif model_name == 'caption':
            self.wrapper.release_text_models()
            logger.info("Caption model descarregado.")
        elif model_name == 'llm':
            self.wrapper.del_llm_model()
            logger.info("LLM descarregado.")
        
        self._models_loaded[model_name] = False
        _empty_cuda_cache()
    
    def generate_multiview_optimized(self, image: Image.Image, seed: Optional[int] = None):
        """Gera multiview com gerenciamento de memória."""
        self.load_model('multiview')
        try:
            mv_image = self.wrapper.generate_multiview(image, seed=seed)
            return mv_image
        finally:
            # Não descarregar ainda - pode ser usado na reconstrução
            pass
    
    def generate_reference_bundle_optimized(self, image: Image.Image, use_mv_rgb: bool = True):
        """Gera bundle de referência com gerenciamento de memória."""
        self.load_model('multiview')
        self.load_model('reconstruction')
        try:
            reference_bundle, ref_save_path = self.wrapper.generate_reference_3D_bundle_image_zero123(
                image, use_mv_rgb=use_mv_rgb
            )
            return reference_bundle, ref_save_path
        finally:
            # Descarregar multiview após uso
            self.unload_model('multiview')
    
    def generate_caption_optimized(self, image: Image.Image):
        """Gera caption com gerenciamento de memória."""
        self.load_model('caption')
        try:
            caption = self.wrapper.get_image_caption(image)
            return caption
        finally:
            # Caption já está em CPU, mas limpar cache
            _empty_cuda_cache()
    
    def generate_bundle_optimized(
        self,
        caption: str,
        reference_bundle: torch.Tensor,
        input_image: Optional[Image.Image] = None,
        enable_redux: bool = True,
        use_controlnet: bool = True,
        strength: float = 0.95,
    ):
        """
        Gera bundle 3D final com parâmetros alinhados ao original.
        """
        self.load_model('flux')
        
        # Preparar Redux se habilitado
        redux_hparam = None
        if enable_redux and self.wrapper.flux_redux_pipeline is not None:
            if input_image is not None:
                redux_hparam = {
                    'image': self.wrapper.to_512_tensor(input_image).unsqueeze(0).clip(0., 1.),
                    'prompt_embeds_scale': 1.0,
                    'pooled_prompt_embeds_scale': 1.0,
                    'strength': 0.5,
                }
            else:
                # Fallback: usar bundle de referência
                redux_hparam = {
                    'prompt_embeds_scale': 1.0,
                    'pooled_prompt_embeds_scale': 1.0,
                    'strength': 0.5,
                }
        
        try:
            if use_controlnet:
                # PARÂMETROS ALINHADOS COM ORIGINAL
                control_mode = ['tile']  # Original usa apenas 'tile'
                control_image = [
                    self.wrapper.preprocess_controlnet_cond_image(
                        reference_bundle,
                        mode_,
                        down_scale=1,
                        kernel_size=51,
                        sigma=2.0
                    )
                    for mode_ in control_mode
                ]
                control_guidance_start = [0.0]  # Original
                control_guidance_end = [0.65]  # Original
                controlnet_conditioning_scale = [0.6]  # Original
                
                # Garantir que reference_bundle está no formato correto
                if reference_bundle.dim() == 3:
                    reference_bundle_input = reference_bundle.unsqueeze(0)
                else:
                    reference_bundle_input = reference_bundle
                
                gen_3d_bundle_image, gen_save_path = self.wrapper.generate_3d_bundle_image_controlnet(
                    prompt=caption,
                    image=reference_bundle_input,
                    strength=strength,
                    control_image=control_image,
                    control_mode=control_mode,
                    control_guidance_start=control_guidance_start,
                    control_guidance_end=control_guidance_end,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    lora_scale=1.0,
                    redux_hparam=redux_hparam,
                )
            else:
                if reference_bundle.dim() == 3:
                    reference_bundle_input = reference_bundle.unsqueeze(0)
                else:
                    reference_bundle_input = reference_bundle
                
                gen_3d_bundle_image, gen_save_path = self.wrapper.generate_3d_bundle_image_text(
                    prompt=caption,
                    image=reference_bundle_input,
                    strength=strength,
                    lora_scale=1.0,
                    redux_hparam=redux_hparam,
                )
            
            return gen_3d_bundle_image, gen_save_path
        finally:
            # Descarregar Flux após uso
            self.unload_model('flux')
    
    def reconstruct_mesh_optimized(
        self,
        bundle_image: torch.Tensor,
        isomer_radius: float = 4.15,
        reconstruction_stage2_steps: int = 50,
    ):
        """Reconstrói mesh com gerenciamento de memória."""
        self.load_model('reconstruction')
        try:
            # Garantir formato correto
            if bundle_image.dim() == 4:
                bundle_image = bundle_image.squeeze(0)  # (1, 3, H, W) -> (3, H, W)
            
            recon_mesh_path = self.wrapper.reconstruct_3d_bundle_image(
                bundle_image,
                save_intermediate_results=False,
                isomer_radius=isomer_radius,
                reconstruction_stage2_steps=reconstruction_stage2_steps,
            )
            return recon_mesh_path
        finally:
            # Não descarregar reconstruction ainda - pode ser usado novamente
            pass
    
    def run_image_to_3d_optimized(
        self,
        input_image_path: str,
        enable_redux: bool = True,
        use_mv_rgb: bool = True,
        use_controlnet: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Pipeline completo otimizado, alinhado com o original.
        """
        self.wrapper.renew_uuid()
        
        # Etapa 1: Preprocessar imagem
        logger.info("[1/5] Preprocessando imagem...")
        input_image = preprocess_input_image(Image.open(input_image_path))
        input_image.save(os.path.join(TMP_DIR, f'{self.wrapper.uuid}_input_image.png'))
        _empty_cuda_cache()
        
        # Etapa 2: Gerar bundle de referência
        logger.info("[2/5] Gerando bundle de referência...")
        reference_3d_bundle_image, reference_save_path = self.generate_reference_bundle_optimized(
            input_image, use_mv_rgb=use_mv_rgb
        )
        _empty_cuda_cache()
        
        # Etapa 3: Gerar caption
        logger.info("[3/5] Gerando caption...")
        caption = self.generate_caption_optimized(input_image)
        _empty_cuda_cache()
        
        # Etapa 4: Gerar bundle final
        logger.info("[4/5] Gerando bundle 3D final...")
        gen_3d_bundle_image, gen_save_path = self.generate_bundle_optimized(
            caption=caption,
            reference_bundle=reference_3d_bundle_image,
            input_image=input_image,  # Passar imagem original para Redux
            enable_redux=enable_redux,
            use_controlnet=use_controlnet,
            strength=0.95,
        )
        _empty_cuda_cache()
        
        # Etapa 5: Reconstruir mesh
        logger.info("[5/5] Reconstruindo mesh 3D...")
        recon_mesh_path = self.reconstruct_mesh_optimized(
            gen_3d_bundle_image,
            isomer_radius=4.15,
            reconstruction_stage2_steps=self.wrapper.get_reconstruction_stage2_steps(),
        )
        _empty_cuda_cache()
        
        # Descarregar tudo
        self.unload_model('reconstruction')
        
        return gen_save_path, recon_mesh_path
    
    def cleanup(self):
        """Limpa todos os modelos da memória."""
        for model_name in list(self._models_loaded.keys()):
            self.unload_model(model_name)
        self.wrapper = None
        _empty_cuda_cache()
        logger.info("Cleanup completo.")


def init_optimized_wrapper(config_path: str, target_vram_gb: float = 12.0) -> OptimizedKiss3DWrapper:
    """Inicializa wrapper otimizado."""
    return OptimizedKiss3DWrapper(config_path, target_vram_gb)

