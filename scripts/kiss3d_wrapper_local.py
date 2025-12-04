import os
import sys
import numpy as np
import random
import torch
import yaml
import uuid
import warnings
import shutil
from typing import Union, Any, Dict, TYPE_CHECKING, List
from einops import rearrange
from PIL import Image
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from models.llm.llm import load_llm_model, get_llm_response
from diffusers import FluxPipeline, DiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from diffusers.schedulers import FlowMatchHeunDiscreteScheduler
from Kiss3DGen.pipeline.custom_pipelines.pipeline_flux_prior_redux import FluxPriorReduxPipeline
from Kiss3DGen.pipeline.custom_pipelines.pipeline_flux_controlnet_image_to_image import (
    FluxControlNetImg2ImgPipeline,
)
from Kiss3DGen.pipeline.custom_pipelines.pipeline_flux_img2img import FluxImg2ImgPipeline
try:
    import cv2
except Exception:
    cv2 = None

from huggingface_hub import hf_hub_download

from omegaconf import OmegaConf
from models.lrm.utils.train_util import instantiate_from_config

try:
    import trimesh
except Exception:
    trimesh = None

from kiss3d_utils_local import (
    logger,
    TMP_DIR,
    OUT_DIR,
    preprocess_input_image,
    lrm_reconstruct,
    isomer_reconstruct,
    render_3d_bundle_image_from_mesh,
    KISS3D_ROOT,
    PROJECT_ROOT,
)

# Importar funções de logging se disponíveis
try:
    from setup_logging import log_memory_usage, log_model_operation
except ImportError:
    # Fallback se setup_logging não estiver disponível
    def log_memory_usage(logger, label="Memory check"):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logger.info(f"[MEMORY] {label} | Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")
    
    def log_model_operation(logger, operation, model_name, device=None, memory_mb=None):
        msg = f"[MODEL] {operation}: {model_name}"
        if device:
            msg += f" | Device: {device}"
        if memory_mb is not None:
            msg += f" | Memory: {memory_mb:.1f} MB"
        logger.info(msg)

if TYPE_CHECKING:
    from Kiss3DGen.pipeline.custom_pipelines.pipeline_flux_prior_redux import FluxPriorReduxPipeline
    from Kiss3DGen.pipeline.custom_pipelines.pipeline_flux_controlnet_image_to_image import (
        FluxControlNetImg2ImgPipeline
    )

# Suprimir warnings comuns e não críticos
warnings.filterwarnings('ignore', message='.*Some weights of.*were not used.*')
warnings.filterwarnings('ignore', message='.*text_projection.*')
warnings.filterwarnings('ignore', message='.*add_prefix_space.*')
warnings.filterwarnings('ignore', message='.*The tokenizer.*needs to be converted.*')
warnings.filterwarnings('ignore', message='.*TRANSFORMERS_CACHE.*')
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')
warnings.filterwarnings('ignore', message='.*_get_vc_env is private.*')


def convert_flux_pipeline(exist_flux_pipe, target_pipe, **kwargs):
    new_pipe = target_pipe(
        scheduler=exist_flux_pipe.scheduler,
        vae=exist_flux_pipe.vae,
        text_encoder=exist_flux_pipe.text_encoder,
        tokenizer=exist_flux_pipe.tokenizer,
        text_encoder_2=exist_flux_pipe.text_encoder_2,
        tokenizer_2=exist_flux_pipe.tokenizer_2,
        transformer=exist_flux_pipe.transformer,
        **kwargs,
    )
    return new_pipe


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")


def _empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _log_cuda_allocation(device, label):
    """Log de alocação CUDA com informações completas"""
    if (
        torch.cuda.is_available()
        and isinstance(device, str)
        and device.startswith("cuda")
    ):
        device_idx = int(device.split(":")[-1]) if ":" in device else 0
        allocated = torch.cuda.memory_allocated(device_idx) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device_idx) / 1024**3  # GB
        total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3  # GB
        free = total - reserved
        logger.info(
            f"GPU memory after {label} on {device}: "
            f"Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, "
            f"Free={free:.2f}GB, Total={total:.2f}GB"
        )


class kiss3d_wrapper(object):
    def __init__(
        self,
        config: Dict,
        flux_pipeline: Union[FluxPipeline, "FluxControlNetImg2ImgPipeline"],
        flux_redux_pipeline: "FluxPriorReduxPipeline",
        multiview_pipeline: DiffusionPipeline,
        caption_processor: AutoProcessor,
        caption_model: AutoModelForCausalLM,
        reconstruction_model_config: Any,
        reconstruction_model: Any,
        llm_model: AutoModelForCausalLM = None,
        llm_tokenizer: AutoTokenizer = None,
        fast_mode: bool = False,
    ):
        self.config = config
        self.flux_pipeline = flux_pipeline
        self.flux_redux_pipeline = flux_redux_pipeline
        self.multiview_pipeline = multiview_pipeline
        self.caption_model = caption_model
        self.caption_processor = caption_processor
        self.recon_model_config = reconstruction_model_config
        self.recon_model = reconstruction_model
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.fast_mode = fast_mode
        self._recon_stage2 = self.config["reconstruction"].get("stage2_steps", 50)
        fast_default = max(24, self._recon_stage2 // 2)
        self._recon_stage2_fast = min(
            self._recon_stage2,
            self.config["reconstruction"].get("stage2_steps_fast", fast_default),
        )
        self.flux_height = self.config["flux"].get("image_height", 1024)
        self.flux_width = self.config["flux"].get("image_width", 2048)
        self.multiview_height = self.config["multiview"].get("image_height", 1024)
        self.multiview_width = self.config["multiview"].get("image_width", self.multiview_height)
        self.multiview_rows = self.config["multiview"].get("grid_rows", 2)
        self.multiview_cols = self.config["multiview"].get("grid_cols", 2)

        self.to_512_tensor = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((512, 512), interpolation=2),
            ]
        )

        self.renew_uuid()

        self._clip_text_encoder_cpu = None
        self._clip_tokenizer_cpu = None
        self._t5_text_encoder_cpu = None
        self._t5_tokenizer_cpu = None

    def get_reconstruction_stage2_steps(self):
        return self._recon_stage2_fast if self.fast_mode else self._recon_stage2

    def renew_uuid(self):
        self.uuid = uuid.uuid4()

    @staticmethod
    def _finalize_lrm_outputs(lrm_obj_path: str, final_obj_path: str, final_glb_path: str, fallback_obj: str | None = None, fallback_glb: str | None = None):
        os.makedirs(os.path.dirname(final_obj_path), exist_ok=True)
        os.makedirs(os.path.dirname(final_glb_path), exist_ok=True)

        if lrm_obj_path and os.path.exists(lrm_obj_path):
            shutil.copy2(lrm_obj_path, final_obj_path)
            logger.info(f"[RECON] OBJ principal atualizado a partir do LRM: {final_obj_path}")
        elif fallback_obj and os.path.exists(fallback_obj):
            shutil.copy2(fallback_obj, final_obj_path)
            logger.warning(f"[RECON] LRM OBJ ausente, usando fallback ISOMER em {fallback_obj}")
        else:
            logger.error("[RECON] Nenhum OBJ disponível para promover")

        glb_promoted = False
        if os.path.exists(final_obj_path) and trimesh is not None:
            try:
                mesh = trimesh.load(final_obj_path, force='mesh', process=False)
                mesh.export(final_glb_path)
                glb_promoted = True
                logger.info(f"[RECON] GLB convertido do OBJ principal: {final_glb_path}")
            except Exception as exc:
                logger.warning(f"[RECON] Falha ao converter OBJ -> GLB ({exc}); usando fallback")

        if not glb_promoted and fallback_glb and os.path.exists(fallback_glb):
            shutil.copy2(fallback_glb, final_glb_path)
            logger.warning(f"[RECON] GLB principal usando fallback ISOMER em {fallback_glb}")
        elif not glb_promoted and fallback_glb is None:
            logger.error("[RECON] Nenhum GLB disponível para salvar")

    def context(self):
        if self.config["use_zero_gpu"]:
            pass
        else:
            return torch.no_grad()

    def _assert_no_duplicate_views(
        self,
        bundle_image: torch.Tensor,
        stage_name: str,
        similarity_threshold: float = 0.995,
        min_duplicate_pairs: int = 3,
        raise_on_fail: bool = True,
    ) -> bool:
        """
        Detects nearly identical RGB views (top row of the bundle grid).
        If many views are almost identical, the downstream reconstruction
        produces a duplicated mesh, so we abort early for manual inspection.
        """
        try:
            if bundle_image is None:
                return

            tensor = bundle_image.detach().float().cpu()
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.dim() != 3:
                logger.warning(
                    "[DUPLICATE_CHECK] Ignorando bundle com shape inesperado %s no estágio %s",
                    tuple(tensor.shape),
                    stage_name,
                )
                return

            channels, height, width = tensor.shape
            if channels < 3 or height < 2 or width < 4:
                return

            if height % 2 != 0 or width % 4 != 0:
                logger.warning(
                    "[DUPLICATE_CHECK] Bundle não está no layout 2x4 esperado no estágio %s (shape=%s)",
                    stage_name,
                    tuple(tensor.shape),
                )
                return

            views = rearrange(
                tensor,
                "c (rows h) (cols w) -> (rows cols) c h w",
                rows=2,
                cols=4,
            )
            rgb_views = views[:4].reshape(4, -1)
            rgb_views = rgb_views - rgb_views.mean(dim=1, keepdim=True)
            rgb_views = F.normalize(rgb_views, dim=1, eps=1e-6)
            sim_matrix = torch.matmul(rgb_views, rgb_views.T)
            tri_indices = torch.triu_indices(sim_matrix.size(0), sim_matrix.size(1), offset=1)
            pair_similarities = sim_matrix[tri_indices[0], tri_indices[1]]

            duplicate_pairs = int((pair_similarities > similarity_threshold).sum().item())
            pixel_std = float(rgb_views.std(dim=1).mean().item())

            if duplicate_pairs >= min_duplicate_pairs:
                message = (
                    f"[DUPLICATE_DETECTED] {stage_name}: {duplicate_pairs}/6 pares de vistas "
                    f"com similaridade > {similarity_threshold:.3f} (std médio {pixel_std:.6f}). "
                    "Abandonando pipeline para evitar malha duplicada."
                )
                if raise_on_fail:
                    logger.error(message)
                    raise RuntimeError(message)
                logger.warning(message + " Prosseguindo mediante tentativa de aleatorização.")
                return True
            return False
        except RuntimeError:
            raise
        except Exception as exc:  # best-effort detector; não bloquear pipeline em erro inesperado
            logger.warning(
                "[DUPLICATE_CHECK] Falha ao avaliar duplicação no estágio %s: %s",
                stage_name,
                exc,
            )
        return False

    @staticmethod
    def _fake_normals_from_view(view: torch.Tensor) -> torch.Tensor:
        normals = view.clone()
        min_vals = normals.amin(dim=(-2, -1), keepdim=True)
        max_vals = normals.amax(dim=(-2, -1), keepdim=True)
        denom = (max_vals - min_vals).clamp_min(1e-6)
        normals = (normals - min_vals) / denom
        normals = normals * 0.6 + 0.2
        return normals

    def _compose_seed_grid_from_views(
        self,
        variations: list[torch.Tensor],
        view_h: int,
        view_w: int,
        flux_device: torch.device,
    ) -> torch.Tensor:
        dtype = variations[0].dtype if variations else torch.float32
        seed_grid = torch.zeros(
            (1, 3, self.flux_height, self.flux_width),
            device=flux_device,
            dtype=dtype,
        )

        for idx, view in enumerate(variations):
            view_device = view.to(flux_device)
            col = idx * view_w
            seed_grid[:, :, :view_h, col : col + view_w] = view_device.unsqueeze(0)
            normals = self._fake_normals_from_view(view_device)
            seed_grid[:, :, view_h:, col : col + view_w] = normals.unsqueeze(0)

        return seed_grid

    def _generate_seed_variations(
        self,
        base: torch.Tensor,
        attempt_idx: int,
        noise_strength: float,
    ) -> list[torch.Tensor]:
        variations: list[torch.Tensor] = []
        device = base.device
        hue_delta = 0.08 + 0.02 * attempt_idx
        sat_scale = 1.15 + 0.05 * attempt_idx
        for idx in range(4):
            view = base.clone()
            if idx == 1:
                view = TF.hflip(view)
            elif idx == 2:
                view = TF.adjust_hue(view, hue_delta)
            elif idx == 3:
                view = TF.adjust_saturation(view, sat_scale)

            if attempt_idx > 0:
                angle = float((torch.rand(1, device=device).item() - 0.5) * 6.0 * attempt_idx)
                max_shift = max(1, int(2 * attempt_idx))
                dx = int((torch.rand(1, device=device).item() - 0.5) * max_shift)
                dy = int((torch.rand(1, device=device).item() - 0.5) * max_shift)
                view = TF.affine(
                    view,
                    angle=angle,
                    translate=[dx, dy],
                    scale=1.0,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )
                contrast = 1.0 + (torch.rand(1, device=device).item() - 0.5) * 0.3 * attempt_idx
                view = TF.adjust_contrast(view, contrast)
                brightness = 1.0 + (torch.rand(1, device=device).item() - 0.5) * 0.25 * attempt_idx
                view = TF.adjust_brightness(view, brightness)

            if noise_strength > 0:
                jitter = noise_strength * max(1, attempt_idx + 1)
                noise = torch.randn_like(view) * jitter
                view = (view + noise).clamp(0.0, 1.0)

            variations.append(view.clamp(0.0, 1.0))

        return variations

    def _move_mesh_to_device(self, vertices, faces, device):
        def _to_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.to(device)
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data).to(device)
            raise TypeError(f"Tipo inesperado para mesh: {type(data)}")

        return _to_tensor(vertices), _to_tensor(faces)

    def _ensure_flux_text_backbone_cpu(self):
        if self._clip_text_encoder_cpu is not None:
            return

        flux_cfg = self.config.get("flux", {})
        repo_id = flux_cfg.get("fallback_base_model") or "black-forest-labs/FLUX.1-dev"
        logger.info("[FLUX] Carregando text encoders CPU auxiliares de %s", repo_id)

        self._clip_text_encoder_cpu = CLIPTextModel.from_pretrained(
            repo_id,
            subfolder="text_encoder",
            torch_dtype=torch.float32,
        ).eval()
        self._clip_tokenizer_cpu = CLIPTokenizer.from_pretrained(
            repo_id,
            subfolder="tokenizer",
        )
        self._t5_text_encoder_cpu = T5EncoderModel.from_pretrained(
            repo_id,
            subfolder="text_encoder_2",
            torch_dtype=torch.float32,
        ).eval()
        self._t5_tokenizer_cpu = T5TokenizerFast.from_pretrained(
            repo_id,
            subfolder="tokenizer_2",
        )

    def _encode_prompt_cpu(self, prompt: str, prompt_2: str, num_images_per_prompt: int):
        self._ensure_flux_text_backbone_cpu()

        clip_tokenizer = self._clip_tokenizer_cpu
        clip_encoder = self._clip_text_encoder_cpu
        t5_tokenizer = self._t5_tokenizer_cpu
        t5_encoder = self._t5_text_encoder_cpu

        clip_inputs = clip_tokenizer(
            prompt,
            padding="max_length",
            max_length=clip_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            clip_out = clip_encoder(clip_inputs.input_ids, output_hidden_states=False)
        pooled_prompt_embeds = clip_out.pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(-1, pooled_prompt_embeds.shape[-1])

        t5_prompt = prompt_2 or prompt
        t5_inputs = t5_tokenizer(
            t5_prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            t5_out = t5_encoder(t5_inputs.input_ids, output_hidden_states=False)[0]
        seq_len = t5_out.shape[1]
        prompt_embeds = t5_out.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(-1, seq_len, t5_out.shape[-1])

        return prompt_embeds, pooled_prompt_embeds

    def _build_flux_seed_bundle(self, input_image: Image.Image) -> torch.Tensor:
        flux_device = torch.device(self.config["flux"].get("device", "cpu"))
        view_h = max(64, self.flux_height // 2)
        view_w = max(64, self.flux_width // 4)

        base = TF.to_tensor(input_image)
        base = TF.resize(base, [view_h, view_w], antialias=True).clamp(0.0, 1.0)

        flux_cfg = self.config.get("flux", {})
        max_retries = int(flux_cfg.get("duplicate_retry_attempts", 3))
        noise_strength = float(flux_cfg.get("duplicate_noise_strength", 0.03))
        max_retries = max(0, max_retries)
        last_seed_grid = None

        for attempt in range(max_retries + 1):
            variations = self._generate_seed_variations(base, attempt, noise_strength)
            seed_grid = self._compose_seed_grid_from_views(variations, view_h, view_w, flux_device)
            has_duplicates = self._assert_no_duplicate_views(
                seed_grid.squeeze(0),
                stage_name="flux_seed_bundle",
                similarity_threshold=0.999,
                raise_on_fail=False,
            )
            if not has_duplicates:
                return seed_grid.clamp(0.0, 1.0)

            logger.warning(
                "[DUPLICATE_CHECK] flux_seed_bundle ainda possui vistas similares (tentativa %d/%d). "
                "Aplicando jitter adicional no seed.",
                attempt + 1,
                max_retries,
            )
            last_seed_grid = seed_grid

        logger.warning(
            "[DUPLICATE_CHECK] Prosseguindo com flux_seed_bundle após %d tentativas; vistas podem permanecer semelhantes.",
            max_retries + 1,
        )
        return (last_seed_grid or seed_grid).clamp(0.0, 1.0)

    def get_image_caption(self, image):
        if self.caption_model is None:
            raise RuntimeError("Caption model is None! Make sure it was loaded correctly.")
        
        caption_device = self.config["caption"].get("device", "cpu")
        torch_dtype = (
            torch.bfloat16
            if isinstance(caption_device, str) and caption_device.startswith("cuda")
            else torch.float32
        )

        if isinstance(image, str):  # If image is a file path
            image = preprocess_input_image(Image.open(image))
        elif not isinstance(image, Image.Image):
            raise NotImplementedError("unexpected image type")

        # Garantir que o caption model está no device correto
        if hasattr(self.caption_model, 'device'):
            model_device = next(self.caption_model.parameters()).device
            if str(model_device) != caption_device:
                logger.info(f"[CAPTION] Movendo caption model de {model_device} para {caption_device}")
                self.caption_model.to(caption_device)

        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.caption_processor(text=prompt, images=image, return_tensors="pt").to(
            caption_device, torch_dtype
        )

        generated_ids = self.caption_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )

        generated_text = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.caption_processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        caption_text = parsed_answer["<MORE_DETAILED_CAPTION>"]  # .replace("The image is ", "")

        logger.info(f'Auto caption result: "{caption_text}"')

        caption_text = self.get_detailed_prompt(caption_text)

        return caption_text

    def get_detailed_prompt(self, prompt, seed=None):
        if self.llm_model is not None:
            # System prompt para refinar o caption
            system_prompt = (
                "You are an expert 3D asset director. When the user provides a base caption, expand it into a precise, "
                "engineering-grade description that can guide a multi-view image-to-3D pipeline. Focus exclusively on the object itself "
                "(no environments or backgrounds). Describe geometry, materials, logos or labels, manufacturing details, wear, and any "
                "asymmetries. Explicitly mention what should be visible from the front, left, rear, and right views so that a 2×4 RGB/normal "
                "grid can be rendered. Use complete sentences in a single paragraph, favoring factual language over stylistic flourishes. "
                "Do not invent accessories that were not implied by the caption; instead, clarify existing features. "
                "Return only the enriched description in English."
            )
            # User prompt é o caption base
            user_prompt = prompt
            detailed_prompt = get_llm_response(self.llm_model, self.llm_tokenizer, user_prompt, seed=seed, system_prompt=system_prompt)

            logger.info(f'LLM refined prompt result: "{detailed_prompt}"')
            return detailed_prompt
        return prompt

    def del_llm_model(self):
        if self.llm_model is not None:
            try:
                logger.info("[MODEL] Descarregando LLM model...")
                # Se o LLM é 'ollama' (string), não precisa fazer nada
                if self.llm_model == 'ollama':
                    logger.info("[MODEL] LLM é Ollama (API), não precisa descarregar")
                elif hasattr(self.llm_model, 'to'):
                    # Convert to float32 before moving to CPU to avoid float16 warnings
                    if hasattr(self.llm_model, 'dtype'):
                        model_dtype = getattr(self.llm_model, 'dtype', None)
                        if model_dtype in [torch.float16, torch.bfloat16]:
                            # Try to convert, but some models don't support it
                            try:
                                logger.debug(f"[MODEL] Convertendo LLM model de {model_dtype} para float32")
                                self.llm_model = self.llm_model.to(torch.float32)
                            except Exception as e:
                                logger.warning(f"[MODEL] Erro ao converter LLM model: {e}")
                    self.llm_model.to("cpu")
                    logger.info("[MODEL] LLM model descarregado para CPU")
            except Exception as e:
                logger.error(f"[MODEL] Erro ao descarregar LLM model: {e}")
        # Não definir como None se for Ollama - apenas limpar referências
        if self.llm_model != 'ollama':
            self.llm_model = None
        self.llm_tokenizer = None
        _empty_cuda_cache()

    def release_text_models(self):
        logger.info("[MODEL] Iniciando descarregamento de modelos de texto (caption + LLM)")
        log_memory_usage(logger, "Antes de descarregar modelos de texto")
        
        if self.caption_model is not None:
            try:
                logger.info("[MODEL] Descarregando caption model...")
                # Convert to float32 before moving to CPU to avoid float16 warnings
                if hasattr(self.caption_model, 'dtype'):
                    model_dtype = getattr(self.caption_model, 'dtype', None)
                    if model_dtype in [torch.float16, torch.bfloat16]:
                        try:
                            logger.debug(f"[MODEL] Convertendo caption model de {model_dtype} para float32")
                            self.caption_model = self.caption_model.to(torch.float32)
                        except Exception as e:
                            logger.warning(f"[MODEL] Erro ao converter caption model: {e}")
                self.caption_model.to("cpu")
                logger.info("[MODEL] Caption model descarregado para CPU")
            except Exception as e:
                logger.error(f"[MODEL] Erro ao descarregar caption model: {e}")
            # NÃO definir como None - apenas mover para CPU para poder reutilizar depois
            # self.caption_model = None
        self.del_llm_model()
        log_memory_usage(logger, "Após descarregar modelos de texto")

    def offload_multiview_pipeline(self):
        if self.multiview_pipeline is not None:
            try:
                logger.info("[MODEL] Descarregando multiview pipeline...")
                log_memory_usage(logger, "Antes de descarregar multiview pipeline")
                # Don't move float16 pipelines to CPU - they should use CPU offload
                pipeline_dtype = getattr(self.multiview_pipeline, 'dtype', None)
                if pipeline_dtype == torch.float16:
                    # Use CPU offload if available, otherwise skip
                    if hasattr(self.multiview_pipeline, 'enable_model_cpu_offload'):
                        logger.info("[MODEL] Habilitando CPU offload para multiview pipeline (float16)")
                        self.multiview_pipeline.enable_model_cpu_offload()
                    else:
                        logger.warning("[MODEL] Multiview pipeline é float16, pulando movimento para CPU para evitar warnings")
                else:
                    logger.info(f"[MODEL] Movendo multiview pipeline para CPU (dtype: {pipeline_dtype})")
                    self.multiview_pipeline.to("cpu")
                logger.info("[MODEL] Multiview pipeline descarregado")
                log_memory_usage(logger, "Após descarregar multiview pipeline")
            except Exception as e:
                logger.error(f"[MODEL] Erro ao descarregar multiview pipeline: {e}")
        _empty_cuda_cache()

    def offload_flux_pipelines(self):
        logger.info("[MODEL] Descarregando pipelines Flux...")
        log_memory_usage(logger, "Antes de descarregar pipelines Flux")
        # Flux pipelines already use CPU offload, don't move them directly
        # They are managed by enable_sequential_cpu_offload()
        if self.flux_pipeline is not None:
            logger.info("[MODEL] Flux pipeline já usa CPU offload sequencial, apenas limpando cache")
            # Just clear cache, pipeline is already offloaded
            pass
        if self.flux_redux_pipeline is not None:
            logger.info("[MODEL] Flux Redux pipeline já usa CPU offload sequencial, apenas limpando cache")
            # Just clear cache, pipeline is already offloaded
            pass
        _empty_cuda_cache()
        log_memory_usage(logger, "Após descarregar pipelines Flux")

    def generate_multiview(self, image, seed=None, num_inference_steps=None):
        if self.multiview_pipeline is None:
            raise RuntimeError(
                "multiview_pipeline é None! O pipeline Zero123++ não foi carregado. "
                "Certifique-se de que pipeline_mode='multiview' foi passado para init_wrapper_from_config."
            )
        
        seed = seed or self.config["multiview"].get("seed", 0)
        mv_device = self.config["multiview"].get("device", "cpu")

        # Garantir que o pipeline está completamente no device correto
        logger.info(f"[MULTIVIEW] Garantindo que pipeline está em {mv_device}")
        
        # Mover o pipeline inteiro para o device correto
        self.multiview_pipeline.to(mv_device)
        
        # Garantir que TODOS os componentes estão no device correto
        # Verificar e mover cada componente individualmente
        if hasattr(self.multiview_pipeline, 'vae'):
            vae_device = next(self.multiview_pipeline.vae.parameters()).device
            if str(vae_device) != mv_device:
                logger.info(f"[MULTIVIEW] Movendo VAE de {vae_device} para {mv_device}")
                self.multiview_pipeline.vae.to(mv_device)
                # Verificar novamente
                vae_device = next(self.multiview_pipeline.vae.parameters()).device
                logger.info(f"[MULTIVIEW] VAE agora em {vae_device}")
        
        if hasattr(self.multiview_pipeline, 'unet'):
            unet_device = next(self.multiview_pipeline.unet.parameters()).device
            if str(unet_device) != mv_device:
                logger.info(f"[MULTIVIEW] Movendo UNet de {unet_device} para {mv_device}")
                self.multiview_pipeline.unet.to(mv_device)
        
        if hasattr(self.multiview_pipeline, 'vision_encoder'):
            vision_device = next(self.multiview_pipeline.vision_encoder.parameters()).device
            if str(vision_device) != mv_device:
                logger.info(f"[MULTIVIEW] Movendo vision_encoder de {vision_device} para {mv_device}")
                self.multiview_pipeline.vision_encoder.to(mv_device)
        
        # Garantir que o text_encoder também está no device correto
        if hasattr(self.multiview_pipeline, 'text_encoder'):
            try:
                text_encoder_device = next(self.multiview_pipeline.text_encoder.parameters()).device
                if str(text_encoder_device) != mv_device:
                    logger.info(f"[MULTIVIEW] Movendo text_encoder de {text_encoder_device} para {mv_device}")
                    self.multiview_pipeline.text_encoder.to(mv_device)
                    # Verificar novamente
                    text_encoder_device = next(self.multiview_pipeline.text_encoder.parameters()).device
                    logger.info(f"[MULTIVIEW] text_encoder agora em {text_encoder_device}")
            except Exception as e:
                logger.warning(f"[MULTIVIEW] Erro ao verificar/mover text_encoder: {e}")
        
        # Garantir que o tokenizer também está configurado (não precisa de device, mas verificar)
        if hasattr(self.multiview_pipeline, 'tokenizer') and self.multiview_pipeline.tokenizer is None:
            logger.warning("[MULTIVIEW] Tokenizer está None!")
        
        # Verificar device final do VAE para o generator
        vae_device = next(self.multiview_pipeline.vae.parameters()).device
        generator = torch.Generator(device=str(vae_device)).manual_seed(seed)
        
        logger.info(f"[MULTIVIEW] Pipeline pronto, VAE em {vae_device}")
        
        with self.context():
            mv_image = self.multiview_pipeline(
                image,
                num_inference_steps=num_inference_steps or self.config["multiview"]["num_inference_steps"],
                width=self.multiview_width,
                height=self.multiview_height,
                generator=generator,
            ).images[0]
        return mv_image

    def reconstruct_from_multiview(self, mv_image, lrm_render_radius=4.15):
        recon_device = self.config["reconstruction"].get("device", "cpu")

        rgb_multi_view = np.asarray(mv_image, dtype=np.float32) / 255.0
        rgb_multi_view = (
            torch.from_numpy(rgb_multi_view).squeeze(0).permute(2, 0, 1).contiguous()
        )
        rgb_multi_view = rgb_multi_view.to(dtype=torch.float32)
        rgb_multi_view = rearrange(
            rgb_multi_view,
            "c (n h) (m w) -> (n m) c h w",
            n=self.multiview_rows,
            m=self.multiview_cols,
        ).unsqueeze(0).to(recon_device)

        with self.context():
            vertices, faces, lrm_multi_view_normals, lrm_multi_view_rgb, lrm_multi_view_albedo = lrm_reconstruct(
                self.recon_model,
                self.recon_model_config.infer_config,
                rgb_multi_view,
                name=self.uuid,
                render_radius=lrm_render_radius,
            )
        _empty_cuda_cache()

        return rgb_multi_view, vertices, faces, lrm_multi_view_normals, lrm_multi_view_rgb, lrm_multi_view_albedo

    def generate_reference_3D_bundle_image_zero123(self, image, use_mv_rgb=False, save_intermediate_results=True):
        mv_image = self.generate_multiview(image)

        if save_intermediate_results:
            mv_image.save(os.path.join(TMP_DIR, f"{self.uuid}_mv_image.png"))

        (
            rgb_multi_view,
            vertices,
            faces,
            lrm_multi_view_normals,
            lrm_multi_view_rgb,
            lrm_multi_view_albedo,
        ) = self.reconstruct_from_multiview(mv_image)

        if use_mv_rgb:
            # As câmeras de entrada do Zero123++ são [0, 90, 180, 270]
            # O rearrange organiza as vistas como: [0, 1, 2, 3] = [top-left, top-right, bottom-left, bottom-right]
            # Mas precisamos alinhar com os azimutes de renderização [270, 0, 90, 180]
            # Então a ordem correta é [3, 0, 1, 2] para corresponder aos azimutes
            # NO ENTANTO, se há duplicação, pode ser que o problema esteja na ordem das vistas
            # Vamos tentar usar a ordem original [0, 1, 2, 3] primeiro para ver se resolve
            ref_3D_bundle_image = torchvision.utils.make_grid(
                torch.cat(
                    [rgb_multi_view.squeeze(0).detach().cpu()[[0, 1, 2, 3]], (lrm_multi_view_normals.cpu() + 1) / 2],
                    dim=0,
                ),
                nrow=4,
                padding=0,
            )  # range [0, 1]
        else:
            ref_3D_bundle_image = torchvision.utils.make_grid(
                torch.cat([lrm_multi_view_rgb.cpu(), (lrm_multi_view_normals.cpu() + 1) / 2], dim=0),
                nrow=4,
                padding=0,
            )  # range [0, 1]

        ref_3D_bundle_image = ref_3D_bundle_image.clip(0.0, 1.0)

        self._assert_no_duplicate_views(
            ref_3D_bundle_image,
            stage_name="reference_bundle_image",
            raise_on_fail=False,
        )

        if save_intermediate_results:
            save_path = os.path.join(TMP_DIR, f"{self.uuid}_ref_3d_bundle_image.png")
            torchvision.utils.save_image(ref_3D_bundle_image, save_path)

            logger.info(f"Save reference 3D bundle image to {save_path}")

            return ref_3D_bundle_image, save_path

        return ref_3D_bundle_image

    def run_multiview_pipeline(
        self,
        input_image: Image.Image,
        reconstruction_stage2_steps: int | None = None,
        save_intermediate_results: bool = True,
        use_mv_rgb: bool = True,
    ):
        recon_device = self.config["reconstruction"].get("device", "cpu")
        reconstruction_stage2_steps = reconstruction_stage2_steps or self.get_reconstruction_stage2_steps()

        mv_image = self.generate_multiview(input_image)
        mv_path = os.path.join(TMP_DIR, f"{self.uuid}_mv_image.png")
        mv_image.save(mv_path)

        (
            rgb_multi_view,
            vertices,
            faces,
            lrm_multi_view_normals,
            lrm_multi_view_rgb,
            _,
        ) = self.reconstruct_from_multiview(mv_image)
        self.offload_multiview_pipeline()

        rgb_source = rgb_multi_view.squeeze(0) if use_mv_rgb else lrm_multi_view_rgb
        rgb_views = rgb_source
        if rgb_views.shape[0] < 4:
            raise RuntimeError(f"Esperado ao menos 4 vistas do Zero123++, recebido {rgb_views.shape[0]}")
        rgb_views = rgb_views[:4].to(recon_device)

        normal_views = lrm_multi_view_normals[:4]
        normal_views = ((normal_views + 1.0) / 2.0).clamp(0.0, 1.0).to(recon_device)

        from utils.tool import get_background

        multi_view_mask = get_background(normal_views.cpu()).to(recon_device)
        rgb_views = rgb_views * multi_view_mask + (1 - multi_view_mask)

        vertices, faces = self._move_mesh_to_device(vertices, faces, recon_device)

        stage1_steps = max(15, reconstruction_stage2_steps // 4)
        save_paths = [
            os.path.join(OUT_DIR, f"{self.uuid}.glb"),
            os.path.join(OUT_DIR, f"{self.uuid}.obj"),
        ]

        isomer_reconstruct(
            rgb_multi_view=rgb_views,
            normal_multi_view=normal_views,
            multi_view_mask=multi_view_mask,
            vertices=vertices,
            faces=faces,
            save_paths=save_paths,
            radius=4.15,
            azimuths=[0, 90, 180, 270],
            elevations=[5, 5, 5, 5],
            reconstruction_stage1_steps=stage1_steps,
            reconstruction_stage2_steps=reconstruction_stage2_steps,
            geo_weights=[1.0, 0.95, 1.0, 0.95],
            color_weights=[1.0, 0.7, 1.0, 0.7],
        )
        _empty_cuda_cache()

        bundle_preview = None
        if save_intermediate_results:
            bundle_tensor = torchvision.utils.make_grid(
                torch.cat([rgb_views.cpu(), normal_views.cpu()], dim=0),
                nrow=4,
                padding=0,
            )
            bundle_preview = os.path.join(TMP_DIR, f"{self.uuid}_multiview_bundle.png")
            torchvision.utils.save_image(bundle_tensor, bundle_preview)

        return save_paths[0], save_paths[1], bundle_preview or mv_path

    def generate_flux_bundle(
        self,
        input_image: Image.Image,
        caption: str,
        enable_redux: bool = True,
        use_controlnet: bool = True,
    ):
        flux_device = self.config["flux"].get("device", "cpu")
        flux_seed = self._build_flux_seed_bundle(input_image).to(flux_device)
        seed_path = os.path.join(TMP_DIR, f"{self.uuid}_flux_seed_bundle.png")
        torchvision.utils.save_image(flux_seed, seed_path)

        redux_hparam = None
        if enable_redux and self.flux_redux_pipeline is not None:
            redux_hparam = {
                "image": self.to_512_tensor(input_image).unsqueeze(0).clip(0.0, 1.0),
                "prompt_embeds_scale": 1.0,
                "pooled_prompt_embeds_scale": 1.0,
                "strength": 0.5,
            }

        flux_cfg = self.config.get("flux", {})

        def _expand_sequence(value, default, target_len):
            if isinstance(value, (list, tuple)):
                seq = list(value)
            else:
                seq = [value if value is not None else default]
            if not seq:
                seq = [default]
            while len(seq) < target_len:
                seq.append(seq[-1])
            return seq

        if use_controlnet and self.flux_pipeline and hasattr(self.flux_pipeline, "controlnet"):
            control_mode = flux_cfg.get("controlnet_modes", ["tile"])
            control_kernel = flux_cfg.get("controlnet_kernel_size", 51)
            control_sigma = flux_cfg.get("controlnet_sigma", 2.0)
            control_downscale = flux_cfg.get("controlnet_down_scale", 1)
            control_guidance_start = flux_cfg.get("controlnet_guidance_start", 0.0)
            control_guidance_end = flux_cfg.get("controlnet_guidance_end", 0.65)
            controlnet_conditioning_scale = flux_cfg.get("controlnet_conditioning_scale", 0.6)

            control_mode = list(control_mode) if isinstance(control_mode, (list, tuple)) else [control_mode]
            control_guidance_start = _expand_sequence(control_guidance_start, 0.0, len(control_mode))
            control_guidance_end = _expand_sequence(control_guidance_end, 0.65, len(control_mode))
            controlnet_conditioning_scale = _expand_sequence(
                controlnet_conditioning_scale,
                0.6,
                len(control_mode),
            )

            control_images = [
                self.preprocess_controlnet_cond_image(
                    flux_seed,
                    mode_,
                    down_scale=control_downscale,
                    kernel_size=control_kernel,
                    sigma=control_sigma,
                )
                for mode_ in control_mode
            ]
            control_dtype = getattr(self.flux_pipeline.controlnet, "dtype", flux_seed.dtype)
            control_images = [
                TF.resize(img, [self.flux_height, self.flux_width], antialias=True)
                .to(device=flux_device, dtype=control_dtype)
                for img in control_images
            ]

            bundle_tensor, bundle_path = self.generate_3d_bundle_image_controlnet(
                prompt=caption,
                image=flux_seed,
                strength=0.95,
                control_image=control_images,
                control_mode=control_mode,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                lora_scale=1.0,
                redux_hparam=redux_hparam,
            )
        else:
            bundle_tensor, bundle_path = self.generate_3d_bundle_image_text(
                prompt=caption,
                image=flux_seed,
                strength=0.95,
                lora_scale=1.0,
                redux_hparam=redux_hparam,
            )

        self.offload_flux_pipelines()
        return bundle_tensor, bundle_path

    def run_flux_pipeline(
        self,
        input_image: Image.Image,
        caption: str,
        enable_redux: bool = True,
        use_controlnet: bool = True,
        reconstruction_stage2_steps: int | None = None,
    ):
        bundle_tensor, bundle_path = self.generate_flux_bundle(
            input_image=input_image,
            caption=caption,
            enable_redux=enable_redux,
            use_controlnet=use_controlnet,
        )
        mesh_path = self.reconstruct_3d_bundle_image(
            bundle_tensor,
            reconstruction_stage2_steps=reconstruction_stage2_steps or self.get_reconstruction_stage2_steps(),
            save_intermediate_results=True,
        )
        return bundle_path, mesh_path

    def generate_3d_bundle_image_controlnet(
        self,
        prompt,
        image=None,
        strength=1.0,
        control_image=[],
        control_mode=[],
        control_guidance_start=None,
        control_guidance_end=None,
        controlnet_conditioning_scale=None,
        lora_scale=1.0,
        num_inference_steps=None,
        seed=None,
        redux_hparam=None,
        save_intermediate_results=True,
        **kwargs,
    ):
        control_mode_dict = {
            "canny": 0,
            "tile": 1,
            "depth": 2,
            "blur": 3,
            "pose": 4,
            "gray": 5,
            "lq": 6,
        }  # for InstantX Controlnet Union only

        flux_device = self.config["flux"].get("device", "cpu")
        seed = seed or self.config["flux"].get("seed", 0)
        num_inference_steps = num_inference_steps or self.config["flux"].get("num_inference_steps", 20)

        generator = torch.Generator(device=flux_device).manual_seed(seed)

        if image is None:
            placeholder_dtype = torch.float16 if isinstance(flux_device, str) and flux_device.startswith("cuda") else torch.float32
            image = torch.zeros(
                (1, 3, self.flux_height, self.flux_width),
                dtype=placeholder_dtype,
                device=flux_device,
            )

        hparam_dict = {
            "prompt": "A grid of 2x4 multi-view image, elevation 5. White background.",
            "prompt_2": " ".join(["A grid of 2x4 multi-view image, elevation 5. White background.", prompt]),
            "image": image,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": 3.5,
            "num_images_per_prompt": 1,
            "width": self.flux_width,
            "height": self.flux_height,
            "output_type": "np",
            "generator": generator,
            "joint_attention_kwargs": {"scale": lora_scale},
        }
        hparam_dict.update(kwargs)

        if redux_hparam is not None:
            assert self.flux_redux_pipeline is not None
            assert "image" in redux_hparam.keys()

            prompt_primary = hparam_dict.get("prompt")
            prompt_secondary = hparam_dict.get("prompt_2")
            redux_hparam_ = {
                "prompt": prompt_primary,
                "prompt_2": prompt_secondary,
            }
            redux_hparam_.update(redux_hparam)

            try:
                with self.context():
                    self.flux_redux_pipeline(**redux_hparam_)
            except RuntimeError as exc:
                logger.error(
                    "[FLUX] Redux pipeline falhou (%s). Prosseguindo sem Redux.",
                    exc,
                )
                redux_hparam = None
            else:
                hparam_dict["prompt"] = prompt_primary
                hparam_dict["prompt_2"] = prompt_secondary

        control_mode_idx = [control_mode_dict[mode] for mode in control_mode]

        try:
            prompt_embeds_cpu, pooled_prompt_embeds_cpu = self._encode_prompt_cpu(
                prompt=hparam_dict["prompt"],
                prompt_2=hparam_dict["prompt_2"],
                num_images_per_prompt=hparam_dict["num_images_per_prompt"],
            )
            prompt_embeds = prompt_embeds_cpu.to(
                device=flux_device,
                dtype=self.flux_pipeline.text_encoder_2.dtype,
            )
            pooled_prompt_embeds = pooled_prompt_embeds_cpu.to(
                device=flux_device,
                dtype=self.flux_pipeline.text_encoder.dtype,
            )
            hparam_dict["prompt_embeds"] = prompt_embeds
            hparam_dict["pooled_prompt_embeds"] = pooled_prompt_embeds
            hparam_dict.pop("prompt", None)
            hparam_dict.pop("prompt_2", None)
            logger.info(
                "[FLUX] Prompt embeddings pré-calculadas via CPU (shape=%s, pooled=%s)",
                tuple(prompt_embeds.shape),
                tuple(pooled_prompt_embeds.shape),
            )
        except Exception as exc:
            logger.warning(
                "[FLUX] Falha ao pré-calcular embeddings do prompt (%s). Voltando ao caminho padrão.",
                exc,
            )

        extra_kwargs = {
            "control_image": control_image,
            "control_mode": control_mode_idx,
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
        }

        with self.context():
            images = self.flux_pipeline(**hparam_dict, **extra_kwargs)["images"]

        images = torch.from_numpy(images).permute(0, 3, 1, 2)
        images = images.clip(0.0, 1.0)

        if save_intermediate_results:
            save_path = os.path.join(TMP_DIR, f"{self.uuid}_generated_bundle.png")
            torchvision.utils.save_image(images, save_path)
        else:
            save_path = None

        self._assert_no_duplicate_views(
            images,
            stage_name="flux_controlnet_bundle",
            raise_on_fail=False,
        )

        return images, save_path

    def generate_3d_bundle_image_text(
        self,
        prompt,
        image=None,
        strength=1.0,
        lora_scale=1.0,
        num_inference_steps=None,
        redux_hparam=None,
        save_intermediate_results=True,
        **kwargs,
    ):
        flux_device = self.config["flux"].get("device", "cpu")
        seed = kwargs.pop("seed", None) or self.config["flux"].get("seed", 0)
        num_inference_steps = num_inference_steps or self.config["flux"].get("num_inference_steps", 20)

        generator = torch.Generator(device=flux_device).manual_seed(seed)

        if image is None:
            placeholder_dtype = torch.float16 if isinstance(flux_device, str) and flux_device.startswith("cuda") else torch.float32
            image = torch.zeros(
                (1, 3, self.flux_height, self.flux_width),
                dtype=placeholder_dtype,
                device=flux_device,
            )

        hparam_dict = {
            "prompt": "A grid of 2x4 multi-view image, elevation 5. White background.",
            "prompt_2": " ".join(["A grid of 2x4 multi-view image, elevation 5. White background.", prompt]),
            "image": image,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": 3.5,
            "num_images_per_prompt": 1,
            "width": self.flux_width,
            "height": self.flux_height,
            "output_type": "np",
            "generator": generator,
            "joint_attention_kwargs": {"scale": lora_scale},
        }
        hparam_dict.update(kwargs)

        if redux_hparam is not None:
            assert self.flux_redux_pipeline is not None
            assert "image" in redux_hparam.keys()
            redux_hparam_ = {
                "prompt": hparam_dict.pop("prompt"),
                "prompt_2": hparam_dict.pop("prompt_2"),
            }
            redux_hparam_.update(redux_hparam)

            with self.context():
                self.flux_redux_pipeline(**redux_hparam_)

        with self.context():
            images = self.flux_pipeline(**hparam_dict)["images"]

        images = torch.from_numpy(images).permute(0, 3, 1, 2)
        images = images.clip(0.0, 1.0)

        if save_intermediate_results:
            save_path = os.path.join(TMP_DIR, f"{self.uuid}_generated_bundle.png")
            torchvision.utils.save_image(images, save_path)
        else:
            save_path = None

        self._assert_no_duplicate_views(
            images,
            stage_name="flux_text_bundle",
            raise_on_fail=False,
        )

        return images, save_path

    def preprocess_controlnet_cond_image(self, reference_3d_bundle_image, mode, **kwargs):
        """
        Normaliza e gera mapas de condicionamento para o ControlNet Union.
        """

        def _ensure_tensor(img):
            if isinstance(img, torch.Tensor):
                tensor = img.detach().clone()
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                return tensor
            if isinstance(img, Image.Image):
                return torchvision.transforms.ToTensor()(img).unsqueeze(0)
            raise NotImplementedError(f"Unexpected control reference type: {type(img)}")

        def _kernel_tuple(value):
            if isinstance(value, int):
                k = value if value % 2 == 1 else value + 1
                return (k, k)
            if isinstance(value, (list, tuple)):
                norm = []
                for v in value:
                    k = int(v)
                    k = k if k % 2 == 1 else k + 1
                    norm.append(k)
                if len(norm) == 1:
                    norm = norm * 2
                return tuple(norm[:2])
            return (51, 51)

        kernel_size = _kernel_tuple(kwargs.get("kernel_size", 51))
        sigma = kwargs.get("sigma", 2.0)
        down_scale = kwargs.get("down_scale", 1)
        threshold1 = kwargs.get("canny_threshold1", 100)
        threshold2 = kwargs.get("canny_threshold2", 200)

        ref_image = _ensure_tensor(reference_3d_bundle_image).to(torch.float32).clamp(0, 1)

        if down_scale and down_scale > 1:
            h = max(32, ref_image.shape[-2] // down_scale)
            w = max(32, ref_image.shape[-1] // down_scale)
            ref_image = TF.resize(ref_image, size=[h, w], antialias=True)

        if mode == "tile":
            blurred = TF.gaussian_blur(ref_image, kernel_size=kernel_size, sigma=sigma)
            return blurred

        if mode == "gray":
            gray = TF.rgb_to_grayscale(ref_image, num_output_channels=3)
            return gray

        if mode == "canny":
            if cv2 is None:
                raise ImportError("opencv-python nao esta instalado; necessario para control_mode 'canny'.")
            np_img = (ref_image.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            canny_outputs = []
            for sample in np_img:
                edges = cv2.Canny(sample, threshold1=threshold1, threshold2=threshold2)
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                canny_outputs.append(edges)
            canny = torch.from_numpy(np.stack(canny_outputs)).permute(0, 3, 1, 2).float() / 255.0
            return canny

        if mode == "depth":
            gray = TF.rgb_to_grayscale(ref_image, num_output_channels=1)
            blurred = TF.gaussian_blur(gray, kernel_size=kernel_size, sigma=sigma)
            depth = (blurred - blurred.amin(dim=(-2, -1), keepdim=True)) / (
                blurred.amax(dim=(-2, -1), keepdim=True) - blurred.amin(dim=(-2, -1), keepdim=True) + 1e-6
            )
            depth = depth.repeat(1, 3, 1, 1)
            return depth

        raise NotImplementedError(f"Unsupported control mode: {mode}")

    def reconstruct_3d_bundle_image(
        self,
        bundle_image,
        isomer_radius=4.15,
        reconstruction_stage2_steps=50,
        save_intermediate_results=True,
    ):
        """
        bundle_image: torch.Tensor, range [0., 1.], shape (3, H, W) ou (1, 3, H, W)
        O bundle_image é um grid 2x4: 2 linhas (RGB no topo, normals embaixo), 4 colunas (4 vistas)
        """
        from utils.tool import get_background
        
        bundle_image = bundle_image.cpu()
        if bundle_image.dim() == 4:
            bundle_image = bundle_image.squeeze(0)  # (1, 3, H, W) -> (3, H, W)
        
        # O bundle_image tem shape (3, 1024, 2048) ou similar - é um grid 2x4
        # Separar em 8 imagens: 4 RGB (linha superior) + 4 normals (linha inferior)
        # Cada vista tem aproximadamente H/2 x W/4
        H, W = bundle_image.shape[1], bundle_image.shape[2]
        h_per_view = H // 2
        w_per_view = W // 4
        
        # Reorganizar: (3, H, W) -> (8, 3, h_per_view, w_per_view)
        images = rearrange(
            bundle_image, 
            'c (n h) (m w) -> (n m) c h w', 
            n=2,  # 2 linhas (RGB + normals)
            m=4,  # 4 colunas (4 vistas)
            h=h_per_view,
            w=w_per_view
        )
        
        # Separar RGB (primeiras 4) e normals (últimas 4)
        rgb_multi_view = images[:4]  # (4, 3, h, w)
        normal_multi_view = images[4:]  # (4, 3, h, w)
        
        # Gerar máscara de background
        recon_device = self.config["reconstruction"].get("device", "cpu")
        multi_view_mask = get_background(normal_multi_view).to(recon_device)
        rgb_multi_view = rgb_multi_view.to(recon_device) * multi_view_mask + (1 - multi_view_mask)

        vertices, faces = [], []
        _empty_cuda_cache()
        
        # LRM reconstruction
        # Habilitar export_texmap para ter texturas intermediárias
        # IMPORTANTE: Usar a mesma ordem de azimutes que o ISOMER espera: [0, 90, 180, 270]
        # Esta ordem corresponde às vistas do Zero123++: [front, right, back, left]
        with self.context():
            vertices, faces, lrm_multi_view_normals, lrm_multi_view_rgb, lrm_multi_view_albedo = lrm_reconstruct(
                self.recon_model,
                self.recon_model_config.infer_config,
                rgb_multi_view.unsqueeze(0).to(recon_device),
                name=self.uuid,
                export_texmap=True,  # HABILITADO: exportar com texturas
                input_camera_type="kiss3d",
                render_azimuths=[0, 90, 180, 270],  # CORRIGIDO: ordem original do projeto
                render_elevations=[5, 5, 5, 5],
                render_radius=isomer_radius,
            )
        
        if save_intermediate_results:
            recon_3D_bundle_image = torchvision.utils.make_grid(
                torch.cat([lrm_multi_view_rgb.cpu(), (lrm_multi_view_normals.cpu() + 1) / 2], dim=0), 
                nrow=4, 
                padding=0
            ).unsqueeze(0)
            torchvision.utils.save_image(
                recon_3D_bundle_image, 
                os.path.join(TMP_DIR, f"{self.uuid}_lrm_recon_3d_bundle_image.png")
            )
        
        lrm_mesh_path = os.path.join(TMP_DIR, f"{self.uuid}_recon_from_kiss3d.obj")
        final_glb_path = os.path.join(OUT_DIR, f"{self.uuid}.glb")
        final_obj_path = os.path.join(OUT_DIR, f"{self.uuid}.obj")
        isomer_paths = [
            os.path.join(OUT_DIR, f"{self.uuid}_isomer.glb"),
            os.path.join(OUT_DIR, f"{self.uuid}_isomer.obj"),
        ]

        # ISOMER reconstruction
        # Parâmetros otimizados para melhor qualidade
        stage1_steps = max(15, reconstruction_stage2_steps // 4)  # 25% dos steps do stage2, mínimo 15
        # Aumentar stage2_steps se muito baixo para melhor qualidade
        if reconstruction_stage2_steps < 40:
            reconstruction_stage2_steps = max(40, reconstruction_stage2_steps)
            logger.info(f"Aumentando stage2_steps para {reconstruction_stage2_steps} para melhor qualidade")
        
        # Garantir que vertices e faces estão no device correto (CUDA) para ISOMER
        # lrm_reconstruct retorna em CPU, mas ISOMER precisa em CUDA
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.to(recon_device)
        elif isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).to(recon_device)
        if isinstance(faces, torch.Tensor):
            faces = faces.to(recon_device)
        elif isinstance(faces, np.ndarray):
            faces = torch.from_numpy(faces).to(recon_device)
        
        # IMPORTANTE: Passar explicitamente os azimutes para garantir consistência com LRM
        # A ordem [0, 90, 180, 270] corresponde a [front, right, back, left]
        isomer_reconstruct(
            rgb_multi_view=rgb_multi_view,
            normal_multi_view=normal_multi_view,
            multi_view_mask=multi_view_mask,
            vertices=vertices,
            faces=faces,
            save_paths=isomer_paths,
            radius=isomer_radius,
            azimuths=[0, 90, 180, 270],  # CORRIGIDO: ordem explícita para consistência
            elevations=[5, 5, 5, 5],
            reconstruction_stage1_steps=stage1_steps,
            reconstruction_stage2_steps=reconstruction_stage2_steps,
            # Parâmetros de qualidade melhorados
            geo_weights=[1.0, 0.95, 1.0, 0.95],  # Pesos mais balanceados
            color_weights=[1.0, 0.7, 1.0, 0.7],  # Melhor projeção de cores
        )
        _empty_cuda_cache()

        self._finalize_lrm_outputs(
            lrm_obj_path=lrm_mesh_path,
            final_obj_path=final_obj_path,
            final_glb_path=final_glb_path,
            fallback_obj=isomer_paths[1] if len(isomer_paths) > 1 else None,
            fallback_glb=isomer_paths[0],
        )

        if save_intermediate_results:
            if lrm_mesh_path and os.path.exists(lrm_mesh_path):
                mesh_for_render = lrm_mesh_path
            elif len(isomer_paths) > 1 and os.path.exists(isomer_paths[1]):
                mesh_for_render = isomer_paths[1]
            else:
                mesh_for_render = isomer_paths[0]
            bundle_image_rendered = render_3d_bundle_image_from_mesh(mesh_for_render)
            render_save_path = os.path.join(TMP_DIR, f"{self.uuid}_bundle_render.png")
            torchvision.utils.save_image(bundle_image_rendered, render_save_path)

        return final_glb_path  # Retorna o .glb principal baseado no LRM

def _initialize_flux_branch(config_, dtype_, fast_mode, load_controlnet, load_redux):
    logger.info("==> Loading Flux model ...")
    log_memory_usage(logger, "Início do carregamento de modelos")
    flux_device = config_["flux"].get("device", "cpu")
    flux_base_model_pth = config_["flux"].get("base_model", None)
    flux_fallback_model = config_["flux"].get("fallback_base_model")
    flux_dtype = (
        config_["flux"].get("dtype")
        or config_["flux"].get("flux_dtype")
        or "bf16"
    )
    flux_controlnet_pth = config_["flux"].get("controlnet", None) if load_controlnet else None
    flux_lora_pth = config_["flux"].get("lora", None)
    flux_redux_pth = config_["flux"].get("redux", None) if load_redux else None
    flux_cpu_offload = config_["flux"].get("cpu_offload", False)

    logger.info(f"[MODEL] Configuração Flux: device={flux_device}, dtype={flux_dtype}, cpu_offload={flux_cpu_offload}")
    logger.info(f"[MODEL] ControlNet: {'carregar' if load_controlnet else 'não carregar'}")
    logger.info(f"[MODEL] Redux: {'carregar' if load_redux else 'não carregar'}")

    def _load_flux_pipeline(model_id: str):
        if model_id.endswith("safetensors") or model_id.startswith("http"):
            if model_id.startswith("http"):
                logger.info(f"[MODEL] Baixando modelo FLUX fp8 de: {model_id}")
                try:
                    if "huggingface.co" in model_id:
                        parts = model_id.split("/")
                        repo_id = f"{parts[3]}/{parts[4]}"
                        filename = parts[-1]
                        logger.info(f"[MODEL] Repo: {repo_id}, Arquivo: {filename}")
                        local_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            repo_type="model"
                        )
                        logger.info(f"[MODEL] Modelo baixado para: {local_path}")
                        model_id = local_path
                    else:
                        raise ValueError(f"URL não suportada: {model_id}")
                except Exception as e:
                    logger.error(f"[MODEL] Erro ao baixar modelo: {e}")
                    raise

            if not os.path.isabs(model_id) and not model_id.startswith("http"):
                original_path = model_id
                candidate_paths = [
                    model_id,
                    os.path.abspath(model_id),
                    KISS3D_ROOT / model_id.lstrip("./"),
                    PROJECT_ROOT / model_id.lstrip("./"),
                    PROJECT_ROOT / model_id.replace("./", ""),
                ]

                found = False
                for candidate in candidate_paths:
                    candidate_str = str(candidate) if hasattr(candidate, "__str__") else candidate
                    if os.path.exists(candidate_str):
                        model_id = os.path.abspath(candidate_str)
                        logger.info(f"[MODEL] Modelo encontrado em: {model_id}")
                        found = True
                        break

                if not found:
                    logger.error(f"[MODEL] Arquivo do modelo não encontrado: {original_path}")
                    logger.error("[MODEL] Caminhos tentados:")
                    for cp in candidate_paths:
                        logger.error(f"  - {cp}")
                    raise FileNotFoundError(f"Arquivo do modelo não encontrado: {original_path}")

            if not os.path.exists(model_id):
                raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_id}")

            if flux_dtype == "fp8":
                load_dtype = torch.bfloat16
                logger.info("[MODEL] Modelo fp8 quantizado será carregado como bfloat16 (fp8 não suportado nativamente)")
            else:
                load_dtype = dtype_[flux_dtype]
            logger.info(f"[MODEL] Carregando modelo FLUX de arquivo safetensors com dtype: {load_dtype}")
            return FluxImg2ImgPipeline.from_single_file(
                model_id,
                torch_dtype=load_dtype,
            )

        if flux_dtype == "fp8":
            load_dtype = torch.bfloat16
            logger.warning("[MODEL] fp8 não suportado. Usando bfloat16 como fallback para modelo do HuggingFace.")
        else:
            load_dtype = dtype_[flux_dtype]

        return FluxImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=load_dtype,
        )

    try:
        logger.info(f"[MODEL] Carregando Flux base model: {flux_base_model_pth}")
        log_memory_usage(logger, "Antes de carregar Flux base")
        if flux_base_model_pth and flux_base_model_pth.startswith("http"):
            logger.info("[MODEL] Detectada URL do modelo, tentando baixar...")
            flux_pipe = _load_flux_pipeline(flux_base_model_pth)
        elif flux_base_model_pth and flux_base_model_pth.endswith("safetensors"):
            try:
                flux_pipe = _load_flux_pipeline(flux_base_model_pth)
            except FileNotFoundError as e:
                logger.warning(
                    "Arquivo do modelo Flux fp8 não encontrado: %s. Tentando fallback.",
                    flux_base_model_pth,
                )
                if flux_fallback_model:
                    logger.info(f"[MODEL] Usando fallback: {flux_fallback_model}")
                    flux_pipe = _load_flux_pipeline(flux_fallback_model)
                else:
                    raise FileNotFoundError(
                        f"Modelo Flux fp8 não encontrado e nenhum fallback configurado: {flux_base_model_pth}"
                    ) from e
        else:
            flux_pipe = _load_flux_pipeline(flux_base_model_pth)
        log_memory_usage(logger, "Após carregar Flux base")
        logger.info("[MODEL] Flux base model carregado com sucesso")
    except (EnvironmentError, FileNotFoundError, OSError) as exc:
        if flux_fallback_model and flux_fallback_model != flux_base_model_pth:
            logger.warning(
                "Falha ao carregar %s (%s). Tentando fallback %s.",
                flux_base_model_pth,
                exc,
                flux_fallback_model,
            )
            flux_pipe = _load_flux_pipeline(flux_fallback_model)
        else:
            raise

    if flux_controlnet_pth is not None:
        logger.info(f"[MODEL] Carregando ControlNet: {flux_controlnet_pth}")
        log_memory_usage(logger, "Antes de carregar ControlNet")
        controlnet_dtype = dtype_[flux_dtype] if flux_dtype != "fp8" else torch.bfloat16
        flux_controlnet = FluxControlNetModel.from_pretrained(
            flux_controlnet_pth,
            torch_dtype=controlnet_dtype,
        )
        flux_pipe = convert_flux_pipeline(
            flux_pipe,
            FluxControlNetImg2ImgPipeline,
            controlnet=[flux_controlnet],
        )
        log_memory_usage(logger, "Após carregar ControlNet")
        logger.info("[MODEL] ControlNet carregado e integrado com sucesso")
    else:
        logger.info("[MODEL] ControlNet não será carregado (load_controlnet=False ou não configurado)")

    flux_pipe.scheduler = FlowMatchHeunDiscreteScheduler.from_config(flux_pipe.scheduler.config)

    logger.info(f"[MODEL] Carregando LoRA: {flux_lora_pth}")
    if not os.path.exists(flux_lora_pth):
        logger.info("[MODEL] LoRA não encontrado localmente, baixando do HuggingFace...")
        flux_lora_pth = hf_hub_download(repo_id="LTT/Kiss3DGen", filename="rgb_normal.safetensors", repo_type="model")
        logger.info(f"[MODEL] LoRA baixado para: {flux_lora_pth}")
    flux_pipe.load_lora_weights(flux_lora_pth)
    logger.info("[MODEL] LoRA carregado com sucesso")

    def _apply_text_encoder_overrides(pipe, device_str, dtype_str):
        if pipe is None or (not device_str and not dtype_str):
            return

        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }

        target_device = None
        if device_str:
            try:
                target_device = torch.device(device_str)
            except (TypeError, ValueError):
                logger.warning("[MODEL] Dispositivo inválido para text_encoder: %s", device_str)

        target_dtype = None
        if dtype_str:
            target_dtype = dtype_map.get(str(dtype_str).lower())
            if target_dtype is None:
                logger.warning("[MODEL] dtype inválido para text_encoder: %s", dtype_str)

        modules = []
        if getattr(pipe, "text_encoder", None) is not None:
            modules.append(("text_encoder", pipe.text_encoder))
        if getattr(pipe, "text_encoder_2", None) is not None:
            modules.append(("text_encoder_2", pipe.text_encoder_2))

        for name, module in modules:
            kwargs = {}
            if target_device is not None:
                kwargs["device"] = target_device
            if target_dtype is not None:
                kwargs["dtype"] = target_dtype
            if not kwargs:
                continue
            try:
                module.to(**kwargs)
                param_sample = None
                try:
                    param_sample = next(module.parameters())
                except StopIteration:
                    param_sample = None
                resolved_device = target_device or (param_sample.device if param_sample is not None else "unknown")
                resolved_dtype = target_dtype or (param_sample.dtype if param_sample is not None else "unknown")
                logger.info(
                    "[MODEL] %s movido para device=%s dtype=%s",
                    name,
                    resolved_device,
                    resolved_dtype,
                )
            except RuntimeError as exc:
                logger.warning("[MODEL] Falha ao mover %s para %s/%s: %s", name, target_device, target_dtype, exc)

    _apply_text_encoder_overrides(
        flux_pipe,
        config_["flux"].get("text_encoder_device"),
        config_["flux"].get("text_encoder_dtype"),
    )

    def _apply_memory_optimizations(pipe):
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing("max")
        if hasattr(pipe, "enable_vae_slicing"):
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass

    def _place_pipeline_on_device(pipe, name="pipeline"):
        nonlocal flux_cpu_offload

        def _offload(target_pipe):
            logger.info("Enabling sequential CPU offload for %s.", name)
            target_pipe.enable_sequential_cpu_offload()
            _apply_memory_optimizations(target_pipe)

        if flux_cpu_offload:
            _offload(pipe)
            return True
        try:
            pipe.to(device=flux_device)
            _apply_memory_optimizations(pipe)
            return False
        except torch.OutOfMemoryError as exc:
            logger.warning(
                "OOM while moving %s to %s: %s. Falling back to CPU offload.",
                name,
                flux_device,
                exc,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _offload(pipe)
            flux_cpu_offload = True
            return True

    flux_cpu_offload = _place_pipeline_on_device(flux_pipe, "Flux pipeline") or flux_cpu_offload

    flux_redux_pipe = None
    if flux_redux_pth is not None:
        logger.info(f"[MODEL] Carregando Flux Redux: {flux_redux_pth}")
        log_memory_usage(logger, "Antes de carregar Flux Redux")
        flux_redux_pipe = FluxPriorReduxPipeline.from_pretrained(
            flux_redux_pth,
            torch_dtype=torch.bfloat16,
        )
        flux_redux_pipe.text_encoder = flux_pipe.text_encoder
        flux_redux_pipe.text_encoder_2 = flux_pipe.text_encoder_2
        flux_redux_pipe.tokenizer = flux_pipe.tokenizer
        flux_redux_pipe.tokenizer_2 = flux_pipe.tokenizer_2

        _place_pipeline_on_device(flux_redux_pipe, "Flux Redux pipeline")
        log_memory_usage(logger, "Após carregar Flux Redux")
        logger.info("[MODEL] Flux Redux carregado com sucesso")
    else:
        logger.info("[MODEL] Flux Redux não será carregado (load_redux=False ou não configurado)")

    _log_cuda_allocation(flux_device, "load flux model")
    return flux_pipe, flux_redux_pipe


def _initialize_multiview_branch(config_):
    logger.info("==> Loading multiview diffusion model ...")
    log_memory_usage(logger, "Antes de carregar multiview model")
    multiview_device = config_["multiview"].get("device", "cpu")
    mv_dtype = (
        torch.float16
        if isinstance(multiview_device, str) and multiview_device.startswith("cuda")
        else torch.float32
    )
    logger.info(f"[MODEL] Carregando Zero123++ multiview: device={multiview_device}, dtype={mv_dtype}")

    custom_pipeline_path = config_["multiview"].get("custom_pipeline", None)
    base_model = config_["multiview"]["base_model"]
    local_model_path = config_["multiview"].get("local_model", None)

    if local_model_path and os.path.exists(local_model_path):
        if os.path.exists(os.path.join(local_model_path, "model_index.json")):
            logger.info(f"[MODEL] Modelo local encontrado: {local_model_path}")
        else:
            logger.warning(f"[MODEL] Diretório local existe mas não tem model_index.json: {local_model_path}")
            local_model_path = None
    elif custom_pipeline_path and os.path.exists(custom_pipeline_path):
        if os.path.exists(os.path.join(custom_pipeline_path, "model_index.json")):
            local_model_path = custom_pipeline_path
            logger.info(f"[MODEL] Modelo local encontrado no custom_pipeline: {local_model_path}")
        else:
            local_model_path = None

    multiview_pipeline = None
    load_errors = []

    if local_model_path:
        try:
            logger.info(f"[MODEL] Tentando carregar modelo local: {local_model_path}")
            logger.info("[MODEL] Zero123++ usa arquivos .bin, tentando sem safetensors primeiro")
            multiview_pipeline = DiffusionPipeline.from_pretrained(
                local_model_path,
                custom_pipeline=custom_pipeline_path,
                torch_dtype=mv_dtype,
                use_safetensors=False,
            )
            logger.info("[MODEL] Zero123++ carregado do modelo local (arquivos .bin)")
        except Exception as e:
            load_errors.append(f"Local (bin): {e}")
            logger.warning(f"[MODEL] Falha ao carregar modelo local sem safetensors: {e}")
            try:
                multiview_pipeline = DiffusionPipeline.from_pretrained(
                    local_model_path,
                    custom_pipeline=custom_pipeline_path,
                    torch_dtype=mv_dtype,
                    use_safetensors=True,
                )
                logger.info("[MODEL] Zero123++ carregado do modelo local com safetensors (fallback)")
            except Exception as e2:
                load_errors.append(f"Local (safetensors): {e2}")
                logger.warning(f"[MODEL] Falha ao carregar modelo local com safetensors: {e2}")

    if multiview_pipeline is None:
        try:
            multiview_pipeline = DiffusionPipeline.from_pretrained(
                base_model,
                custom_pipeline=custom_pipeline_path,
                torch_dtype=mv_dtype,
                use_safetensors=True,
            )
            logger.info("[MODEL] Zero123++ base carregado do HuggingFace com safetensors")
        except Exception as e:
            load_errors.append(f"HuggingFace safetensors: {e}")
            logger.warning("[MODEL] Falha ao carregar Zero123++ com safetensors: %s", e)
            try:
                multiview_pipeline = DiffusionPipeline.from_pretrained(
                    base_model,
                    custom_pipeline=custom_pipeline_path,
                    torch_dtype=mv_dtype,
                    use_safetensors=False,
                )
                logger.info("[MODEL] Zero123++ base carregado do HuggingFace sem safetensors (fallback)")
            except Exception as e2:
                load_errors.append(f"HuggingFace fallback: {e2}")
                logger.error(f"[MODEL] Falha ao carregar Zero123++: {e2}")
                raise RuntimeError(f"Não foi possível carregar o modelo Zero123++. Erros: {load_errors}")

    multiview_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        multiview_pipeline.scheduler.config,
        timestep_spacing="trailing",
    )

    logger.info("[MODEL] Carregando UNet customizado (FlexGen) para Zero123++")
    unet_ckpt_path = config_["multiview"].get("unet", None)
    if not os.path.exists(unet_ckpt_path):
        logger.info("[MODEL] UNet não encontrado localmente, baixando do HuggingFace...")
        unet_ckpt_path = hf_hub_download(repo_id="LTT/Kiss3DGen", filename="flexgen.ckpt", repo_type="model")
        logger.info(f"[MODEL] UNet baixado para: {unet_ckpt_path}")
    state_dict = torch.load(unet_ckpt_path, map_location="cpu", weights_only=True)
    multiview_pipeline.unet.load_state_dict(state_dict, strict=True)
    logger.info("[MODEL] UNet customizado carregado com sucesso")

    multiview_pipeline.to(multiview_device)
    log_memory_usage(logger, "Após carregar multiview model")
    _log_cuda_allocation(multiview_device, "load multiview model")
    logger.info("[MODEL] Multiview model carregado e movido para dispositivo")
    return multiview_pipeline


def init_wrapper_from_config(
    config_path,
    fast_mode=False,
    disable_llm=False,
    load_controlnet=True,
    load_redux=True,
    pipeline_mode: str = "flux",
):
    with open(config_path, "r") as config_file:
        config_ = yaml.load(config_file, yaml.FullLoader)

    if torch.cuda.is_available():
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if not fast_mode and total_mem_gb <= 12.5:
            fast_mode = True
            logger.warning(
                "VRAM <= %.1fGB detectada. Fast mode habilitado automaticamente para evitar OOM.",
                total_mem_gb,
            )

    if fast_mode:
        flux_fast = config_["flux"].get("num_inference_steps_fast")
        if flux_fast:
            config_["flux"]["num_inference_steps"] = min(
                config_["flux"].get("num_inference_steps", flux_fast),
                flux_fast,
            )
        mv_fast = config_["multiview"].get("num_inference_steps_fast")
        if mv_fast:
            config_["multiview"]["num_inference_steps"] = min(
                config_["multiview"].get("num_inference_steps", mv_fast),
                mv_fast,
            )
        recon_fast = config_["reconstruction"].get("stage2_steps_fast")
        if recon_fast:
            config_["reconstruction"]["stage2_steps"] = max(10, recon_fast)
        config_["caption"]["device"] = "cpu"
        # Não forçar disable_llm no fast_mode - deixar o usuário decidir
        # Se o usuário quiser desabilitar, pode passar --disable-llm explicitamente

    # Verificar se fp8 está disponível no PyTorch
    try:
        _ = torch.float8_e4m3fn
        fp8_available = True
    except (AttributeError, TypeError):
        fp8_available = False
        logger.warning("[MODEL] fp8 não está disponível no PyTorch. Usando bfloat16 como fallback.")
    
    dtype_ = {
        "fp8": torch.bfloat16 if not fp8_available else torch.float8_e4m3fn,  # Fallback para bfloat16 se fp8 não disponível
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    # SEMPRE carregar ambos os modelos (como no commit que funcionava)
    # O offload adequado será feito depois
    flux_pipe, flux_redux_pipe = _initialize_flux_branch(
        config_,
        dtype_,
        fast_mode,
        load_controlnet,
        load_redux,
    )

    multiview_pipeline = _initialize_multiview_branch(config_)


    logger.info("==> Loading caption model ...")
    log_memory_usage(logger, "Antes de carregar caption model")
    caption_device = config_["caption"].get("device", "cpu")
    caption_dtype = (
        torch.bfloat16
        if isinstance(caption_device, str) and caption_device.startswith("cuda")
        else torch.float32
    )
    logger.info(f"[MODEL] Carregando caption model: device={caption_device}, dtype={caption_dtype}")
    caption_model = AutoModelForCausalLM.from_pretrained(
        config_["caption"]["base_model"],
        torch_dtype=caption_dtype,
        trust_remote_code=True,
    ).to(caption_device)
    caption_processor = AutoProcessor.from_pretrained(config_["caption"]["base_model"], trust_remote_code=True)
    log_memory_usage(logger, "Após carregar caption model")
    _log_cuda_allocation(caption_device, "load caption model")
    logger.info("[MODEL] Caption model carregado com sucesso")

    logger.info("==> Loading reconstruction model (LRM) ...")
    log_memory_usage(logger, "Antes de carregar LRM")
    recon_device = config_["reconstruction"].get("device", "cpu")
    logger.info(f"[MODEL] Carregando LRM: device={recon_device}")
    logger.info("[MODEL] [LRM] Passo 1/7: Carregando configuração do modelo...")
    sys.stdout.flush()
    recon_model_config = OmegaConf.load(config_["reconstruction"]["model_config"])
    logger.info("[MODEL] [LRM] Passo 2/7: Instanciando modelo a partir da configuração...")
    sys.stdout.flush()
    recon_model = instantiate_from_config(recon_model_config.model_config)
    logger.info("[MODEL] [LRM] Passo 3/7: Modelo instanciado, verificando checkpoint...")
    sys.stdout.flush()
    model_ckpt_path = config_["reconstruction"]["base_model"]
    if not os.path.exists(model_ckpt_path):
        logger.info("[MODEL] LRM checkpoint não encontrado localmente, baixando do HuggingFace...")
        sys.stdout.flush()
        model_ckpt_path = hf_hub_download(repo_id="LTT/PRM", filename="final_ckpt.ckpt", repo_type="model")
        logger.info(f"[MODEL] LRM checkpoint baixado para: {model_ckpt_path}")
        sys.stdout.flush()
    logger.info(f"[MODEL] [LRM] Passo 4/7: Carregando pesos do LRM de: {model_ckpt_path}")
    sys.stdout.flush()
    state_dict = torch.load(model_ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
    logger.info("[MODEL] [LRM] Passo 5/7: Processando state_dict...")
    sys.stdout.flush()
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")}
    logger.info("[MODEL] [LRM] Passo 6/7: Carregando state_dict no modelo...")
    sys.stdout.flush()
    recon_model.load_state_dict(state_dict, strict=True)
    logger.info("[MODEL] [LRM] Movendo modelo para dispositivo...")
    sys.stdout.flush()
    recon_model.to(recon_device)
    logger.info("[MODEL] Inicializando geometria FlexiCubes do LRM...")
    sys.stdout.flush()
    recon_model.init_flexicubes_geometry(recon_device, fovy=50.0)
    logger.info("[MODEL] [LRM] Geometria inicializada, configurando eval()...")
    sys.stdout.flush()
    recon_model.eval()
    log_memory_usage(logger, "Após carregar LRM")
    _log_cuda_allocation(recon_device, "load reconstruction model")
    logger.info("[MODEL] LRM carregado e inicializado com sucesso")
    sys.stdout.flush()

    llm_configs = None if disable_llm else config_.get("llm", None)
    if llm_configs is not None:
        logger.info("==> Loading LLM ...")
        log_memory_usage(logger, "Antes de carregar LLM")
        llm_device = llm_configs.get("device", "cpu")
        
        # Verificar se Ollama está disponível primeiro
        from models.llm.llm import check_ollama_available
        if check_ollama_available():
            logger.info("[MODEL] Ollama detectado! Usando Ollama API ao invés de Hugging Face")
            llm, llm_tokenizer = 'ollama', None
            logger.info("[MODEL] LLM usando Ollama API (não carrega modelo na GPU)")
        else:
            logger.info(f"[MODEL] Ollama não disponível. Tentando carregar do Hugging Face: device={llm_device}")
            try:
                llm, llm_tokenizer = load_llm_model(
                    llm_configs["base_model"],
                    device_map=llm_device,
                    use_ollama=False,  # Forçar Hugging Face já que Ollama não está disponível
                )
                # Se retornou 'ollama', não é um modelo PyTorch, então não precisa mover para CUDA
                if llm != 'ollama' and isinstance(llm_device, str) and llm_device.startswith("cuda"):
                    llm.to(llm_device)
                log_memory_usage(logger, "Após carregar LLM")
                if llm != 'ollama':
                    _log_cuda_allocation(llm_device, "load llm model")
                logger.info("[MODEL] LLM carregado com sucesso do Hugging Face")
            except Exception as e:
                logger.error(f"[MODEL] Erro ao carregar LLM do Hugging Face: {e}")
                logger.error("[MODEL] LLM não será usado.")
                llm, llm_tokenizer = None, None
    else:
        logger.info("[MODEL] LLM não será carregado (disable_llm=True ou não configurado)")
        llm, llm_tokenizer = None, None
    
    log_memory_usage(logger, "Final do carregamento de todos os modelos")
    logger.info("="*80)
    logger.info("TODOS OS MODELOS CARREGADOS COM SUCESSO")
    logger.info("="*80)

    return kiss3d_wrapper(
        config=config_,
        flux_pipeline=flux_pipe,
        flux_redux_pipeline=flux_redux_pipe,
        multiview_pipeline=multiview_pipeline,
        caption_processor=caption_processor,
        caption_model=caption_model,
        reconstruction_model_config=recon_model_config,
        reconstruction_model=recon_model,
        llm_model=llm,
        llm_tokenizer=llm_tokenizer,
        fast_mode=fast_mode,
    )


def run_image_to_3d(
    k3d_wrapper,
    input_image_path,
    enable_redux=True,
    use_mv_rgb=True,
    use_controlnet=True,
    pipeline_mode: str = "flux",
):
    k3d_wrapper.renew_uuid()

    input_image = preprocess_input_image(Image.open(input_image_path))
    input_image.save(os.path.join(TMP_DIR, f"{k3d_wrapper.uuid}_input_image.png"))

    caption = k3d_wrapper.get_image_caption(input_image)
    k3d_wrapper.release_text_models()

    if pipeline_mode == "multiview":
        mesh_glb, mesh_obj, _ = k3d_wrapper.run_multiview_pipeline(
            input_image,
            reconstruction_stage2_steps=k3d_wrapper.get_reconstruction_stage2_steps(),
            save_intermediate_results=True,
            use_mv_rgb=use_mv_rgb,
        )
        return None, mesh_glb or mesh_obj

    bundle_tensor, bundle_path = k3d_wrapper.generate_flux_bundle(
        input_image=input_image,
        caption=caption,
        enable_redux=enable_redux,
        use_controlnet=use_controlnet,
    )
    recon_mesh_path = k3d_wrapper.reconstruct_3d_bundle_image(
        bundle_tensor,
        save_intermediate_results=True,
        isomer_radius=4.15,
        reconstruction_stage2_steps=k3d_wrapper.get_reconstruction_stage2_steps(),
    )
    return bundle_path, recon_mesh_path


