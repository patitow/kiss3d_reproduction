import os
import sys
import numpy as np
import random
import torch
import yaml
import uuid
from typing import Union, Any, Dict
from einops import rearrange
from PIL import Image
import time

import torchvision
from torchvision.transforms import functional as TF
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from models.llm.llm import load_llm_model, get_llm_response

from diffusers import FluxPipeline, DiffusionPipeline, EulerAncestralDiscreteScheduler, FluxTransformer2DModel
from diffusers.models.controlnets.controlnet_flux import FluxMultiControlNetModel, FluxControlNetModel
from diffusers.schedulers import FlowMatchHeunDiscreteScheduler

try:
    import cv2
except Exception:
    cv2 = None

from huggingface_hub import hf_hub_download

from omegaconf import OmegaConf
from models.lrm.utils.train_util import instantiate_from_config

from kiss3d_utils_local import (
    logger,
    TMP_DIR,
    OUT_DIR,
    preprocess_input_image,
    lrm_reconstruct,
    isomer_reconstruct,
    render_3d_bundle_image_from_mesh,
    KISS3D_ROOT,
)

CUSTOM_PIPELINE_DIR = KISS3D_ROOT / "pipeline" / "custom_pipelines"
if str(CUSTOM_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_PIPELINE_DIR))

from pipeline_flux_prior_redux import FluxPriorReduxPipeline
from pipeline_flux_controlnet_image_to_image import FluxControlNetImg2ImgPipeline
from pipeline_flux_img2img import FluxImg2ImgPipeline


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
    if (
        torch.cuda.is_available()
        and isinstance(device, str)
        and device.startswith("cuda")
    ):
        allocated = torch.cuda.memory_allocated(device=device) / 1024**3
        logger.warning(
            f"GPU memory allocated after {label} on {device}: {allocated} GB"
        )


class kiss3d_wrapper(object):
    def __init__(
        self,
        config: Dict,
        flux_pipeline: Union[FluxPipeline, FluxControlNetImg2ImgPipeline],
        flux_redux_pipeline: FluxPriorReduxPipeline,
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

    def get_reconstruction_stage2_steps(self):
        return self._recon_stage2_fast if self.fast_mode else self._recon_stage2

    def renew_uuid(self):
        self.uuid = uuid.uuid4()

    def context(self):
        if self.config["use_zero_gpu"]:
            pass
        else:
            return torch.no_grad()

    def get_image_caption(self, image):
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
            instruction = (
                "You are an expert 3D asset prompt engineer. Given the base caption below, "
                "produce a single, detailed paragraph that describes the object's geometry, "
                "materials, colors, finishes, logos/text, and any distinctive features for all four canonical views "
                "(front, left, back, right). Avoid backgrounds or scene descriptions; focus strictly on the object. "
                "Mention relative proportions and any manufacturing marks or textures if visible."
            )
            llm_input = f"{instruction}\n\nBase caption:\n{prompt}\n"
            detailed_prompt = get_llm_response(self.llm_model, self.llm_tokenizer, llm_input, seed=seed)

            logger.info(f'LLM refined prompt result: "{detailed_prompt}"')
            return detailed_prompt
        return prompt

    def del_llm_model(self):
        if self.llm_model is not None:
            try:
                self.llm_model.to("cpu")
            except Exception:
                pass
        self.llm_model = None
        self.llm_tokenizer = None
        _empty_cuda_cache()

    def release_text_models(self):
        if self.caption_model is not None:
            try:
                self.caption_model.to("cpu")
            except Exception:
                pass
            self.caption_model = None
        self.del_llm_model()

    def offload_multiview_pipeline(self):
        if self.multiview_pipeline is not None:
            try:
                self.multiview_pipeline.to("cpu")
            except Exception:
                pass
        _empty_cuda_cache()

    def offload_flux_pipelines(self):
        if self.flux_pipeline is not None:
            try:
                self.flux_pipeline.to("cpu")
            except Exception:
                pass
        if self.flux_redux_pipeline is not None:
            try:
                self.flux_redux_pipeline.to("cpu")
            except Exception:
                pass
        _empty_cuda_cache()

    def generate_multiview(self, image, seed=None, num_inference_steps=None):
        seed = seed or self.config["multiview"].get("seed", 0)
        mv_device = self.config["multiview"].get("device", "cpu")
        gen_device = mv_device if isinstance(mv_device, str) and mv_device.startswith("cuda") else "cpu"

        generator = torch.Generator(device=gen_device).manual_seed(seed)
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
            ref_3D_bundle_image = torchvision.utils.make_grid(
                torch.cat(
                    [rgb_multi_view.squeeze(0).detach().cpu()[[3, 0, 1, 2]], (lrm_multi_view_normals.cpu() + 1) / 2],
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

        if save_intermediate_results:
            save_path = os.path.join(TMP_DIR, f"{self.uuid}_ref_3d_bundle_image.png")
            torchvision.utils.save_image(ref_3D_bundle_image, save_path)

            logger.info(f"Save reference 3D bundle image to {save_path}")

            return ref_3D_bundle_image, save_path

        return ref_3D_bundle_image

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
            redux_hparam_ = {
                "prompt": hparam_dict.pop("prompt"),
                "prompt_2": hparam_dict.pop("prompt_2"),
            }
            redux_hparam_.update(redux_hparam)

            with self.context():
                self.flux_redux_pipeline(**redux_hparam_)

        control_mode_idx = [control_mode_dict[mode] for mode in control_mode]

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
        bundle_image = bundle_image.cpu()
        rgb_multi_view = bundle_image[:, [0, 1, 2, 3], ...]
        normal_multi_view = bundle_image[:, [4, 5, 6, 7], ...]
        multi_view_mask = torch.ones_like(rgb_multi_view[:, [0], ...])

        vertices, faces = [], []
        _empty_cuda_cache()
        with self.context():
            vertices, faces = lrm_reconstruct(
                self.recon_model,
                self.recon_model_config.infer_config,
                rgb_multi_view,
                name=self.uuid,
                input_camera_type="kiss3d",
                render_azimuths=[270, 0, 90, 180],
                render_elevations=[5, 5, 5, 5],
                render_radius=isomer_radius,
            )[:2]

        save_paths = os.path.join(OUT_DIR, f"{self.uuid}.glb")
        isomer_reconstruct(
            rgb_multi_view,
            normal_multi_view,
            multi_view_mask,
            vertices,
            faces,
            save_paths=[save_paths],
            radius=isomer_radius,
            reconstruction_stage2_steps=reconstruction_stage2_steps,
        )
        _empty_cuda_cache()

        if save_intermediate_results:
            bundle_image_rendered = render_3d_bundle_image_from_mesh(save_paths)
            render_save_path = os.path.join(TMP_DIR, f"{self.uuid}_bundle_render.png")
            torchvision.utils.save_image(bundle_image_rendered, render_save_path)

        return save_paths


def init_wrapper_from_config(
    config_path,
    fast_mode=False,
    disable_llm=False,
    load_controlnet=True,
    load_redux=True,
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
        disable_llm = True

    dtype_ = {
        "fp8": torch.float8_e4m3fn,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    logger.info("==> Loading Flux model ...")
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

    def _load_flux_pipeline(model_id: str):
        if model_id.endswith("safetensors"):
            return FluxImg2ImgPipeline.from_single_file(
                model_id,
                torch_dtype=dtype_[flux_dtype],
            )
        return FluxImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype_[flux_dtype],
        )

    try:
        flux_pipe = _load_flux_pipeline(flux_base_model_pth)
    except EnvironmentError as exc:
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

    flux_pipe.scheduler = FlowMatchHeunDiscreteScheduler.from_config(flux_pipe.scheduler.config)

    if not os.path.exists(flux_lora_pth):
        flux_lora_pth = hf_hub_download(repo_id="LTT/Kiss3DGen", filename="rgb_normal.safetensors", repo_type="model")
    flux_pipe.load_lora_weights(flux_lora_pth)

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
        flux_redux_pipe = FluxPriorReduxPipeline.from_pretrained(
            flux_redux_pth,
            torch_dtype=torch.bfloat16,
        )
        flux_redux_pipe.text_encoder = flux_pipe.text_encoder
        flux_redux_pipe.text_encoder_2 = flux_pipe.text_encoder_2
        flux_redux_pipe.tokenizer = flux_pipe.tokenizer
        flux_redux_pipe.tokenizer_2 = flux_pipe.tokenizer_2

        _place_pipeline_on_device(flux_redux_pipe, "Flux Redux pipeline")

    _log_cuda_allocation(flux_device, "load flux model")

    logger.info("==> Loading multiview diffusion model ...")
    multiview_device = config_["multiview"].get("device", "cpu")
    mv_dtype = (
        torch.float16
        if isinstance(multiview_device, str) and multiview_device.startswith("cuda")
        else torch.float32
    )
    multiview_pipeline = DiffusionPipeline.from_pretrained(
        config_["multiview"]["base_model"],
        custom_pipeline=config_["multiview"]["custom_pipeline"],
        torch_dtype=mv_dtype,
    )
    multiview_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        multiview_pipeline.scheduler.config, timestep_spacing="trailing"
    )

    unet_ckpt_path = config_["multiview"].get("unet", None)
    if not os.path.exists(unet_ckpt_path):
        unet_ckpt_path = hf_hub_download(repo_id="LTT/Kiss3DGen", filename="flexgen.ckpt", repo_type="model")
    state_dict = torch.load(unet_ckpt_path, map_location="cpu")
    multiview_pipeline.unet.load_state_dict(state_dict, strict=True)

    multiview_pipeline.to(multiview_device)
    _log_cuda_allocation(multiview_device, "load multiview model")

    logger.info("==> Loading caption model ...")
    caption_device = config_["caption"].get("device", "cpu")
    caption_dtype = (
        torch.bfloat16
        if isinstance(caption_device, str) and caption_device.startswith("cuda")
        else torch.float32
    )
    caption_model = AutoModelForCausalLM.from_pretrained(
        config_["caption"]["base_model"],
        torch_dtype=caption_dtype,
        trust_remote_code=True,
    ).to(caption_device)
    caption_processor = AutoProcessor.from_pretrained(config_["caption"]["base_model"], trust_remote_code=True)
    _log_cuda_allocation(caption_device, "load caption model")

    logger.info("==> Loading reconstruction model ...")
    recon_device = config_["reconstruction"].get("device", "cpu")
    recon_model_config = OmegaConf.load(config_["reconstruction"]["model_config"])
    recon_model = instantiate_from_config(recon_model_config.model_config)
    model_ckpt_path = config_["reconstruction"]["base_model"]
    if not os.path.exists(model_ckpt_path):
        model_ckpt_path = hf_hub_download(repo_id="LTT/PRM", filename="final_ckpt.ckpt", repo_type="model")
    state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")}
    recon_model.load_state_dict(state_dict, strict=True)
    recon_model.to(recon_device)
    recon_model.init_flexicubes_geometry(recon_device, fovy=50.0)
    recon_model.eval()
    _log_cuda_allocation(recon_device, "load reconstruction model")

    llm_configs = None if disable_llm else config_.get("llm", None)
    if llm_configs is not None:
        logger.info("==> Loading LLM ...")
        llm_device = llm_configs.get("device", "cpu")
        llm, llm_tokenizer = load_llm_model(
            llm_configs["base_model"],
            device_map=llm_device,
        )
        if isinstance(llm_device, str) and llm_device.startswith("cuda"):
            llm.to(llm_device)
        _log_cuda_allocation(llm_device, "load llm model")
    else:
        llm, llm_tokenizer = None, None

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


def run_image_to_3d(k3d_wrapper, input_image_path, enable_redux=True, use_mv_rgb=True, use_controlnet=True):
    k3d_wrapper.renew_uuid()

    input_image = preprocess_input_image(Image.open(input_image_path))
    input_image.save(os.path.join(TMP_DIR, f"{k3d_wrapper.uuid}_input_image.png"))

    reference_3d_bundle_image, reference_save_path = k3d_wrapper.generate_reference_3D_bundle_image_zero123(
        input_image, use_mv_rgb=use_mv_rgb
    )
    k3d_wrapper.offload_multiview_pipeline()
    caption = k3d_wrapper.get_image_caption(input_image)
    k3d_wrapper.release_text_models()

    if enable_redux:
        redux_hparam = {
            "image": k3d_wrapper.to_512_tensor(input_image).unsqueeze(0).clip(0.0, 1.0),
            "prompt_embeds_scale": 1.0,
            "pooled_prompt_embeds_scale": 1.0,
            "strength": 0.5,
        }
    else:
        redux_hparam = None

    flux_cfg = k3d_wrapper.config.get("flux", {})

    def _expand_sequence(value, default, target_len):
        if isinstance(value, (list, tuple)):
            seq = list(value)
        else:
            seq = [value]
        if not seq:
            seq = [default]
        while len(seq) < target_len:
            seq.append(seq[-1])
        return seq

    flux_ready_image = TF.resize(
        reference_3d_bundle_image,
        [k3d_wrapper.flux_height, k3d_wrapper.flux_width],
        antialias=True,
    ).unsqueeze(0).clamp(0.0, 1.0)
    logger.info(
        "Flux input ajustado para %dx%d",
        k3d_wrapper.flux_height,
        k3d_wrapper.flux_width,
    )
    logger.info("Flux input resized to %s", tuple(flux_ready_image.shape))

    if use_controlnet:
        control_mode = flux_cfg.get("controlnet_modes", ["tile"])
        control_kernel = flux_cfg.get("controlnet_kernel_size", 51)
        control_sigma = flux_cfg.get("controlnet_sigma", 2.0)
        control_downscale = flux_cfg.get("controlnet_down_scale", 1)
        control_guidance_start = flux_cfg.get("controlnet_guidance_start", 0.0)
        control_guidance_end = flux_cfg.get("controlnet_guidance_end", 0.65)
        controlnet_conditioning_scale = flux_cfg.get("controlnet_conditioning_scale", 0.6)

        control_mode = list(control_mode) if isinstance(control_mode, (list, tuple)) else [control_mode]
        if k3d_wrapper.fast_mode and len(control_mode) > 1:
            logger.info(
                "Fast mode ativo: reduzindo controlnet_modes de %s para %s",
                control_mode,
                control_mode[:1],
            )
            control_mode = control_mode[:1]
        control_guidance_start = _expand_sequence(control_guidance_start, 0.0, len(control_mode))
        control_guidance_end = _expand_sequence(control_guidance_end, 0.65, len(control_mode))
        controlnet_conditioning_scale = _expand_sequence(
            controlnet_conditioning_scale, 0.6, len(control_mode)
        )

        control_image = [
            k3d_wrapper.preprocess_controlnet_cond_image(
                reference_3d_bundle_image,
                mode_,
                down_scale=control_downscale,
                kernel_size=control_kernel,
                sigma=control_sigma,
            )
            for mode_ in control_mode
        ]
        control_dtype = getattr(k3d_wrapper.flux_pipeline.controlnet, "dtype", flux_ready_image.dtype)
        control_image = [
            TF.resize(img, [k3d_wrapper.flux_height, k3d_wrapper.flux_width], antialias=True).to(
                dtype=control_dtype
            )
            for img in control_image
        ]

        gen_3d_bundle_image, gen_save_path = k3d_wrapper.generate_3d_bundle_image_controlnet(
            prompt=caption,
            image=flux_ready_image,
            strength=0.95,
            control_image=control_image,
            control_mode=control_mode,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            lora_scale=1.0,
            redux_hparam=redux_hparam,
        )
    else:
        gen_3d_bundle_image, gen_save_path = k3d_wrapper.generate_3d_bundle_image_text(
            prompt=caption,
            image=flux_ready_image,
            strength=0.95,
            lora_scale=1.0,
            redux_hparam=redux_hparam,
        )

    k3d_wrapper.offload_flux_pipelines()

    recon_mesh_path = k3d_wrapper.reconstruct_3d_bundle_image(
        gen_3d_bundle_image,
        save_intermediate_results=False,
        isomer_radius=4.15,
        reconstruction_stage2_steps=k3d_wrapper.get_reconstruction_stage2_steps(),
    )

    return gen_save_path, recon_mesh_path


