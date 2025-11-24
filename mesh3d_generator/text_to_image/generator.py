"""
Gerador de imagens a partir de texto usando modelos de difusão
"""

from typing import Optional
from PIL import Image
import torch


class TextToImageGenerator:
    """
    Gera imagens a partir de texto descritivo usando modelos de difusão.
    
    Esta classe será implementada para usar Stable Diffusion ou SDXL
    para gerar imagens a partir de prompts de texto.
    
    Pode usar LLM (Ollama) para melhorar prompts antes da geração.
    """
    
    def __init__(self, 
                 model_name: str = "runwayml/stable-diffusion-v1-5",
                 use_llm_for_prompts: bool = False,
                 llm_generator: Optional[object] = None):
        """
        Inicializa o gerador de imagens.
        
        Args:
            model_name: Nome do modelo de difusão a ser usado
            use_llm_for_prompts: Se True, usa LLM para melhorar prompts
            llm_generator: Instância de TextGenerator para melhorar prompts (opcional)
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_llm_for_prompts = use_llm_for_prompts
        self.llm_generator = llm_generator
        # TODO: Carregar modelo de difusão
        
    def _enhance_prompt(self, prompt: str) -> str:
        """
        Melhora o prompt usando LLM se configurado.
        
        Args:
            prompt: Prompt original
            
        Returns:
            Prompt melhorado
        """
        if self.use_llm_for_prompts and self.llm_generator:
            try:
                enhanced = self.llm_generator.generate_prompt_for_text_to_image(prompt)
                return enhanced
            except Exception as e:
                print(f"[AVISO]  Erro ao melhorar prompt com LLM: {e}")
                print(f"   Usando prompt original")
                return prompt
        return prompt
        
    def generate(self, prompt: str, num_inference_steps: int = 50, enhance_prompt: bool = None) -> Image.Image:
        """
        Gera uma imagem a partir de um prompt de texto.
        
        Args:
            prompt: Texto descritivo da imagem desejada
            num_inference_steps: Número de passos de inferência
            enhance_prompt: Se True, melhora o prompt com LLM (usa self.use_llm_for_prompts se None)
            
        Returns:
            Imagem gerada (PIL Image)
        """
        # Melhorar prompt se solicitado
        if enhance_prompt is None:
            enhance_prompt = self.use_llm_for_prompts
        
        if enhance_prompt:
            prompt = self._enhance_prompt(prompt)
            print(f" Prompt melhorado: {prompt[:100]}...")
        
        # TODO: Implementar geração de imagem
        raise NotImplementedError("Metodo generate() sera implementado na Etapa 0")
    
    def generate_batch(self, prompts: list[str], **kwargs) -> list[Image.Image]:
        """
        Gera múltiplas imagens a partir de uma lista de prompts.
        
        Args:
            prompts: Lista de textos descritivos
            **kwargs: Argumentos adicionais para generate()
            
        Returns:
            Lista de imagens geradas
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]


