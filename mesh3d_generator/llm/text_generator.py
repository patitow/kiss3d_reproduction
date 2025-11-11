"""
Gerador de texto detalhado usando LLM local (Ollama) para guiar o refinamento de malhas
"""

from typing import Optional
from pathlib import Path
import base64
import requests
from PIL import Image
import json


class TextGenerator:
    """
    Gera texto extremamente detalhado usando LLM local (Ollama) para descrever cenas 3D.
    
    Este módulo usa modelos de linguagem locais via Ollama para expandir descrições simples
    em textos detalhados que guiam o refinamento de malhas 3D.
    
    Suporta modelos multimodais (llava, bakllava, etc.) para gerar descrições a partir de imagens.
    """
    
    def __init__(self, 
                 model_name: str = "llama3.2",
                 multimodal_model: str = "llava",
                 ollama_url: str = "http://localhost:11434"):
        """
        Inicializa o gerador de texto com Ollama.
        
        Args:
            model_name: Nome do modelo LLM textual (llama3.2, mistral, etc.)
            multimodal_model: Nome do modelo multimodal (llava, bakllava, etc.)
            ollama_url: URL do servidor Ollama (padrão: http://localhost:11434)
        """
        self.model_name = model_name
        self.multimodal_model = multimodal_model
        self.ollama_url = ollama_url
        self._check_ollama_connection()
        
    def _check_ollama_connection(self):
        """Verifica se o Ollama está rodando."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            print(f"✅ Conectado ao Ollama em {self.ollama_url}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Aviso: Não foi possível conectar ao Ollama em {self.ollama_url}")
            print(f"   Certifique-se de que o Ollama está rodando: ollama serve")
            print(f"   Erro: {e}")
    
    def _call_ollama(self, prompt: str, model: Optional[str] = None, images: Optional[list] = None) -> str:
        """
        Chama o modelo Ollama com um prompt.
        
        Args:
            prompt: Texto do prompt
            model: Nome do modelo (usa self.model_name se None)
            images: Lista de imagens em base64 (para modelos multimodais)
            
        Returns:
            Resposta do modelo
        """
        model = model or self.model_name
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        # Adicionar imagens se for modelo multimodal
        if images:
            payload["images"] = images
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Erro ao chamar Ollama: {e}")
    
    def _image_to_base64(self, image_path: str) -> str:
        """
        Converte uma imagem para base64.
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            String base64 da imagem
        """
        if isinstance(image_path, (str, Path)):
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image_path, Image.Image):
            import io
            buffer = io.BytesIO()
            image_path.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            raise ValueError(f"Tipo de imagem não suportado: {type(image_path)}")
    
    def generate_detailed_description(self, initial_text: str, 
                                     include_geometry: bool = True,
                                     include_materials: bool = True,
                                     include_lighting: bool = True) -> str:
        """
        Gera uma descrição extremamente detalhada a partir de um texto inicial.
        
        Args:
            initial_text: Texto inicial simples
            include_geometry: Incluir detalhes geométricos
            include_materials: Incluir detalhes de materiais
            include_lighting: Incluir detalhes de iluminação
            
        Returns:
            Texto detalhado expandido
        """
        details = []
        if include_geometry:
            details.append("geometria e forma dos objetos (dimensões, proporções, curvas, ângulos)")
        if include_materials:
            details.append("materiais e texturas (tipo de superfície, acabamento, reflexos)")
        if include_lighting:
            details.append("iluminação e sombras (direção da luz, intensidade, sombras projetadas)")
        
        details_str = ", ".join(details)
        
        prompt = f"""Você é um especialista em visão computacional e modelagem 3D. 

Baseado na seguinte descrição inicial, gere uma descrição EXTREMAMENTE DETALHADA que será usada para refinar uma malha 3D. 

A descrição deve incluir informações precisas sobre:
- {details_str}
- Perspectiva e profundidade (posição relativa dos objetos, distâncias)
- Detalhes de superfície (rugosidade, brilho, padrões)
- Estrutura e composição dos elementos da cena

Descrição inicial: {initial_text}

Gere uma descrição detalhada e técnica, focada em aspectos que ajudem na reconstrução 3D precisa:"""
        
        return self._call_ollama(prompt)
    
    def generate_from_image(self, image_path: str, 
                           additional_context: Optional[str] = None) -> str:
        """
        Gera uma descrição detalhada a partir de uma imagem usando modelo multimodal.
        
        Args:
            image_path: Caminho para a imagem ou objeto PIL Image
            additional_context: Contexto adicional opcional
            
        Returns:
            Texto detalhado para refinamento 3D
        """
        # Converter imagem para base64
        image_base64 = self._image_to_base64(image_path)
        
        context_text = f"\n\nContexto adicional: {additional_context}" if additional_context else ""
        
        prompt = f"""Você é um especialista em visão computacional e modelagem 3D.

Analise esta imagem e gere uma descrição EXTREMAMENTE DETALHADA que será usada para criar e refinar uma malha 3D.

A descrição deve incluir:
- Geometria e forma dos objetos (dimensões, proporções, curvas, ângulos, estrutura)
- Materiais e texturas (tipo de superfície, acabamento, reflexos, padrões, cores)
- Iluminação e sombras (direção da luz, intensidade, sombras projetadas, highlights)
- Perspectiva e profundidade (posição relativa dos objetos, distâncias, escala)
- Detalhes de superfície (rugosidade, brilho, imperfeições, detalhes finos)
- Estrutura e composição (como os elementos se relacionam, hierarquia visual)

Seja específico e técnico, focando em aspectos que ajudem na reconstrução 3D precisa.{context_text}

Descrição detalhada:"""
        
        return self._call_ollama(prompt, model=self.multimodal_model, images=[image_base64])
    
    def generate_from_image_description(self, image_description: str) -> str:
        """
        Gera texto detalhado a partir de uma descrição de imagem.
        
        Args:
            image_description: Descrição da imagem
            
        Returns:
            Texto detalhado para refinamento 3D
        """
        return self.generate_detailed_description(
            image_description,
            include_geometry=True,
            include_materials=True,
            include_lighting=True
        )
    
    def generate_prompt_for_text_to_image(self, initial_text: str) -> str:
        """
        Gera um prompt otimizado para geração de imagens (text-to-image).
        
        Args:
            initial_text: Texto inicial simples
            
        Returns:
            Prompt detalhado para text-to-image
        """
        prompt = f"""Você é um especialista em geração de imagens com IA.

Baseado na seguinte descrição, gere um prompt detalhado e otimizado para modelos de difusão (Stable Diffusion, SDXL, etc.).

O prompt deve:
- Ser descritivo e específico
- Incluir detalhes visuais importantes
- Usar termos técnicos de arte e fotografia quando apropriado
- Ser estruturado de forma clara

Descrição inicial: {initial_text}

Prompt otimizado para text-to-image:"""
        
        return self._call_ollama(prompt)
    
    def list_available_models(self) -> list:
        """
        Lista os modelos disponíveis no Ollama.
        
        Returns:
            Lista de nomes de modelos disponíveis
        """
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return models
        except requests.exceptions.RequestException as e:
            print(f"Erro ao listar modelos: {e}")
            return []


