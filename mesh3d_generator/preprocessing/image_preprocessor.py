"""
Preprocessamento de imagens para geração 3D
Baseado no pipeline do Kiss3DGen (models/lrm/utils/infer_util.py)
"""

import sys
from pathlib import Path
from typing import Optional, Any
from PIL import Image
import numpy as np

# Adicionar path para rembg se disponível
try:
    import rembg
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("[AVISO] rembg nao disponivel. Instale: pip install rembg[new]")


# Sessão global rembg (reutilizar para performance)
_rembg_session = None

def get_rembg_session():
    """Obtém ou cria sessão rembg global"""
    global _rembg_session
    if _rembg_session is None and REMBG_AVAILABLE:
        _rembg_session = rembg.new_session("isnet-general-use")
    return _rembg_session


def to_rgb_image(maybe_rgba: Image.Image) -> tuple[Image.Image, Optional[Image.Image]]:
    """
    Converte imagem para RGB seguindo o padrão do Kiss3DGen.
    
    Args:
        maybe_rgba: Imagem PIL (RGB ou RGBA)
    
    Returns:
        Tupla (imagem RGB, alpha channel ou None)
    """
    assert isinstance(maybe_rgba, Image.Image)
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba, None
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        # Criar fundo cinza (127-128) como no Kiss3DGen
        img = np.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img, rgba.getchannel('A')
    else:
        # Converter para RGB se for outro modo
        return maybe_rgba.convert('RGB'), None


def remove_background(image: Image.Image,
                     rembg_session: Any = None,
                     force: bool = False,
                     bgcolor: tuple = (255, 255, 255, 255),
                     **rembg_kwargs) -> Image.Image:
    """
    Remove background da imagem usando rembg, seguindo o padrão do Kiss3DGen.
    
    Args:
        image: Imagem PIL
        rembg_session: Sessão rembg (opcional, usa global se None)
        force: Forçar remoção mesmo se já tiver alpha
        bgcolor: Cor de fundo (R, G, B, A)
        **rembg_kwargs: Argumentos adicionais para rembg
    
    Returns:
        Imagem RGBA com background removido
    """
    if not REMBG_AVAILABLE:
        print("[AVISO] rembg nao disponivel. Retornando imagem original.")
        return image.convert('RGBA')
    
    # Verificar se já tem alpha e não precisa remover
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    
    if do_remove:
        if rembg_session is None:
            rembg_session = get_rembg_session()
        image = rembg.remove(image, session=rembg_session, bgcolor=bgcolor, **rembg_kwargs)
    
    # Garantir que retorna RGBA
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    return image


def resize_foreground(image: Image.Image,
                     ratio: float = 0.85,
                     pad_value: int = 255) -> Image.Image:
    """
    Redimensiona o foreground e adiciona padding, seguindo o padrão do Kiss3DGen.
    
    IMPORTANTE: Espera imagem RGBA (com canal alpha).
    
    Args:
        image: Imagem PIL RGBA
        ratio: Razão de redimensionamento (0.85 = 85% do tamanho)
        pad_value: Valor de padding (255 = branco)
    
    Returns:
        Imagem RGBA redimensionada e com padding
    """
    image = np.array(image)
    
    # Garantir que tem canal alpha
    if image.shape[-1] != 4:
        # Se não tem alpha, criar um (assumir tudo é foreground)
        if image.shape[-1] == 3:
            alpha = np.ones((image.shape[0], image.shape[1]), dtype=image.dtype) * 255
            image = np.concatenate([image, alpha[..., np.newaxis]], axis=-1)
        else:
            raise ValueError(f"Imagem deve ter 3 ou 4 canais, recebeu {image.shape[-1]}")
    
    # Encontrar bounding box do foreground (onde alpha > 0)
    alpha = np.where(image[..., 3] > 0)
    if len(alpha[0]) == 0:
        # Sem foreground, retornar imagem com padding
        h, w = image.shape[:2]
        new_size = int(max(h, w) / ratio)
        padded = np.full((new_size, new_size, 4), pad_value, dtype=image.dtype)
        padded[..., 3] = 0  # Alpha = 0 (transparente)
        return Image.fromarray(padded)
    
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max() + 1,
        alpha[1].min(),
        alpha[1].max() + 1,
    )
    
    # Crop do foreground
    fg = image[y1:y2, x1:x2]
    
    # Pad para quadrado
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((pad_value, pad_value), (pad_value, pad_value), (pad_value, pad_value)),
    )
    
    # Calcular padding de acordo com o ratio
    new_size = int(new_image.shape[0] / ratio)
    # Pad para new_size, ambos os lados
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((pad_value, pad_value), (pad_value, pad_value), (pad_value, pad_value)),
    )
    
    return Image.fromarray(new_image)


def preprocess_input_image(input_image: Image.Image,
                          target_size: tuple[int, int] = (512, 512),
                          remove_bg: bool = True,
                          resize_ratio: float = 0.85) -> Image.Image:
    """
    Preprocessa imagem de entrada seguindo EXATAMENTE o pipeline do Kiss3DGen.
    
    Pipeline:
    1. Converter para RGB (se necessário)
    2. Remover background usando rembg (retorna RGBA)
    3. Redimensionar foreground e adicionar padding (ratio=0.85, pad branco)
    4. Redimensionar para target_size (512x512)
    5. Converter para RGB final
    
    Args:
        input_image: Imagem PIL de entrada
        target_size: Tamanho alvo (width, height) - padrão (512, 512)
        remove_bg: Se True, remove background
        resize_ratio: Razão de redimensionamento do foreground (0.85 = 85%)
    
    Returns:
        Imagem preprocessada (RGB, target_size, background branco)
    """
    # 1. Converter para RGB
    image, _ = to_rgb_image(input_image)
    
    # 2. Remover background (retorna RGBA)
    if remove_bg:
        image = remove_background(image, bgcolor=(255, 255, 255, 255))
    else:
        # Se não remover background, garantir que tem alpha
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
    
    # 3. Redimensionar foreground e adicionar padding (espera RGBA)
    image = resize_foreground(image, ratio=resize_ratio, pad_value=255)
    
    # 4. Redimensionar para tamanho alvo
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # 5. Converter para RGB final (remover alpha)
    image = to_rgb_image(image)[0]
    
    return image

