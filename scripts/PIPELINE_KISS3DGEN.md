# Pipeline Kiss3DGen - Análise e Implementação

## Pipeline Original do Kiss3DGen

### 1. Preprocessamento de Imagem
```python
input_image = preprocess_input_image(Image.open(input_image_path))
# - Remove background usando rembg
# - Resize foreground (ratio=0.85)
# - Pad com branco
# - Converte para RGB 512x512
```

### 2. Geração de 3D Bundle Image (Referência)
```python
reference_3d_bundle_image, reference_save_path = k3d_wrapper.generate_reference_3D_bundle_image_zero123(
    input_image, use_mv_rgb=use_mv_rgb
)
# - Usa Zero123 para gerar múltiplas views (4 views: 270°, 0°, 90°, 180°)
# - Elevação: 5° para todas
# - Gera RGB + Normal maps
# - Retorna bundle image (grid 4x2: 4 RGB + 4 Normal)
```

### 3. Geração de Caption
```python
caption = k3d_wrapper.get_image_caption(input_image)
# - Usa Florence2 ou LLM para gerar descrição
```

### 4. Geração de 3D Bundle Image Refinado
```python
# Com Redux (opcional)
redux_hparam = {
    'image': k3d_wrapper.to_512_tensor(input_image).unsqueeze(0).clip(0., 1.),
    'prompt_embeds_scale': 1.0,
    'pooled_prompt_embeds_scale': 1.0,
    'strength': 0.5
}

# Com ControlNet
gen_3d_bundle_image, gen_save_path = k3d_wrapper.generate_3d_bundle_image_controlnet(
    prompt=caption,
    image=reference_3d_bundle_image.unsqueeze(0),
    strength=0.95,
    control_image=control_image,  # ControlNet-Tile
    control_mode=['tile'],
    control_guidance_start=[0.0],
    control_guidance_end=[0.65],
    controlnet_conditioning_scale=[0.6],
    lora_scale=1.0,
    redux_hparam=redux_hparam
)
```

### 5. Reconstrução 3D (LRM ou ISOMER)
```python
recon_mesh_path = k3d_wrapper.reconstruct_3d_bundle_image(
    gen_3d_bundle_image, 
    save_intermediate_results=False,
    isomer_radius=4.15, 
    reconstruction_stage2_steps=50
)
# - Usa LRM ou ISOMER para reconstruir mesh
# - Input: 3D bundle image (8 imagens: 4 RGB + 4 Normal)
# - Output: Mesh OBJ
```

## O que Precisamos Implementar

### ✅ Já Implementado
1. Carregamento de modelo original (corrigido)
2. Geração de descrição com LLM (Ollama)
3. Visualização comparativa
4. GIF rotativo

### ❌ Falta Implementar

#### 1. Preprocessamento de Imagem
- [ ] Remover background (rembg)
- [ ] Resize foreground
- [ ] Pad com branco

#### 2. Geração de 3D Bundle Image
- [ ] Integrar Zero123 ou similar via ComfyUI
- [ ] Gerar 4 views RGB (270°, 0°, 90°, 180°)
- [ ] Gerar 4 views Normal maps
- [ ] Combinar em bundle image

#### 3. Geração de Normal Maps
- [ ] Implementar via ComfyUI workflow
- [ ] Usar MiDaS ou DPT
- [ ] Converter depth para normal map

#### 4. Refinamento com ControlNet
- [ ] ControlNet-Tile via ComfyUI
- [ ] ControlNet-Normal via ComfyUI
- [ ] Integrar texto detalhado no prompt

#### 5. Reconstrução 3D
- [ ] Integrar LRM via ComfyUI (se disponível)
- [ ] Ou usar InstantMesh
- [ ] Ou implementar reconstrução básica

## Próximos Passos

1. **Imediato**: Corrigir renderização do modelo original (já feito)
2. **Curto prazo**: Implementar preprocessamento de imagem
3. **Médio prazo**: Integrar geração de 3D bundle image via ComfyUI
4. **Longo prazo**: Implementar reconstrução 3D completa

