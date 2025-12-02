import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

print("tokenizer max len", pipe.tokenizer.model_max_length)
print("text encoder vocab", pipe.text_encoder.config.vocab_size)
prompt = (
    "The 3D model should accurately represent a small, round, pink cylindrical object "
    "with a smooth, plush texture."
)

inputs = pipe.tokenizer(
    prompt,
    padding="max_length",
    max_length=pipe.tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
ids = inputs.input_ids
print("ids shape", ids.shape, "max id", ids.max().item(), "min id", ids.min().item())

text_encoder = pipe.text_encoder
with torch.no_grad():
    out = text_encoder(ids, output_hidden_states=False)
print("done", out.pooler_output.shape)

