from diffusers import StableDiffusionPipeline
import torch


prompt = "A fantasy landscape with mountains and rivers"

# Load Stable Diffusion 1.5 pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    cache_dir="D:\\hf",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Generate image
image = pipe(prompt).images[0]

# Save image
image.save("generated_image.png")
