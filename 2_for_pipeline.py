from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from PIL import Image
from tqdm.auto import tqdm


prompt = "A fantasy landscape with mountains and rivers"
negative_prompt = ""  # Use empty string for default unconditional guidance
guidance_scale = 7.5  # Default in pipeline
num_inference_steps = 50  # Default in pipeline
height = 512  # Default resolution
width = 512
generator = None  # Optional: torch.Generator(device="cuda").manual_seed(seed) for reproducibility

# Device and dtype setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32  # Use float16 on GPU for efficiency

# Model ID and cache directory
model_id = "runwayml/stable-diffusion-v1-5"
cache_dir = "D:\\hf"

# Load individual components
scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir=cache_dir)
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", cache_dir=cache_dir, torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", cache_dir=cache_dir)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", cache_dir=cache_dir, torch_dtype=dtype)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", cache_dir=cache_dir, torch_dtype=dtype)

# Move models to device
vae.to(device)
text_encoder.to(device)
unet.to(device)

# Compute the VAE scale factor for spatial dimensions
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

# Encode the prompt and negative prompt for classifier-free guidance
text_input = tokenizer(
    [negative_prompt, prompt],
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

# Prepare initial noise latents
batch_size = 1
latents = torch.randn(
    (batch_size, unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor),
    generator=generator,
    device=device,
    dtype=dtype,
)
latents = latents * scheduler.init_noise_sigma

# Set the timesteps for the scheduler
scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = scheduler.timesteps

# The visible diffusion for loop (denoising loop)
for t in tqdm(timesteps):
    # Expand latents for classifier-free guidance (concatenate for uncond and conditioned)
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # Predict the noise with UNet
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # Perform classifier-free guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Compute the previous noisy sample x_{t-1}
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# Decode the latents to image
latents = 1 / vae.config.scaling_factor * latents  # Scale latents before decoding
with torch.no_grad():
    image = vae.decode(latents).sample[0]  # Decode to pixel space

# Post-process the image
image = (image / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(1, 2, 0).numpy()
image = (image * 255).round().astype("uint8")
pil_image = Image.fromarray(image)

# Save the image
pil_image.save("generated_image_for.png")