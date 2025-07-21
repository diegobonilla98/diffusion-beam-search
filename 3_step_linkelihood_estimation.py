"""
Forward Likelihood Estimation for Diffusion Models

This script implements forward likelihood estimation to evaluate the quality of 
diffusion model predictions during sampling. Unlike backward likelihood (which 
measures sampling process entropy), forward likelihood directly assesses how 
plausible the predicted noise is under the forward diffusion process, providing 
a better proxy for prediction confidence and model performance.

The forward NLL is computed as:
NLL = 0.5 * log(2π * variance_t) + 0.5 * (||ε_pred||² / d)

Where lower values indicate better predictions (ε_pred closer to unit variance).
"""

from diffusers import AutoencoderKL, UNet2DConditionModel
from scheduling_ddpm import DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import math
import torch
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt


prompt = "A fantasy landscape with mountains and rivers"
negative_prompt = ""  # Use empty string for default unconditional guidance
guidance_scale = 7.5  # Default in pipeline
num_inference_steps = 50  # Default in pipeline
height = 512  # Default resolution
width = 512
generator = None  # Optional: torch.Generator(device="cuda").manual_seed(seed) for reproducibility

# Device and dtype setup
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use float32 for accurate NLL computation to avoid precision issues in late steps
dtype = torch.float32

# Model ID and cache directory
model_id = "runwayml/stable-diffusion-v1-5"
cache_dir = "D:\\hf"

# Load individual components
# Change to stochastic scheduler for meaningful NLL
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir=cache_dir)
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

# --- helper to pull scheduler coefficients as tensors on the right device ---
alphas_cumprod = scheduler.alphas_cumprod.to(device, dtype)        # shape [N_train]
betas          = scheduler.betas.to(device, dtype)                 # shape [N_train]
alphas         = scheduler.alphas.to(device, dtype)

scores = []  # To store per-step forward NLL (per element)
score_estimates = []  # To store score function estimates
intermediate_x0_samples = []  # To store pred_original_sample every 10 steps
intermediate_step_indices = []  # To store which steps we saved

def compute_forward_nll(noise_pred, variance_t, latent_dims):
    """
    Compute forward-process NLL (per element) based on noise prediction quality.
    This measures how plausible the predicted noise is under the forward diffusion process.
    Returns a float (lower is better).
    """
    epsilon_norm_sq = torch.sum(noise_pred ** 2).item()
    d = latent_dims
    log_term = 0.5 * math.log(2 * math.pi * variance_t)
    quad_term = 0.5 * (epsilon_norm_sq / d)
    nll_per_element = log_term + quad_term
    return nll_per_element

for step_idx, t in enumerate(tqdm(timesteps)):
    # Expand latents for classifier-free guidance
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # Predict the noise with UNet
    with torch.no_grad():
        noise_pred_out = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # Perform classifier-free guidance
    noise_pred_uncond, noise_pred_text = noise_pred_out.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Compute mu_theta (predicted mean)
    t_idx = t.item()
    alpha_prod_t = alphas_cumprod[t_idx].float()
    beta_prod_t = 1 - alpha_prod_t
    sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
    sqrt_one_minus_alpha_prod_t = torch.sqrt(beta_prod_t)

    noise_pred_float = noise_pred.float()
    latents_float = latents.float()

    # Predict x0
    pred_original_sample = (latents_float - sqrt_one_minus_alpha_prod_t * noise_pred_float) / sqrt_alpha_prod_t

    # Clip pred_original_sample
    pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
    
    # Store intermediate x0 predictions every 10 steps
    if step_idx % 10 == 0 or step_idx == len(timesteps) - 1:
        intermediate_x0_samples.append(pred_original_sample.clone())
        intermediate_step_indices.append(step_idx)

    # Compute forward NLL per element (measures prediction quality, lower better)
    variance_t = beta_prod_t.item()  # 1 - alpha_prod_t
    d = latents.numel()
    nll_per_element = compute_forward_nll(noise_pred_float, variance_t, d)
    scores.append(nll_per_element)

    # Compute score function estimate (the trained approximation to the score)
    score_estimate = -noise_pred_float / sqrt_one_minus_alpha_prod_t
    score_estimates.append(score_estimate.clone())

    # Compute the previous noisy sample x_{t-1}
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# After the loop: `scores` is a Python list of length = num_inference_steps
print("Per-step forward NLL (per element, lower better - measures prediction quality):")
for step, score in enumerate(scores):
    print(f"Step {step + 1} (timestep {timesteps[step].item()}): {score:.4f}")
print(f"Average forward NLL over chain: {np.mean(scores):.4f}")

# Show some statistics about score estimates
score_norms = [torch.norm(score).item() for score in score_estimates]
print("\nScore estimate statistics:")
print(f"Average score norm: {np.mean(score_norms):.4f}")
print(f"Score norm std: {np.std(score_norms):.4f}")
print(f"Min score norm: {np.min(score_norms):.4f}")
print(f"Max score norm: {np.max(score_norms):.4f}")


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
pil_image.save("generated_image_lh.png")

# Create a plot showing intermediate x0 predictions
def latent_to_image(latent_tensor, vae, device):
    """Convert a single latent tensor to PIL image"""
    with torch.no_grad():
        latent_scaled = latent_tensor / vae.config.scaling_factor
        # Ensure the latent tensor has the same dtype as the VAE
        latent_scaled = latent_scaled.to(dtype=vae.dtype)
        decoded = vae.decode(latent_scaled).sample[0]
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = decoded.cpu().permute(1, 2, 0).numpy()
        decoded = (decoded * 255).round().astype("uint8")
        return Image.fromarray(decoded)

# Convert intermediate samples to images
print(f"\nConverting {len(intermediate_x0_samples)} intermediate samples to images...")
intermediate_images = []
for i, x0_sample in enumerate(intermediate_x0_samples):
    img = latent_to_image(x0_sample, vae, device)
    intermediate_images.append(img)

# Create subplot grid
n_samples = len(intermediate_images)
cols = 5
rows = 1

fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
if rows == 1:
    axes = [axes] if cols == 1 else axes
else:
    axes = axes.flatten()

for i, (img, step_idx) in enumerate(zip(intermediate_images, intermediate_step_indices)):
    if i < len(axes):
        axes[i].imshow(img)
        axes[i].set_title(f'Step {step_idx + 1}\n(t={timesteps[step_idx].item()})')
        axes[i].axis('off')

# Hide unused subplots
for i in range(len(intermediate_images), len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("intermediate_x0_predictions.png", dpi=150, bbox_inches='tight')
plt.show()
