from diffusers import AutoencoderKL, UNet2DConditionModel
from scheduling_ddpm import DDPMScheduler, diverse_randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer
import math
import torch
from PIL import Image
from tqdm.auto import tqdm
import numpy as np


prompt = "A fantasy landscape with mountains and rivers"
negative_prompt = ""  # Use empty string for default unconditional guidance
guidance_scale = 7.5  # Default in pipeline
num_inference_steps = 50  # Default in pipeline
height = 512  # Default resolution
width = 512
generator = None  # Optional: torch.Generator(device="cuda").manual_seed(seed) for reproducibility

# Number of candidates per step for best-of-N selection
N = 10

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

    # Compute forward NLL per element (measures prediction quality, lower better)
    variance_t = beta_prod_t.item()  # 1 - alpha_prod_t
    d = latents.numel()
    nll_per_element = compute_forward_nll(noise_pred_float, variance_t, d)
    scores.append(nll_per_element)

    # Compute score function estimate (the trained approximation to the score)
    score_estimate = -noise_pred_float / sqrt_one_minus_alpha_prod_t
    score_estimates.append(score_estimate.clone())

    # Use the custom scheduler step implementation
    # The scheduler will handle all the computation internally
    
    # Get the previous timestep for NLL computation
    prev_t = scheduler.previous_timestep(t)
    
    # Get beta_prod_t_prev for variance calculation
    if prev_t >= 0:
        alpha_prod_t_prev = alphas_cumprod[prev_t.item()]
        beta_prod_t_prev = 1 - alpha_prod_t_prev
    else:
        beta_prod_t_prev = torch.tensor(0.0, device=device, dtype=dtype)

    # 6. Use custom scheduler step with diverse noise sampling
    if prev_t >= 0:
        # Compute variance_prev for NLL calculations  
        variance_prev = beta_prod_t_prev.item()

        # Generate N diverse noise samples using diverse_randn_tensor
        diverse_noise = diverse_randn_tensor(
            shape=latents.shape[1:],
            N=N,
            generator=generator,
            device=device,
            dtype=dtype
        )

        # Generate N candidates and choose the best based on forward NLL at prev_t
        min_nll = float('inf')
        best_latents = None
        for i in range(N):
            noise = diverse_noise[i:i+1]  # Get i-th noise sample with batch dimension
            
            # Use custom scheduler step with the diverse noise
            step_output = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=latents,
                generator=generator,
                return_dict=True,
                random_noise=noise
            )
            candidate_latents = step_output.prev_sample

            # Compute NLL for this candidate at prev_t
            latent_model_input = torch.cat([candidate_latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, prev_t)
            with torch.no_grad():
                noise_pred_out = unet(latent_model_input, prev_t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred_out.chunk(2)
            noise_pred_i = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            nll_i = compute_forward_nll(noise_pred_i.float(), variance_prev, d)

            print(f"Step {step_idx + 1}, Candidate NLL: {nll_i:.4f}")

            if nll_i < min_nll:
                min_nll = nll_i
                best_latents = candidate_latents
        latents = best_latents
    else:
        # Use custom scheduler step without noise for final step
        step_output = scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=latents,
            generator=generator,
            return_dict=True
        )
        latents = step_output.prev_sample

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
pil_image.save("generated_image_best_of_N.png")
