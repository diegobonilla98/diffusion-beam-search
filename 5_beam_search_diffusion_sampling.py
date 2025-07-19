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

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
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

# Beam search parameters
beam_width = 3
num_candidates_per_beam = 3
lookahead_steps = 2

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
latents = torch.randn(
    (beam_width, unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor),
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

scores = []  # To store per-step average forward NLL (per element)
score_estimates = []  # To store average score function estimate norms

def compute_forward_nll_batch(noise_pred, variance_t, d):
    """
    Compute forward-process NLL (per element) based on noise prediction quality for a batch.
    Returns a tensor of NLL values (lower is better).
    """
    epsilon_norm_sq = torch.sum(noise_pred ** 2, dim=[1,2,3])
    log_term = 0.5 * math.log(2 * math.pi * variance_t)
    quad_term = 0.5 * (epsilon_norm_sq / d)
    nll_per_element = log_term + quad_term
    return nll_per_element

latent_dims = latents[0].numel()  # Dimensions per latent

for step_idx, t in enumerate(tqdm(timesteps)):
    # Expand latents for classifier-free guidance
    batch_size = latents.shape[0]  # beam_width
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # Prepare encoder hidden states
    encoder_hidden_states = torch.cat([text_embeddings[:1].repeat(batch_size, 1, 1), text_embeddings[1:].repeat(batch_size, 1, 1)], dim=0)

    # Predict the noise with UNet
    with torch.no_grad():
        noise_pred_out = unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states).sample

    # Perform classifier-free guidance
    noise_pred_uncond, noise_pred_text = noise_pred_out.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Compute current NLL for statistics
    t_idx = t.item()
    alpha_prod_t = alphas_cumprod[t_idx].float()
    beta_prod_t = 1 - alpha_prod_t
    variance_t = beta_prod_t.item()
    nll_current = compute_forward_nll_batch(noise_pred.float(), variance_t, latent_dims)
    scores.append(nll_current.mean().item())

    # Compute score function estimate
    sqrt_one_minus_alpha_prod_t = torch.sqrt(beta_prod_t)
    score_estimate = -noise_pred.float() / sqrt_one_minus_alpha_prod_t
    score_norms = torch.norm(score_estimate.view(batch_size, -1), dim=1).mean().item()
    score_estimates.append(score_norms)  # Store average norm

    # Predict x0
    sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
    pred_original_sample = (latents.float() - sqrt_one_minus_alpha_prod_t * noise_pred.float()) / sqrt_alpha_prod_t

    # Clip pred_original_sample
    pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)

    # --- Compute the previous noisy sample x_{t-1} candidates ---

    # 1. Get the previous timestep
    prev_t = scheduler.previous_timestep(t)

    # 2. Get alpha_prod for the previous timestep
    if prev_t >= 0:
        alpha_prod_t_prev = alphas_cumprod[prev_t.item()]
    else:
        alpha_prod_t_prev = torch.tensor(1.0, device=device, dtype=dtype)

    # 3. Compute effective alpha_t and beta_t for the current step transition
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # 4. Compute the mean of the posterior
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    pred_original_sample_coeff = (torch.sqrt(alpha_prod_t_prev) * current_beta_t) / beta_prod_t
    current_sample_coeff = (torch.sqrt(current_alpha_t) * beta_prod_t_prev) / beta_prod_t
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

    # 5. Compute the variance and standard deviation
    variance = (beta_prod_t_prev / beta_prod_t) * current_beta_t
    variance = torch.clamp(variance, min=1e-20)
    std_dev_t = torch.sqrt(variance)

    # 6. Generate candidates
    total_candidates = batch_size * num_candidates_per_beam
    if prev_t >= 0:
        # Replicate pred_prev_sample
        pred_prev_sample_rep = pred_prev_sample.repeat_interleave(num_candidates_per_beam, dim=0)
        # Sample noises
        noise = torch.randn((total_candidates, *latents.shape[1:]), device=device, dtype=dtype)
        candidate_latents = pred_prev_sample_rep + std_dev_t * noise
    else:
        # No branching at last step
        latents = pred_prev_sample
        continue

    # Evaluate candidates using lookahead
    lookahead_scores = torch.zeros(total_candidates, device=device, dtype=torch.float32)
    eval_latents = candidate_latents.clone()
    eval_timestep = prev_t
    for m in range(lookahead_steps):
        if eval_timestep < 0:
            break

        # Compute noise_pred for eval_latents
        eval_model_input = torch.cat([eval_latents] * 2)
        eval_model_input = scheduler.scale_model_input(eval_model_input, eval_timestep)
        eval_encoder_hidden = torch.cat([text_embeddings[:1].repeat(total_candidates, 1, 1), text_embeddings[1:].repeat(total_candidates, 1, 1)], dim=0)
        with torch.no_grad():
            eval_noise_pred_out = unet(eval_model_input, eval_timestep, encoder_hidden_states=eval_encoder_hidden).sample
        eval_noise_pred_uncond, eval_noise_pred_text = eval_noise_pred_out.chunk(2)
        eval_noise_pred = eval_noise_pred_uncond + guidance_scale * (eval_noise_pred_text - eval_noise_pred_uncond)

        # Compute NLL
        eval_t_idx = eval_timestep.item()
        eval_alpha_prod_t = alphas_cumprod[eval_t_idx].float()
        eval_variance_t = (1 - eval_alpha_prod_t).item()
        lookahead_nll = compute_forward_nll_batch(eval_noise_pred.float(), eval_variance_t, latent_dims)
        lookahead_scores += lookahead_nll

        # Advance to next eval_latents deterministically if not last
        if m < lookahead_steps - 1:
            eval_sqrt_alpha_prod_t = torch.sqrt(eval_alpha_prod_t)
            eval_sqrt_one_minus = torch.sqrt(1 - eval_alpha_prod_t)
            eval_pred_original_sample = (eval_latents.float() - eval_sqrt_one_minus * eval_noise_pred.float()) / eval_sqrt_alpha_prod_t
            eval_pred_original_sample = torch.clamp(eval_pred_original_sample, -1.0, 1.0)

            eval_prev_t = scheduler.previous_timestep(eval_timestep)
            if eval_prev_t >= 0:
                eval_alpha_prod_t_prev = alphas_cumprod[eval_prev_t.item()]
            else:
                eval_alpha_prod_t_prev = torch.tensor(1.0, device=device, dtype=dtype)

            eval_current_alpha_t = eval_alpha_prod_t / eval_alpha_prod_t_prev
            eval_current_beta_t = 1 - eval_current_alpha_t
            eval_beta_prod_t_prev = 1 - eval_alpha_prod_t_prev

            eval_pred_original_sample_coeff = (torch.sqrt(eval_alpha_prod_t_prev) * eval_current_beta_t) / (1 - eval_alpha_prod_t)
            eval_current_sample_coeff = (torch.sqrt(eval_current_alpha_t) * eval_beta_prod_t_prev) / (1 - eval_alpha_prod_t)
            eval_pred_prev_sample = eval_pred_original_sample_coeff * eval_pred_original_sample + eval_current_sample_coeff * eval_latents

            eval_variance = (eval_beta_prod_t_prev / (1 - eval_alpha_prod_t)) * eval_current_beta_t
            eval_variance = torch.clamp(eval_variance, min=1e-20)
            eval_std_dev_t = torch.sqrt(eval_variance)

            # Deterministic advance
            eval_latents = eval_pred_prev_sample

            eval_timestep = eval_prev_t

    # Select top beam_width candidates with lowest lookahead_scores
    _, top_indices = torch.topk(lookahead_scores, beam_width, largest=False)
    latents = candidate_latents[top_indices]

# After the loop: `scores` is a Python list of length = num_inference_steps
print("Per-step average forward NLL (per element, lower better - measures prediction quality):")
for step, score in enumerate(scores):
    print(f"Step {step + 1} (timestep {timesteps[step].item()}): {score:.4f}")
print(f"Average forward NLL over chain: {np.mean(scores):.4f}")

# Show some statistics about score estimates
print("\nScore estimate statistics:")
print(f"Average score norm: {np.mean(score_estimates):.4f}")
print(f"Score norm std: {np.std(score_estimates):.4f}")
print(f"Min score norm: {np.min(score_estimates):.4f}")
print(f"Max score norm: {np.max(score_estimates):.4f}")

# Decode the first latent to image
final_latents = latents[0].unsqueeze(0)  # Take the first beam
final_latents = 1 / vae.config.scaling_factor * final_latents  # Scale latents before decoding
with torch.no_grad():
    image = vae.decode(final_latents).sample[0]  # Decode to pixel space

# Post-process the image
image = (image / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(1, 2, 0).numpy()
image = (image * 255).round().astype("uint8")
pil_image = Image.fromarray(image)

# Save the image
pil_image.save("generated_image_beam_search.png")
