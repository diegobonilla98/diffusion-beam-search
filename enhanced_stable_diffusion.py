"""
Enhanced Stable Diffusion 1.5 Pipeline with Advanced Sampling Methods

This class provides a unified interface for multiple sampling strategies:
1. Standard sampling (default diffusion process)
2. Best-of-N sampling (generates N candidates, selects best based on forward NLL)
3. Beam search sampling (maintains multiple beams with lookahead evaluation)

All methods use forward likelihood estimation for quality assessment.
"""

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import math
import torch
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
from typing import Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class SamplingConfig:
    """Configuration for sampling parameters"""
    method: str = "standard"  # "standard", "best_of_n", "beam_search"
    
    # Best-of-N parameters
    n_candidates: int = 10
    
    # Beam search parameters
    beam_width: int = 3
    num_candidates_per_beam: int = 3
    lookahead_steps: int = 2
    
    # General parameters
    use_stochastic_scheduler: bool = False  # Use DDPM instead of PNDM for NLL evaluation


class EnhancedStableDiffusion:
    """
    Enhanced Stable Diffusion pipeline with multiple sampling strategies.
    
    Supports standard sampling, Best-of-N sampling, and Beam search sampling
    with forward likelihood estimation for quality assessment.
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        use_float32_for_nll: bool = True
    ):
        """
        Initialize the Enhanced Stable Diffusion pipeline.
        
        Args:
            model_id: HuggingFace model identifier
            cache_dir: Cache directory for downloaded models
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            dtype: Data type for models (None for auto-select)
            use_float32_for_nll: Use float32 for accurate NLL computation
        """
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine dtype based on device and requirements
        if dtype is None:
            if use_float32_for_nll:
                self.dtype = torch.float32
            else:
                self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.dtype = dtype
            
        self.use_float32_for_nll = use_float32_for_nll
        
        # Load models
        self._load_models()
        
        # Compute VAE scale factor
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
        # Statistics storage
        self.scores = []
        self.score_estimates = []
    
    def _load_models(self):
        """Load all required model components"""
        print(f"Loading models on {self.device} with dtype {self.dtype}...")
        
        # Load components with appropriate scheduler
        self.scheduler_pndm = PNDMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler", cache_dir=self.cache_dir
        )
        self.scheduler_ddpm = DDPMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler", cache_dir=self.cache_dir
        )
        
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id, subfolder="vae", cache_dir=self.cache_dir, torch_dtype=self.dtype
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer", cache_dir=self.cache_dir
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder", cache_dir=self.cache_dir, torch_dtype=self.dtype
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet", cache_dir=self.cache_dir, torch_dtype=self.dtype
        )
        
        # Move models to device
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)
        
        print("Models loaded successfully!")
    
    def _encode_prompts(self, prompt: str, negative_prompt: str = "") -> torch.Tensor:
        """Encode prompts for classifier-free guidance"""
        text_input = self.tokenizer(
            [negative_prompt, prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        return text_embeddings
    
    def _compute_forward_nll_batch(self, noise_pred: torch.Tensor, variance_t: float, d: int) -> torch.Tensor:
        """
        Compute forward-process NLL (per element) for a batch of predictions.
        Returns tensor of NLL values (lower is better).
        """
        epsilon_norm_sq = torch.sum(noise_pred ** 2, dim=[1, 2, 3])
        log_term = 0.5 * math.log(2 * math.pi * variance_t)
        quad_term = 0.5 * (epsilon_norm_sq / d)
        nll_per_element = log_term + quad_term
        return nll_per_element
    
    def _compute_forward_nll_single(self, noise_pred: torch.Tensor, variance_t: float, d: int) -> float:
        """Compute forward-process NLL for a single prediction"""
        epsilon_norm_sq = torch.sum(noise_pred ** 2).item()
        log_term = 0.5 * math.log(2 * math.pi * variance_t)
        quad_term = 0.5 * (epsilon_norm_sq / d)
        return log_term + quad_term
    
    def _prepare_scheduler_coefficients(self, scheduler):
        """Prepare scheduler coefficients as tensors on the correct device"""
        alphas_cumprod = scheduler.alphas_cumprod.to(self.device, self.dtype)
        betas = scheduler.betas.to(self.device, self.dtype)
        alphas = scheduler.alphas.to(self.device, self.dtype)
        return alphas_cumprod, betas, alphas
    
    def _standard_sampling(
        self,
        text_embeddings: torch.Tensor,
        latents: torch.Tensor,
        scheduler,
        guidance_scale: float,
        num_inference_steps: int
    ) -> torch.Tensor:
        """Standard diffusion sampling process"""
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps
        
        for t in tqdm(timesteps, desc="Standard sampling"):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents
    
    def _best_of_n_sampling(
        self,
        text_embeddings: torch.Tensor,
        latents: torch.Tensor,
        scheduler,
        guidance_scale: float,
        num_inference_steps: int,
        n_candidates: int
    ) -> torch.Tensor:
        """Best-of-N sampling with forward NLL evaluation"""
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps
        alphas_cumprod, betas, alphas = self._prepare_scheduler_coefficients(scheduler)
        
        latent_dims = latents.numel()
        
        for step_idx, t in enumerate(tqdm(timesteps, desc="Best-of-N sampling")):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            with torch.no_grad():
                noise_pred_out = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred_out.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute forward NLL for statistics
            t_idx = t.item()
            alpha_prod_t = alphas_cumprod[t_idx].float()
            beta_prod_t = 1 - alpha_prod_t
            variance_t = beta_prod_t.item()
            nll_current = self._compute_forward_nll_single(noise_pred.float(), variance_t, latent_dims)
            self.scores.append(nll_current)
            
            # Compute score function estimate
            sqrt_one_minus_alpha_prod_t = torch.sqrt(beta_prod_t)
            score_estimate = -noise_pred.float() / sqrt_one_minus_alpha_prod_t
            self.score_estimates.append(score_estimate.clone())
            
            # Predict x0
            sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
            pred_original_sample = (latents.float() - sqrt_one_minus_alpha_prod_t * noise_pred.float()) / sqrt_alpha_prod_t
            pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
            
            # Compute previous timestep
            prev_t = scheduler.previous_timestep(t)
            
            if prev_t >= 0:
                alpha_prod_t_prev = alphas_cumprod[prev_t.item()]
            else:
                alpha_prod_t_prev = torch.tensor(1.0, device=self.device, dtype=self.dtype)
            
            # Compute transition parameters
            current_alpha_t = alpha_prod_t / alpha_prod_t_prev
            current_beta_t = 1 - current_alpha_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            
            # Compute mean
            pred_original_sample_coeff = (torch.sqrt(alpha_prod_t_prev) * current_beta_t) / beta_prod_t
            current_sample_coeff = (torch.sqrt(current_alpha_t) * beta_prod_t_prev) / beta_prod_t
            pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
            
            # Compute variance
            variance = (beta_prod_t_prev / beta_prod_t) * current_beta_t
            variance = torch.clamp(variance, min=1e-20)
            std_dev_t = torch.sqrt(variance)
            
            if prev_t >= 0:
                # Generate N candidates and select best
                variance_prev = beta_prod_t_prev.item()
                min_nll = float('inf')
                best_latents = None
                
                for _ in range(n_candidates):
                    noise = torch.randn_like(latents)
                    candidate_latents = pred_prev_sample + std_dev_t * noise
                    
                    # Evaluate candidate
                    latent_model_input = torch.cat([candidate_latents] * 2)
                    latent_model_input = scheduler.scale_model_input(latent_model_input, prev_t)
                    with torch.no_grad():
                        noise_pred_out = self.unet(latent_model_input, prev_t, encoder_hidden_states=text_embeddings).sample
                    noise_pred_uncond, noise_pred_text = noise_pred_out.chunk(2)
                    noise_pred_candidate = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    nll_candidate = self._compute_forward_nll_single(noise_pred_candidate.float(), variance_prev, latent_dims)
                    
                    if nll_candidate < min_nll:
                        min_nll = nll_candidate
                        best_latents = candidate_latents
                
                latents = best_latents
            else:
                latents = pred_prev_sample
        
        return latents
    
    def _beam_search_sampling(
        self,
        text_embeddings: torch.Tensor,
        initial_latents: torch.Tensor,
        scheduler,
        guidance_scale: float,
        num_inference_steps: int,
        beam_width: int,
        num_candidates_per_beam: int,
        lookahead_steps: int
    ) -> torch.Tensor:
        """Beam search sampling with lookahead evaluation"""
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps
        alphas_cumprod, betas, alphas = self._prepare_scheduler_coefficients(scheduler)
        
        # Start with multiple beams
        latents = initial_latents.repeat(beam_width, 1, 1, 1)
        latent_dims = latents[0].numel()
        
        for step_idx, t in enumerate(tqdm(timesteps, desc="Beam search sampling")):
            batch_size = latents.shape[0]
            
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            # Prepare encoder hidden states
            encoder_hidden_states = torch.cat([
                text_embeddings[:1].repeat(batch_size, 1, 1),
                text_embeddings[1:].repeat(batch_size, 1, 1)
            ], dim=0)
            
            # Predict noise
            with torch.no_grad():
                noise_pred_out = self.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states).sample
            
            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred_out.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute current NLL for statistics
            t_idx = t.item()
            alpha_prod_t = alphas_cumprod[t_idx].float()
            beta_prod_t = 1 - alpha_prod_t
            variance_t = beta_prod_t.item()
            nll_current = self._compute_forward_nll_batch(noise_pred.float(), variance_t, latent_dims)
            self.scores.append(nll_current.mean().item())
            
            # Compute score function estimate
            sqrt_one_minus_alpha_prod_t = torch.sqrt(beta_prod_t)
            score_estimate = -noise_pred.float() / sqrt_one_minus_alpha_prod_t
            score_norms = torch.norm(score_estimate.view(batch_size, -1), dim=1).mean().item()
            self.score_estimates.append(score_norms)
            
            # Predict x0
            sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
            pred_original_sample = (latents.float() - sqrt_one_minus_alpha_prod_t * noise_pred.float()) / sqrt_alpha_prod_t
            pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
            
            # Compute previous timestep
            prev_t = scheduler.previous_timestep(t)
            
            if prev_t >= 0:
                alpha_prod_t_prev = alphas_cumprod[prev_t.item()]
            else:
                alpha_prod_t_prev = torch.tensor(1.0, device=self.device, dtype=self.dtype)
                latents = pred_original_sample
                continue
            
            # Compute transition parameters
            current_alpha_t = alpha_prod_t / alpha_prod_t_prev
            current_beta_t = 1 - current_alpha_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            
            # Compute mean
            pred_original_sample_coeff = (torch.sqrt(alpha_prod_t_prev) * current_beta_t) / beta_prod_t
            current_sample_coeff = (torch.sqrt(current_alpha_t) * beta_prod_t_prev) / beta_prod_t
            pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
            
            # Compute variance
            variance = (beta_prod_t_prev / beta_prod_t) * current_beta_t
            variance = torch.clamp(variance, min=1e-20)
            std_dev_t = torch.sqrt(variance)
            
            # Generate candidates
            total_candidates = batch_size * num_candidates_per_beam
            pred_prev_sample_rep = pred_prev_sample.repeat_interleave(num_candidates_per_beam, dim=0)
            noise = torch.randn((total_candidates, *latents.shape[1:]), device=self.device, dtype=self.dtype)
            candidate_latents = pred_prev_sample_rep + std_dev_t * noise
            
            # Evaluate candidates using lookahead
            lookahead_scores = torch.zeros(total_candidates, device=self.device, dtype=torch.float32)
            eval_latents = candidate_latents.clone()
            eval_timestep = prev_t
            
            for m in range(lookahead_steps):
                if eval_timestep < 0:
                    break
                
                # Compute noise prediction for evaluation
                eval_model_input = torch.cat([eval_latents] * 2)
                eval_model_input = scheduler.scale_model_input(eval_model_input, eval_timestep)
                eval_encoder_hidden = torch.cat([
                    text_embeddings[:1].repeat(total_candidates, 1, 1),
                    text_embeddings[1:].repeat(total_candidates, 1, 1)
                ], dim=0)
                
                with torch.no_grad():
                    eval_noise_pred_out = self.unet(eval_model_input, eval_timestep, encoder_hidden_states=eval_encoder_hidden).sample
                eval_noise_pred_uncond, eval_noise_pred_text = eval_noise_pred_out.chunk(2)
                eval_noise_pred = eval_noise_pred_uncond + guidance_scale * (eval_noise_pred_text - eval_noise_pred_uncond)
                
                # Compute NLL
                eval_t_idx = eval_timestep.item()
                eval_alpha_prod_t = alphas_cumprod[eval_t_idx].float()
                eval_variance_t = (1 - eval_alpha_prod_t).item()
                lookahead_nll = self._compute_forward_nll_batch(eval_noise_pred.float(), eval_variance_t, latent_dims)
                lookahead_scores += lookahead_nll
                
                # Advance to next timestep if not last
                if m < lookahead_steps - 1:
                    eval_sqrt_alpha_prod_t = torch.sqrt(eval_alpha_prod_t)
                    eval_sqrt_one_minus = torch.sqrt(1 - eval_alpha_prod_t)
                    eval_pred_original_sample = (eval_latents.float() - eval_sqrt_one_minus * eval_noise_pred.float()) / eval_sqrt_alpha_prod_t
                    eval_pred_original_sample = torch.clamp(eval_pred_original_sample, -1.0, 1.0)
                    
                    eval_prev_t = scheduler.previous_timestep(eval_timestep)
                    if eval_prev_t >= 0:
                        eval_alpha_prod_t_prev = alphas_cumprod[eval_prev_t.item()]
                    else:
                        eval_alpha_prod_t_prev = torch.tensor(1.0, device=self.device, dtype=self.dtype)
                    
                    eval_current_alpha_t = eval_alpha_prod_t / eval_alpha_prod_t_prev
                    eval_current_beta_t = 1 - eval_current_alpha_t
                    eval_beta_prod_t_prev = 1 - eval_alpha_prod_t_prev
                    
                    eval_pred_original_sample_coeff = (torch.sqrt(eval_alpha_prod_t_prev) * eval_current_beta_t) / (1 - eval_alpha_prod_t)
                    eval_current_sample_coeff = (torch.sqrt(eval_current_alpha_t) * eval_beta_prod_t_prev) / (1 - eval_alpha_prod_t)
                    eval_pred_prev_sample = eval_pred_original_sample_coeff * eval_pred_original_sample + eval_current_sample_coeff * eval_latents
                    
                    eval_latents = eval_pred_prev_sample
                    eval_timestep = eval_prev_t
            
            # Select top beam_width candidates
            _, top_indices = torch.topk(lookahead_scores, beam_width, largest=False)
            latents = candidate_latents[top_indices]
        
        return latents[0].unsqueeze(0)  # Return best beam
    
    def _decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """Decode latents to PIL Image"""
        latents = 1 / self.vae.config.scaling_factor * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample[0]
        
        # Post-process
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(1, 2, 0).numpy()
        image = (image * 255).round().astype("uint8")
        return Image.fromarray(image)
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        sampling_config: Optional[SamplingConfig] = None,
        return_statistics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, dict]]:
        """
        Generate an image using the specified sampling method.
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt for classifier-free guidance
            height: Image height in pixels
            width: Image width in pixels
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            generator: Random generator for reproducibility
            sampling_config: Configuration for sampling method and parameters
            return_statistics: Whether to return generation statistics
        
        Returns:
            Generated PIL Image, optionally with statistics dictionary
        """
        if sampling_config is None:
            sampling_config = SamplingConfig()
        
        # Clear previous statistics
        self.scores = []
        self.score_estimates = []
        
        # Select scheduler
        scheduler = self.scheduler_ddpm if sampling_config.use_stochastic_scheduler else self.scheduler_pndm
        
        # Encode prompts
        text_embeddings = self._encode_prompts(prompt, negative_prompt)
        
        # Prepare initial latents
        batch_size = 1 if sampling_config.method != "beam_search" else 1  # Start with 1, expand in beam search
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // self.vae_scale_factor, width // self.vae_scale_factor),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        latents = latents * scheduler.init_noise_sigma
        
        # Select sampling method
        if sampling_config.method == "standard":
            final_latents = self._standard_sampling(
                text_embeddings, latents, scheduler, guidance_scale, num_inference_steps
            )
        elif sampling_config.method == "best_of_n":
            final_latents = self._best_of_n_sampling(
                text_embeddings, latents, scheduler, guidance_scale, 
                num_inference_steps, sampling_config.n_candidates
            )
        elif sampling_config.method == "beam_search":
            final_latents = self._beam_search_sampling(
                text_embeddings, latents, scheduler, guidance_scale, num_inference_steps,
                sampling_config.beam_width, sampling_config.num_candidates_per_beam, 
                sampling_config.lookahead_steps
            )
        else:
            raise ValueError(f"Unknown sampling method: {sampling_config.method}")
        
        # Decode to image
        image = self._decode_latents(final_latents)
        
        if return_statistics:
            stats = {
                "method": sampling_config.method,
                "scores": self.scores.copy(),
                "score_estimates": [s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in self.score_estimates],
                "average_nll": np.mean(self.scores) if self.scores else None,
                "num_steps": len(self.scores)
            }
            return image, stats
        
        return image


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipe = EnhancedStableDiffusion(cache_dir="D:\\hf")
    
    prompt = "A fantasy landscape with mountains and rivers"
    
    # Standard sampling
    print("Generating with standard sampling...")
    config_standard = SamplingConfig(method="standard")
    image_standard = pipe.generate(prompt, sampling_config=config_standard)
    image_standard.save("enhanced_standard.png")
    
    # Best-of-N sampling
    print("Generating with Best-of-N sampling...")
    config_best_of_n = SamplingConfig(
        method="best_of_n", 
        n_candidates=10, 
        use_stochastic_scheduler=True
    )
    image_best_of_n, stats = pipe.generate(
        prompt, 
        sampling_config=config_best_of_n, 
        return_statistics=True
    )
    image_best_of_n.save("enhanced_best_of_n.png")
    print(f"Best-of-N average NLL: {stats['average_nll']:.4f}")
    
    # Beam search sampling
    print("Generating with Beam search sampling...")
    config_beam = SamplingConfig(
        method="beam_search",
        beam_width=3,
        num_candidates_per_beam=3,
        lookahead_steps=2,
        use_stochastic_scheduler=True
    )
    image_beam, stats = pipe.generate(
        prompt,
        sampling_config=config_beam,
        return_statistics=True
    )
    image_beam.save("enhanced_beam_search.png")
    print(f"Beam search average NLL: {stats['average_nll']:.4f}")
    
    print("Generation complete! Check the generated images.")
