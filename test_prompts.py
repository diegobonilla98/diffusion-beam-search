"""
Test script for comparing sampling methods across multiple prompts.

This script loads prompts from prompts.json and generates images using:
1. Standard sampling
2. Best-of-N sampling

Results are saved in organized folders for easy comparison.
"""

import json
import time
from pathlib import Path
from enhanced_stable_diffusion import EnhancedStableDiffusion, SamplingConfig
import torch


def create_experiment_directories(base_path: str, prompt_names: list):
    """Create directory structure for the experiment"""
    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each sampling method
    methods = ["standard", "best_of_n"]
    
    for method in methods:
        method_dir = base_dir / method
        method_dir.mkdir(exist_ok=True)
        
        # Create numbered folders for each prompt
        for i, prompt_name in enumerate(prompt_names):
            prompt_dir = method_dir / f"prompt_{i:02d}_{prompt_name}"
            prompt_dir.mkdir(exist_ok=True)
    
    return base_dir


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Convert prompt text to a safe filename"""
    # Remove special characters and replace spaces with underscores
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    sanitized = "".join(c if c in safe_chars else "_" for c in text)
    
    # Remove multiple consecutive underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    
    # Trim to max length and remove trailing underscores
    sanitized = sanitized[:max_length].strip("_")
    
    return sanitized


def save_generation_info(filepath: str, prompt: str, config: SamplingConfig, 
                        generation_time: float, stats: dict = None):
    """Save generation metadata to a text file"""
    info = {
        "prompt": prompt,
        "sampling_method": config.method,
        "generation_time_seconds": generation_time,
        "config": {
            "method": config.method,
            "n_candidates": config.n_candidates if hasattr(config, 'n_candidates') else None,
            "beam_width": config.beam_width if hasattr(config, 'beam_width') else None,
            "num_candidates_per_beam": config.num_candidates_per_beam if hasattr(config, 'num_candidates_per_beam') else None,
            "lookahead_steps": config.lookahead_steps if hasattr(config, 'lookahead_steps') else None,
            "use_stochastic_scheduler": config.use_stochastic_scheduler
        }
    }
    
    if stats:
        info["statistics"] = {
            "average_nll": stats.get("average_nll"),
            "num_steps": stats.get("num_steps"),
            "method": stats.get("method")
        }
    
    info_path = filepath.replace(".png", "_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Method: {config.method}\n")
        f.write(f"Generation time: {generation_time:.2f} seconds\n")
        f.write(f"Use stochastic scheduler: {config.use_stochastic_scheduler}\n")
        
        if config.method == "best_of_n":
            f.write(f"N candidates: {config.n_candidates}\n")
        elif config.method == "beam_search":
            f.write(f"Beam width: {config.beam_width}\n")
            f.write(f"Candidates per beam: {config.num_candidates_per_beam}\n")
            f.write(f"Lookahead steps: {config.lookahead_steps}\n")
        
        if stats:
            f.write("\nStatistics:\n")
            f.write(f"Average NLL: {stats.get('average_nll', 'N/A')}\n")
            f.write(f"Number of steps: {stats.get('num_steps', 'N/A')}\n")


def run_experiment():
    """Main experiment function"""
    # Load prompts
    print("Loading prompts from prompts.json...")
    try:
        with open("prompts.json", "r", encoding="utf-8") as f:
            prompts = json.load(f)
    except FileNotFoundError:
        print("Error: prompts.json not found!")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing prompts.json: {e}")
        return
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Create prompt names for directory structure
    prompt_names = [sanitize_filename(prompt) for prompt in prompts]
    
    # Create experiment directories
    experiment_base = "./generated_images/experiment_01"
    base_dir = create_experiment_directories(experiment_base, prompt_names)
    print(f"Created experiment directories in: {base_dir}")
    
    # Initialize the enhanced stable diffusion pipeline
    print("Initializing Enhanced Stable Diffusion pipeline...")
    try:
        pipe = EnhancedStableDiffusion(cache_dir="D:\\hf")
        print("Pipeline initialized successfully!")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return
    
    # Configuration for sampling methods
    configs = {
        "standard": SamplingConfig(
            method="standard",
            use_stochastic_scheduler=False  # Use PNDM for speed
        ),
        "best_of_n": SamplingConfig(
            method="best_of_n",
            n_candidates=10,
            use_stochastic_scheduler=True  # Use DDPM for proper NLL evaluation
        )
    }
    
    # Generation parameters
    generation_params = {
        "height": 512,
        "width": 512,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "generator": torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
    }
    
    # Track overall experiment statistics
    total_prompts = len(prompts)
    total_generations = total_prompts * len(configs)
    completed_generations = 0
    start_time = time.time()
    
    print("\nStarting generation experiment:")
    print(f"- {total_prompts} prompts")
    print(f"- {len(configs)} sampling methods")
    print(f"- {total_generations} total generations")
    print(f"- Estimated time: {total_generations * 1.5:.1f} minutes (rough estimate)\n")
    
    # Run experiments
    for prompt_idx, prompt in enumerate(prompts):
        prompt_name = prompt_names[prompt_idx]
        print(f"\n{'='*60}")
        print(f"Prompt {prompt_idx + 1}/{total_prompts}: {prompt[:60]}...")
        print(f"{'='*60}")
        
        for method_name, config in configs.items():
            # Create output path
            method_dir = base_dir / method_name / f"prompt_{prompt_idx:02d}_{prompt_name}"
            image_path = method_dir / f"{method_name}_generation.png"
            
            # Skip if image already exists
            if image_path.exists():
                print(f"✓ Skipping {method_name} for prompt {prompt_idx+1}: image already exists.")
                completed_generations += 1
                continue

            print(f"\nGenerating with {method_name} sampling...")
            try:
                # Generate image
                gen_start_time = time.time()
                
                if method_name == "standard":
                    image = pipe.generate(
                        prompt=prompt,
                        sampling_config=config,
                        **generation_params
                    )
                    stats = None
                else:
                    image, stats = pipe.generate(
                        prompt=prompt,
                        sampling_config=config,
                        return_statistics=True,
                        **generation_params
                    )
                
                generation_time = time.time() - gen_start_time
                
                # Save image
                image.save(str(image_path), quality=100, subsampling=0)
                
                # Save generation info
                save_generation_info(str(image_path), prompt, config, generation_time, stats)
                
                # Print results
                print(f"✓ Generated in {generation_time:.2f}s")
                if stats and stats.get("average_nll"):
                    print(f"  Average NLL: {stats['average_nll']:.4f}")
                
                completed_generations += 1
                
                # Progress update
                progress = (completed_generations / total_generations) * 100
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time * total_generations / completed_generations
                remaining_time = estimated_total_time - elapsed_time
                
                print(f"  Progress: {completed_generations}/{total_generations} ({progress:.1f}%)")
                print(f"  Estimated remaining time: {remaining_time/60:.1f} minutes")
                
            except Exception as e:
                print(f"✗ Error generating with {method_name}: {e}")
                continue
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Completed generations: {completed_generations}/{total_generations}")
    print(f"Average time per generation: {total_time/completed_generations:.1f}s")
    print(f"Results saved in: {base_dir}")
    
    # Create a summary file
    summary_path = base_dir / "experiment_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Enhanced Stable Diffusion Experiment Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total prompts: {total_prompts}\n")
        f.write(f"Sampling methods: {list(configs.keys())}\n")
        f.write(f"Total generations: {total_generations}\n")
        f.write(f"Completed generations: {completed_generations}\n")
        f.write(f"Total experiment time: {total_time/60:.1f} minutes\n")
        f.write(f"Average time per generation: {total_time/completed_generations:.1f}s\n\n")
        
        f.write("Generation Parameters:\n")
        for key, value in generation_params.items():
            if key != "generator":
                f.write(f"  {key}: {value}\n")
        
        f.write("\nSampling Configurations:\n")
        for method_name, config in configs.items():
            f.write(f"  {method_name}:\n")
            f.write(f"    method: {config.method}\n")
            f.write(f"    use_stochastic_scheduler: {config.use_stochastic_scheduler}\n")
            if hasattr(config, 'n_candidates'):
                f.write(f"    n_candidates: {config.n_candidates}\n")
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    run_experiment()