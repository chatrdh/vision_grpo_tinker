"""
Visual R1 GRPO Training Script using Tinker API
================================================

This script trains a VLM (Qwen3-VL) on geometry problem solving using GRPO.

Key changes from original:
1. Removed dependency on tinker_cookbook (not installed)
2. Use Tinker API directly for VLM inputs
3. Simplified data pipeline
"""

import os
import io
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# --- Import Local Modules ---
from dataset import Geometry3K_PandasDataset, visual_r1_collate_fn
from reward import visual_cot_reward_fn, visual_cot_reward_fn_detailed
from grpo_utils import compute_group_advantages

# --- Import Tinker ---
import tinker
from tinker import types, TensorData

# --- Import Transformers for tokenizer ---
from transformers import AutoTokenizer, AutoProcessor

# --- Configuration ---
CONFIG = {
    "parquet_path": "train-00000-of-00001.parquet",
    "model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "output_dir": "./visual_r1_checkpoints",
    # GRPO Hyperparameters
    "group_size": 4,
    "batch_size": 4,           # More stable gradient estimates
    "max_steps": 500,          # Longer training for convergence
    "learning_rate": 1e-6,
    "temperature": 0.9,        # Slightly focused exploration
    "max_new_tokens": 768,     # More room for chain-of-thought
    "clip_epsilon": 0.2,
    "beta": 0.04,
    # LoRA Config
    "lora_rank": 16,
    # Wandb Config
    "wandb_project": "visual-r1-grpo",
    "wandb_run_name": None,  # Auto-generated if None
}


def train():
    print("=" * 80)
    print("VISUAL R1 TRAINING - GRPO with Tinker API")
    print("=" * 80)
    
    # Initialize wandb
    wandb.init(
        project=CONFIG["wandb_project"],
        name=CONFIG["wandb_run_name"],
        config=CONFIG,
    )
    print(f"[INIT] ✓ Wandb initialized: {wandb.run.name}")
    
    # 1. Initialize Dataset
    print(f"\n[INIT] Loading Dataset from {CONFIG['parquet_path']}...")
    try:
        dataset = Geometry3K_PandasDataset(CONFIG['parquet_path'])
        print(f"[INIT] ✓ Dataset loaded successfully")
        print(f"[INIT] - Total samples: {len(dataset)}")
    except FileNotFoundError:
        print(f"[ERROR] Dataset file not found: {CONFIG['parquet_path']}")
        print("[ERROR] Run 'python dataset.py' first to download the dataset.")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=visual_r1_collate_fn,
        drop_last=True
    )
    print(f"[INIT] ✓ DataLoader created (batch_size={CONFIG['batch_size']})")
    
    # 2. Initialize Tinker Clients
    print(f"\n[INIT] Initializing Tinker clients...")
    try:
        service_client = tinker.ServiceClient()
        print(f"[INIT] ✓ Service client created")
        
        training_client = service_client.create_lora_training_client(
            base_model=CONFIG["model_id"],
            rank=CONFIG["lora_rank"],
        )
        print(f"[INIT] ✓ Training client created (LoRA rank={CONFIG['lora_rank']})")
        
        # Get tokenizer from training client
        tokenizer = training_client.get_tokenizer()
        print(f"[INIT] ✓ Tokenizer loaded")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize Tinker: {e}")
        print("[ERROR] Make sure TINKER_API_KEY is set")
        return
    
    # Define stop sequences for Qwen3
    stop_sequences = [151645]  # <|im_end|> token for Qwen3
    
    # 3. Training Loop
    print(f"\n[TRAIN] Starting GRPO Loop")
    print(f"[TRAIN] - Max steps: {CONFIG['max_steps']}")
    print(f"[TRAIN] - Group size: {CONFIG['group_size']}")
    print(f"[TRAIN] - Learning rate: {CONFIG['learning_rate']}")
    print("=" * 80 + "\n")
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    step = 0
    progress_bar = tqdm(total=CONFIG["max_steps"], desc="Training")
    
    while step < CONFIG["max_steps"]:
        for batch_idx, batch in enumerate(dataloader):
            if step >= CONFIG["max_steps"]:
                break
            
            images_pil, raw_prompts, ground_truths, ids = batch
            
            # --- Phase 1: Sample Trajectories ---
            print(f"\n[Step {step}] Sampling trajectories...")
            
            # Get sampling client with current weights
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"step_{step}"
            )
            
            all_completions = []
            all_sample_data = []  # (tokens, logprobs) tuples
            
            sampling_params = types.SamplingParams(
                max_tokens=CONFIG["max_new_tokens"],
                temperature=CONFIG["temperature"],
                stop=stop_sequences
            )
            
            for sample_idx, (img_pil, raw_prompt) in enumerate(zip(images_pil, raw_prompts)):
                # Clean the prompt text
                clean_text = raw_prompt.replace("<|image_pad|>", "").replace("User:", "").replace("Assistant:", "").replace("<think>", "").strip()
                
                # Build prompt with image
                prompt = f"""<|im_start|>user
<|vision_start|><|vision_end|>{clean_text}

Think step by step. Provide reasoning in <think>...</think> tags and final answer in <answer>...</answer> tags.<|im_end|>
<|im_start|>assistant
<think>"""
                
                # Encode prompt 
                prompt_tokens = tokenizer.encode(prompt)
                
                # Convert image to bytes
                img_buffer = io.BytesIO()
                img_pil.save(img_buffer, format='PNG')
                image_bytes = img_buffer.getvalue()
                
                # Create ModelInput with image
                model_input = types.ModelInput(chunks=[
                    types.EncodedTextChunk(tokens=prompt_tokens[:10]),  # Pre-image tokens
                    types.ImageChunk(data=image_bytes, format="png"),
                    types.EncodedTextChunk(tokens=prompt_tokens[10:]),  # Post-image tokens
                ])
                
                # Sample G completions
                try:
                    result = sampling_client.sample(
                        prompt=model_input,
                        sampling_params=sampling_params,
                        num_samples=CONFIG["group_size"]
                    ).result()
                    
                    for seq in result.sequences:
                        completion_text = "<think>" + tokenizer.decode(seq.tokens)
                        all_completions.append(completion_text)
                        all_sample_data.append((list(seq.tokens), list(seq.logprobs) if seq.logprobs else []))
                        
                except Exception as e:
                    print(f"[WARNING] Sampling failed for sample {sample_idx}: {e}")
                    # Add empty completions
                    for _ in range(CONFIG["group_size"]):
                        all_completions.append("")
                        all_sample_data.append(([], []))
            
            if not all_completions:
                print("[WARNING] No completions generated, skipping step")
                continue
            
            # --- DEBUG: Show sample completions ---
            print(f"\n[DEBUG] Sample completions (first 2):")
            for i, comp in enumerate(all_completions[:2]):
                has_think = "</think>" in comp
                has_answer = "<answer>" in comp and "</answer>" in comp
                print(f"  [{i}] Has </think>: {has_think}, Has <answer>: {has_answer}")
                print(f"      Preview: {comp[:200]}..." if len(comp) > 200 else f"      Full: {comp}")
            print(f"  Total completions: {len(all_completions)}")
            
            # --- Phase 2: Compute Rewards ---
            print(f"[Step {step}] Computing rewards...")
            expanded_truths = [gt for gt in ground_truths for _ in range(CONFIG["group_size"])]
            
            # Get detailed reward breakdown
            reward_details = visual_cot_reward_fn_detailed(all_completions, expanded_truths)
            raw_rewards = reward_details['total_rewards']
            
            # Compute statistics
            n_samples = len(raw_rewards)
            avg_reward = sum(raw_rewards) / n_samples if raw_rewards else 0
            avg_format = sum(reward_details['format_rewards']) / n_samples if n_samples else 0
            avg_accuracy = sum(reward_details['accuracy_rewards']) / n_samples if n_samples else 0
            avg_visual = sum(reward_details['visual_rewards']) / n_samples if n_samples else 0
            avg_keywords = sum(reward_details['found_keywords']) / n_samples if n_samples else 0
            
            # Count statistics
            n_formatted = sum(1 for r in reward_details['format_rewards'] if r > 0)
            n_correct = sum(1 for r in reward_details['accuracy_rewards'] if r > 0)
            n_with_visual = sum(1 for r in reward_details['visual_rewards'] if r > 0)
            
            print(f"\n[Step {step}] ═══════════════ REWARD BREAKDOWN ═══════════════")
            print(f"  📊 Total Reward:     {avg_reward:.4f}")
            print(f"  📝 Format Reward:    {avg_format:.4f} ({n_formatted}/{n_samples} formatted)")
            print(f"  ✓ Accuracy Reward:  {avg_accuracy:.4f} ({n_correct}/{n_samples} correct)")
            print(f"  👁 Visual Reward:    {avg_visual:.4f} ({n_with_visual}/{n_samples} with visual)")
            print(f"  🔑 Avg Keywords:     {avg_keywords:.2f}")
            
            # Log extraction methods if available
            if 'extraction_methods' in reward_details:
                from collections import Counter
                method_counts = Counter(reward_details['extraction_methods'])
                print(f"  📋 Extraction Methods: {dict(method_counts)}")
            
            print(f"  ════════════════════════════════════════════════")
            
            # --- Phase 3: Compute Advantages ---
            print(f"[Step {step}] Computing advantages...")
            try:
                advantages = compute_group_advantages(raw_rewards, CONFIG["group_size"])
            except ValueError as e:
                print(f"[WARNING] Advantage computation failed: {e}")
                continue
            
            # --- Phase 4: Build Training Data ---
            print(f"[Step {step}] Building training data...")
            training_data = []
            
            for traj_idx, ((tokens, logprobs), adv) in enumerate(zip(all_sample_data, advantages)):
                if not tokens or not logprobs:
                    continue
                
                # Ensure logprobs match tokens
                min_len = min(len(tokens), len(logprobs))
                if min_len == 0:
                    continue
                
                tokens = tokens[:min_len]
                logprobs = logprobs[:min_len]
                
                # Create training datum
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(tokens, dtype=torch.long)),
                        "logprobs": TensorData.from_torch(torch.tensor(logprobs, dtype=torch.float32)),
                        "advantages": TensorData.from_torch(torch.tensor([float(adv)] * len(tokens), dtype=torch.float32)),
                    }
                )
                training_data.append(datum)
            
            if not training_data:
                print("[WARNING] No valid training data, skipping step")
                continue
            
            print(f"[Step {step}] Built {len(training_data)} training datums")
            
            # --- Phase 5: Forward-Backward with Importance Sampling Loss ---
            print(f"[Step {step}] Running forward-backward...")
            try:
                fwd_bwd_result = training_client.forward_backward(
                    training_data,
                    loss_fn="importance_sampling"  # REINFORCE-style policy gradient
                ).result()
                
                # Log loss metrics from forward_backward
                print(f"\n[Step {step}] ═══════════════ LOSS METRICS ═══════════════")
                
                # Use the metrics attribute for loss info
                if hasattr(fwd_bwd_result, 'metrics') and fwd_bwd_result.metrics:
                    metrics = fwd_bwd_result.metrics
                    if isinstance(metrics, dict):
                        for key, value in metrics.items():
                            print(f"  {key}: {value}")
                    else:
                        print(f"  Metrics: {metrics}")
                else:
                    print(f"  No metrics available")
                
                print(f"  ════════════════════════════════════════════════")
                print(f"[Step {step}] ✓ Forward-backward complete")
            except Exception as e:
                print(f"[ERROR] Forward-backward failed: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # --- Phase 6: Optimizer Step ---
            print(f"[Step {step}] Applying optimizer step...")
            try:
                training_client.optim_step(
                    types.AdamParams(learning_rate=CONFIG["learning_rate"])
                ).result()
                print(f"[Step {step}] ✓ Optimizer step complete")
            except Exception as e:
                print(f"[ERROR] Optimizer step failed: {e}")
                continue
            
            # --- Update Progress ---
            step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                "Rew": f"{avg_reward:.3f}",
                "Acc": f"{avg_accuracy:.2f}",
                "Fmt": f"{avg_format:.2f}",
            })
            
            # Summary log
            print(f"\n[Step {step}] ═══════════════ STEP SUMMARY ═══════════════")
            print(f"  Step: {step}/{CONFIG['max_steps']}")
            print(f"  Total Reward: {avg_reward:.4f}")
            print(f"  Accuracy Rate: {n_correct}/{n_samples} ({100*n_correct/n_samples:.1f}%)")
            print(f"  Format Rate: {n_formatted}/{n_samples} ({100*n_formatted/n_samples:.1f}%)")
            print(f"  ════════════════════════════════════════════════\n")
            
            # --- Log to wandb ---
            wandb.log({
                "step": step,
                "reward/total": avg_reward,
                "reward/format": avg_format,
                "reward/accuracy": avg_accuracy,
                "reward/visual": avg_visual,
                "reward/avg_keywords": avg_keywords,
                "rates/accuracy": n_correct / n_samples,
                "rates/format": n_formatted / n_samples,
                "rates/visual": n_with_visual / n_samples,
                "samples/total": n_samples,
                "samples/correct": n_correct,
                "samples/formatted": n_formatted,
            })
            
            # --- Checkpoint ---
            if step % 50 == 0:
                print(f"\n[CHECKPOINT] Saving at step {step}...")
                training_client.save_state(name=f"checkpoint_{step}")
                print(f"[CHECKPOINT] ✓ Saved")
    
    progress_bar.close()
    print("\n" + "=" * 80)
    print("[COMPLETE] Training finished! Saving final model...")
    training_client.save_state(name="final")
    print("[COMPLETE] ✓ Training state saved (for resuming training)")
    
    # Save sampler-compatible weights for inference/evaluation
    sampler_path = training_client.save_weights_for_sampler(name="final").result().path
    print(f"[COMPLETE] ✓ Sampler weights saved: {sampler_path}")
    print("[COMPLETE] Use this path in eval.py for evaluation")
    print("=" * 80)
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    train()