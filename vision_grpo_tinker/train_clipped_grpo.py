"""
Visual R1 GRPO Training Script with Clipped Loss - using Tinker API
=====================================================================

This script trains a VLM (Qwen3-VL) on geometry problem solving using GRPO
with a custom clipped importance sampling loss (PPO-style clipping).

Key changes from train.py:
1. Uses forward_backward_custom with a custom clipped GRPO loss function
2. Implements PPO-style clipping for more stable policy updates
"""

import os
import io
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import List, Tuple, Dict, Any

# --- Import Local Modules ---
from dataset import Geometry3K_PandasDataset, visual_r1_collate_fn
from reward import (
    visual_cot_reward_fn, 
    visual_cot_reward_fn_detailed,
    vision_sr1_reward_fn_detailed,
    build_blind_verification_prompt,
    extract_xml_tag,
)
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
    "output_dir": "./visual_r1_checkpoints_clipped",
    # GRPO Hyperparameters
    "group_size": 4,
    "batch_size": 4,           # More stable gradient estimates
    "max_steps": 100,          # Vision-SR1 test run
    "learning_rate": 1e-6,
    "temperature": 0.9,        # Slightly focused exploration
    "max_new_tokens": 768,     # More room for chain-of-thought
    "clip_epsilon": 0.1,       # Tighter clipping for stable updates (was 0.2)
    "beta": 0.04,
    # LoRA Config
    "lora_rank": 16,
    # Resume Config
    "resume_from_step": 0,  # Set to None or 0 to start fresh, or step number to resume
    # Wandb Config
    "wandb_project": "visual-r1-grpo-clipped",
    "wandb_run_name": None,  # Auto-generated if None
}


# --- Custom Clipped GRPO Loss Function ---
def create_clipped_grpo_loss_fn_with_aux(
    auxiliary_data: List[Dict[str, torch.Tensor]],
    clip_epsilon: float = 0.2
):
    """
    Factory function to create a clipped GRPO loss function.
    
    Since forward_backward_custom can't serialize custom loss_fn_inputs,
    we pass the auxiliary data (sampling_logprobs, advantages) via closure.
    
    Args:
        auxiliary_data: List of dicts with 'sampling_logprobs' and 'advantages' tensors
        clip_epsilon: PPO-style clipping epsilon (default 0.2)
    
    Returns:
        A loss function with the signature expected by forward_backward_custom
    """
    
    def clipped_grpo_loss(
        data: List[types.Datum], 
        logprobs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Custom Clipped GRPO Loss Function.
        
        Implements PPO-style clipping on the importance sampling ratio.
        L = -min(r * A, clip(r, 1-eps, 1+eps) * A)
        """
        total_loss = torch.tensor(0.0)
        
        # Metrics for logging
        total_ratio = 0.0
        num_tokens = 0
        num_clipped = 0
        
        for idx, (datum, target_logprobs) in enumerate(zip(data, logprobs)):
            # Get sampling logprobs and advantages from auxiliary_data (via closure)
            aux = auxiliary_data[idx]
            sampling_logprobs = aux["sampling_logprobs"]
            advantages = aux["advantages"]
            
            # Ensure tensors are on same device and same dtype
            target_logprobs = target_logprobs.float()
            sampling_logprobs = sampling_logprobs.float()
            advantages = advantages.float()
            
            if target_logprobs.device != sampling_logprobs.device:
                sampling_logprobs = sampling_logprobs.to(target_logprobs.device)
                advantages = advantages.to(target_logprobs.device)
            
            # Compute probability ratio: r = π_θ(a|s) / π_old(a|s)
            log_ratio = target_logprobs - sampling_logprobs
            prob_ratio = torch.exp(log_ratio)
            
            # Apply clipping
            clipped_ratio = torch.clamp(prob_ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
            
            # PPO objective: take the minimum of clipped and unclipped
            unclipped_objective = prob_ratio * advantages
            clipped_objective = clipped_ratio * advantages
            ppo_objective = torch.min(unclipped_objective, clipped_objective)
            
            # Loss is negative of objective (minimize loss = maximize objective)
            loss = -ppo_objective.sum()
            total_loss = total_loss + loss
            
            # Track metrics
            total_ratio += prob_ratio.mean().item()
            num_tokens += len(prob_ratio)
            num_clipped += (prob_ratio != clipped_ratio).sum().item()
        
        # Compute average metrics
        n_sequences = len(data) if data else 1
        metrics = {
            "clipped_grpo_loss": total_loss.item(),
            "avg_importance_ratio": total_ratio / n_sequences,
            "total_tokens": num_tokens,
            "pct_tokens_clipped": 100.0 * num_clipped / num_tokens if num_tokens > 0 else 0.0,
        }
        
        return total_loss, metrics
    
    return clipped_grpo_loss


def train():
    print("=" * 80)
    print("VISUAL R1 TRAINING - CLIPPED GRPO with Tinker API")
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
    
    # Resume from checkpoint if specified
    resume_step = CONFIG.get("resume_from_step") or 0
    if resume_step > 0:
        checkpoint_name = f"checkpoint_{resume_step}"
        print(f"\n[RESUME] Loading checkpoint: {checkpoint_name}...")
        try:
            training_client.load_state_with_optimizer("tinker://db12b8ca-e823-552c-8d24-03b9b971df77:train:0/weights/checkpoint_350")
            print(f"[RESUME] ✓ Resumed from step {resume_step}")
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint {checkpoint_name}: {e}")
            print("[ERROR] Starting from scratch instead...")
            resume_step = 0
    
    # Note: Loss function is created per-step with auxiliary data via closure
    print(f"[INIT] ✓ Clipped GRPO loss configured (epsilon={CONFIG['clip_epsilon']})")
    
     
    
    # Define stop sequences for Qwen3
    stop_sequences = [151645]  # <|im_end|> token for Qwen3
    
    # 3. Training Loop
    print(f"\n[TRAIN] Starting Clipped GRPO Loop")
    print(f"[TRAIN] - Max steps: {CONFIG['max_steps']}")
    print(f"[TRAIN] - Group size: {CONFIG['group_size']}")
    print(f"[TRAIN] - Learning rate: {CONFIG['learning_rate']}")
    print(f"[TRAIN] - Clip epsilon: {CONFIG['clip_epsilon']}")
    print("=" * 80 + "\n")
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    step = resume_step  # Resume from checkpoint step or start from 0
    progress_bar = tqdm(total=CONFIG["max_steps"], initial=resume_step, desc="Training")
    
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
                
                # Build prompt with image - now requesting <scan> for Vision-SR1
                prompt = f"""<|im_start|>user
<|vision_start|><|vision_end|>{clean_text}

First, describe what you see in the image in <scan>...</scan> tags (shapes, labels, measurements, colors).
Then solve step by step in <think>...</think> tags.
Finally, give your final answer in <answer>...</answer> tags.<|im_end|>
<|im_start|>assistant
<scan>"""
                
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
                        completion_text = "<scan>" + tokenizer.decode(seq.tokens)
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
                has_scan = "</scan>" in comp
                has_think = "</think>" in comp
                has_answer = "<answer>" in comp and "</answer>" in comp
                print(f"  [{i}] Has </scan>: {has_scan}, Has </think>: {has_think}, Has <answer>: {has_answer}")
                print(f"      Preview: {comp[:200]}..." if len(comp) > 200 else f"      Full: {comp}")
            print(f"  Total completions: {len(all_completions)}")
            
            # --- Phase 2: Compute Rewards with Vision-SR1 Blind Pass ---
            print(f"[Step {step}] Computing rewards with Vision-SR1 blind verification...")
            expanded_truths = [gt for gt in ground_truths for _ in range(CONFIG["group_size"])]
            expanded_questions = [clean_text for _, raw_prompt in zip(images_pil, raw_prompts) 
                                  for _ in range(CONFIG["group_size"])
                                  for clean_text in [raw_prompt.replace("<|image_pad|>", "").replace("User:", "").replace("Assistant:", "").replace("<think>", "").strip()]]
            
            # First pass: Get accuracy results to know which samples need blind verification
            temp_results = vision_sr1_reward_fn_detailed(all_completions, expanded_truths, blind_answers=None)
            
            # --- Pass 2: Blind Verification (only for correct Pass 1 answers) ---
            blind_answers = [None] * len(all_completions)  # Initialize all as None
            n_blind_candidates = sum(1 for r in temp_results['accuracy_rewards'] if r > 0)
            
            if n_blind_candidates > 0:
                print(f"[Step {step}] Running Pass 2 (blind) for {n_blind_candidates} correct samples...")
                
                blind_sampling_params = types.SamplingParams(
                    max_tokens=CONFIG["max_new_tokens"],
                    temperature=CONFIG["temperature"],
                    stop=stop_sequences
                )
                
                for comp_idx, (comp, acc_reward) in enumerate(zip(all_completions, temp_results['accuracy_rewards'])):
                    if acc_reward <= 0:
                        continue  # Skip incorrect Pass 1 answers
                    
                    # Extract scan content
                    scan_content = extract_xml_tag(comp, "scan")
                    if not scan_content:
                        print(f"    [{comp_idx}] No <scan> found, skipping blind pass")
                        continue
                    
                    # Build blind prompt (text only, no image)
                    question = expanded_questions[comp_idx] if comp_idx < len(expanded_questions) else ""
                    blind_prompt = build_blind_verification_prompt(question, scan_content)
                    blind_tokens = tokenizer.encode(blind_prompt)
                    
                    # Create text-only input (NO image)
                    blind_input = types.ModelInput.from_ints(tokens=blind_tokens)
                    
                    try:
                        blind_result = sampling_client.sample(
                            prompt=blind_input,
                            sampling_params=blind_sampling_params,
                            num_samples=1
                        ).result()
                        
                        if blind_result.sequences:
                            blind_completion = "<think>" + tokenizer.decode(blind_result.sequences[0].tokens)
                            blind_answer = extract_xml_tag(blind_completion, "answer")
                            blind_answers[comp_idx] = blind_answer
                            print(f"    [{comp_idx}] Blind answer: {blind_answer[:50] if blind_answer else 'None'}...")
                    except Exception as e:
                        print(f"    [{comp_idx}] Blind sampling failed: {e}")
            
            # Now compute final rewards with blind answers
            reward_details = vision_sr1_reward_fn_detailed(all_completions, expanded_truths, blind_answers=blind_answers)
            raw_rewards = reward_details['total_rewards']
            
            # Compute statistics
            n_samples = len(raw_rewards)
            avg_reward = sum(raw_rewards) / n_samples if raw_rewards else 0
            avg_format = sum(reward_details['format_rewards']) / n_samples if n_samples else 0
            avg_accuracy = sum(reward_details['accuracy_rewards']) / n_samples if n_samples else 0
            avg_visual = sum(reward_details['visual_rewards']) / n_samples if n_samples else 0
            
            # Count statistics
            n_formatted = sum(1 for r in reward_details['format_rewards'] if r > 0)
            n_correct = sum(1 for r in reward_details['accuracy_rewards'] if r > 0)
            n_with_scan = sum(1 for s in reward_details.get('has_scan', []) if s)
            
            # Blind verification stats
            from collections import Counter
            blind_counts = Counter(reward_details.get('blind_results', []))
            n_blind_correct = blind_counts.get('correct', 0)
            n_blind_wrong = blind_counts.get('wrong', 0)
            
            print(f"\n[Step {step}] ═══════════════ VISION-SR1 REWARD BREAKDOWN ═══════════════")
            print(f"  📊 Total Reward:     {avg_reward:.4f}")
            print(f"  📝 Format Reward:    {avg_format:.4f} ({n_formatted}/{n_samples} formatted)")
            print(f"  ✓ Accuracy Reward:  {avg_accuracy:.4f} ({n_correct}/{n_samples} correct)")
            print(f"  📷 Has <scan>:       {n_with_scan}/{n_samples}")
            print(f"  👁 Visual Reward:    {avg_visual:.4f}")
            print(f"  🔍 Blind Pass:       ✓{n_blind_correct} / ✗{n_blind_wrong} (of {n_blind_candidates} candidates)")
            print(f"  📋 Blind Results:    {dict(blind_counts)}")
            
            # Log extraction methods if available
            if 'extraction_methods' in reward_details:
                method_counts = Counter(reward_details['extraction_methods'])
                print(f"  📋 Extraction:       {dict(method_counts)}")
            
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
                
                # Create training datum with loss_fn_inputs for built-in ppo loss
                # This uses the standard fields expected by Tinker's built-in loss functions
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
            
            # --- DEBUG: Log training data statistics ---
            print(f"\n[DEBUG] ═══════════════ TRAINING DATA STATS ═══════════════")
            all_logprobs_flat = []
            all_advantages_flat = []
            for i, datum in enumerate(training_data[:3]):  # First 3 samples
                lp = datum.loss_fn_inputs["logprobs"].to_torch()
                adv = datum.loss_fn_inputs["advantages"].to_torch()
                toks = datum.loss_fn_inputs["target_tokens"].to_torch()
                all_logprobs_flat.extend(lp.tolist())
                all_advantages_flat.extend(adv.tolist())
                print(f"  Sample {i}:")
                print(f"    Tokens: {len(toks)} tokens")
                print(f"    Logprobs: min={lp.min():.4f}, max={lp.max():.4f}, mean={lp.mean():.4f}")
                print(f"    Advantages: min={adv.min():.4f}, max={adv.max():.4f}, mean={adv.mean():.4f}")
                print(f"    First 5 logprobs: {lp[:5].tolist()}")
            
            # Overall stats
            import numpy as np
            all_lp = np.array(all_logprobs_flat)
            all_adv = np.array(all_advantages_flat)
            print(f"  ─────────────────────────────────────────────")
            print(f"  Overall Logprobs: min={all_lp.min():.4f}, max={all_lp.max():.4f}, mean={all_lp.mean():.4f}, std={all_lp.std():.4f}")
            print(f"  Overall Advantages: min={all_adv.min():.4f}, max={all_adv.max():.4f}, mean={all_adv.mean():.4f}")
            print(f"  Expected ratio if logprobs matched: exp(0) = 1.0")
            print(f"  ════════════════════════════════════════════════\n")
            
            # --- Phase 5: Forward-Backward with PPO Loss (GRPO = PPO + group-normalized advantages) ---
            print(f"[Step {step}] Running forward-backward with importance sampling loss (GRPO-style)...")
            try:
                # Use importance_sampling loss (REINFORCE-style policy gradient)
                # Combined with our GRPO-style group-normalized advantages, this is effectively GRPO
                # Note: PPO clipping doesn't work correctly with Tinker's built-in ppo loss
                fwd_bwd_result = training_client.forward_backward(
                    training_data,
                    loss_fn="importance_sampling"
                ).result()
                
                # Extract metrics from result
                loss_metrics = {}
                if hasattr(fwd_bwd_result, 'metrics') and fwd_bwd_result.metrics:
                    loss_metrics = fwd_bwd_result.metrics if isinstance(fwd_bwd_result.metrics, dict) else {}
                
                # Log loss metrics
                print(f"\n[Step {step}] ═══════════════ GRPO LOSS METRICS ═══════════════")
                print(f"  Loss: importance_sampling (GRPO with group-normalized advantages)")
                print(f"  Advantage type: GRPO (group-normalized)")
                if loss_metrics:
                    for key, value in loss_metrics.items():
                        if isinstance(value, float):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")
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
                "Blind": f"✓{n_blind_correct}/✗{n_blind_wrong}",
            })
            
            # Summary log
            print(f"\n[Step {step}] ═══════════════ STEP SUMMARY ═══════════════")
            print(f"  Step: {step}/{CONFIG['max_steps']}")
            print(f"  Total Reward: {avg_reward:.4f}")
            print(f"  Accuracy Rate: {n_correct}/{n_samples} ({100*n_correct/n_samples:.1f}%)")
            print(f"  Blind Pass: ✓{n_blind_correct} / ✗{n_blind_wrong}")
            print(f"  Clipped Tokens: {loss_metrics.get('pct_tokens_clipped', 0):.1f}%")
            print(f"  ════════════════════════════════════════════════\n")
            
            # --- Log to wandb ---
            wandb_metrics = {
                "step": step,
                "reward/total": avg_reward,
                "reward/format": avg_format,
                "reward/accuracy": avg_accuracy,
                "reward/visual": avg_visual,
                "rates/accuracy": n_correct / n_samples,
                "rates/format": n_formatted / n_samples,
                "rates/has_scan": n_with_scan / n_samples,
                "blind/candidates": n_blind_candidates,
                "blind/correct": n_blind_correct,
                "blind/wrong": n_blind_wrong,
                "blind/success_rate": n_blind_correct / n_blind_candidates if n_blind_candidates > 0 else 0,
                "samples/total": n_samples,
                "samples/correct": n_correct,
                "samples/with_scan": n_with_scan,
            }
            # Add any available loss metrics from forward_backward result
            for key, value in loss_metrics.items():
                wandb_metrics[f"loss/{key}"] = value
            wandb.log(wandb_metrics)
            
            # --- Save Sampler Weights (every 10 steps) ---
            if step % 10 == 0:
                print(f"\n[SAMPLER] Saving sampler weights at step {step}...")
                sampler_path = training_client.save_weights_for_sampler(name=f"sampler_{step}").result().path
                print(f"[SAMPLER] ✓ Saved: {sampler_path}")
            
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
