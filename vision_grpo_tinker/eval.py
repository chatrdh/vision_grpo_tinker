
import os
import io
import torch
from tqdm import tqdm
import wandb

# --- Import Local Modules ---
from dataset import Geometry3K_PandasDataset
from reward import visual_cot_reward_fn_detailed

# --- Import Tinker ---
import tinker
from tinker import types

# --- Configuration ---
CONFIG = {
    # Dataset - use test set for evaluation
    "parquet_path": "test-00000-of-00001.parquet",  # or validation parquet
    
    # Model - use the trained checkpoint
    "model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    # For trained model eval, use tinker:// checkpoint path:
    "checkpoint_path": "tinker://363dd8ce-052d-5843-b283-3afd2d99b3b4:train:0/sampler_weights/sampler_10",
    # For baseline eval, use HuggingFace model:
    # "checkpoint_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    
    # Inference settings
    "num_samples": 1,  # Number of samples per question (1 for deterministic eval)
    "max_new_tokens": 768,
    "temperature": 0.0,  # Greedy decoding for eval
    
    # Batch settings
    "max_eval_samples": 50,  # Limit for quick testing (None for full eval)
    
    # Wandb Config
    "wandb_project": "visual-r1-eval",
    "wandb_run_name": "trained_model_eval",
}


def download_test_data():
    """Download test parquet if not exists."""
    import urllib.request
    
    parquet_path = CONFIG["parquet_path"]
    if not os.path.exists(parquet_path):
        print(f"Downloading {parquet_path}...")
        url = f"https://huggingface.co/datasets/hiyouga/geometry3k/resolve/main/data/{parquet_path}"
        urllib.request.urlretrieve(url, parquet_path)
        print("Download complete!")
    return parquet_path


def evaluate():
    print("=" * 80)
    print("VISUAL R1 EVALUATION")
    print("=" * 80)
    
    # Initialize wandb
    wandb.init(
        project=CONFIG["wandb_project"],
        name=CONFIG["wandb_run_name"],
        config=CONFIG,
    )
    print(f"[EVAL] \u2713 Wandb initialized: {wandb.run.name}")
    
    # 1. Download and load test data
    print(f"\n[EVAL] Loading test data...")
    try:
        parquet_path = download_test_data()
        dataset = Geometry3K_PandasDataset(parquet_path)
        print(f"[EVAL] ✓ Loaded {len(dataset)} test samples")
    except Exception as e:
        print(f"[ERROR] Failed to load test data: {e}")
        return
    
    # Limit samples for quick testing
    if CONFIG["max_eval_samples"]:
        eval_indices = list(range(min(CONFIG["max_eval_samples"], len(dataset))))
    else:
        eval_indices = list(range(len(dataset)))
    print(f"[EVAL] Evaluating {len(eval_indices)} samples")
    
    # 2. Initialize Tinker client with trained weights
    print(f"\n[EVAL] Loading trained model...")
    try:
        service_client = tinker.ServiceClient()
        
        # Create sampling client - use base_model for HF models, model_path for tinker:// checkpoints
        checkpoint = CONFIG['checkpoint_path']
        if checkpoint.startswith("tinker://"):
            sampling_client = service_client.create_sampling_client(
                model_path=checkpoint
            )
        else:
            sampling_client = service_client.create_sampling_client(
                base_model=checkpoint
            )
        
        # Get tokenizer
        tokenizer = sampling_client.get_tokenizer() if hasattr(sampling_client, 'get_tokenizer') else None
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_id"], trust_remote_code=True)
        
        print(f"[EVAL] ✓ Model loaded (checkpoint: {CONFIG['checkpoint_path']})")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # 3. Run evaluation
    print(f"\n[EVAL] Running inference...")
    
    all_completions = []
    all_ground_truths = []
    
    stop_sequences = [151645]  # <|im_end|>
    sampling_params = types.SamplingParams(
        max_tokens=CONFIG["max_new_tokens"],
        temperature=CONFIG["temperature"],
        stop=stop_sequences
    )
    
    for idx in tqdm(eval_indices, desc="Evaluating"):
        sample = dataset[idx]
        img_pil = sample["image"]
        raw_prompt = sample["prompt"]
        ground_truth = sample["ground_truth"]
        
        # Clean prompt
        clean_text = raw_prompt.replace("<|image_pad|>", "").replace("User:", "").replace("Assistant:", "").replace("<think>", "").strip()
        
        # Build prompt
        prompt = f"""<|im_start|>user
<|vision_start|><|vision_end|>{clean_text}

Think step by step. Provide reasoning in <think>...</think> tags and final answer in <answer>...</answer> tags.<|im_end|>
<|im_start|>assistant
<think>"""
        
        # Encode and create ModelInput with image
        prompt_tokens = tokenizer.encode(prompt)
        
        img_buffer = io.BytesIO()
        img_pil.save(img_buffer, format='PNG')
        image_bytes = img_buffer.getvalue()
        
        model_input = types.ModelInput(chunks=[
            types.EncodedTextChunk(tokens=prompt_tokens[:10]),
            types.ImageChunk(data=image_bytes, format="png"),
            types.EncodedTextChunk(tokens=prompt_tokens[10:]),
        ])
        
        # Generate with timeout to prevent hangs
        try:
            future = sampling_client.sample(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=CONFIG["num_samples"]
            )
            result = future.result(timeout=60)  # 60 second timeout per sample
            
            for seq in result.sequences:
                completion_text = "<think>" + tokenizer.decode(seq.tokens)
                all_completions.append(completion_text)
                all_ground_truths.append(ground_truth)
                
        except TimeoutError:
            print(f"[WARNING] Sample {idx} timed out after 60s - skipping")
            all_completions.append("")
            all_ground_truths.append(ground_truth)
        except Exception as e:
            print(f"[WARNING] Sample {idx} failed: {e}")
            all_completions.append("")
            all_ground_truths.append(ground_truth)
    
    # Count skipped samples
    n_skipped = sum(1 for c in all_completions if c == "")
    if n_skipped > 0:
        print(f"[EVAL] Note: {n_skipped} samples were skipped due to errors/timeouts")
    
    # 4. Compute metrics
    print(f"\n[EVAL] Computing metrics...")
    reward_details = visual_cot_reward_fn_detailed(all_completions, all_ground_truths, lenient=True)
    
    n_samples = len(all_completions)
    n_correct = sum(1 for r in reward_details['accuracy_rewards'] if r > 0)
    n_formatted = sum(1 for r in reward_details['format_rewards'] if r > 0)
    n_visual = sum(1 for r in reward_details['visual_rewards'] if r > 0)
    
    avg_reward = sum(reward_details['total_rewards']) / n_samples
    avg_accuracy = sum(reward_details['accuracy_rewards']) / n_samples
    avg_format = sum(reward_details['format_rewards']) / n_samples
    avg_visual = sum(reward_details['visual_rewards']) / n_samples
    
    # Count extraction methods
    from collections import Counter
    method_counts = Counter(reward_details['extraction_methods'])
    
    # 5. Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\n📊 Overall Metrics:")
    print(f"  Samples Evaluated: {n_samples}")
    print(f"  Total Reward:      {avg_reward:.4f}")
    print(f"\n✓ Accuracy:")
    print(f"  Correct:           {n_correct}/{n_samples} ({100*n_correct/n_samples:.1f}%)")
    print(f"  Avg Accuracy:      {avg_accuracy:.4f}")
    print(f"\n📝 Format:")
    print(f"  Formatted:         {n_formatted}/{n_samples} ({100*n_formatted/n_samples:.1f}%)")
    print(f"  Avg Format:        {avg_format:.4f}")
    print(f"\n👁 Visual Grounding:")
    print(f"  With Visual:       {n_visual}/{n_samples} ({100*n_visual/n_samples:.1f}%)")
    print(f"  Avg Visual:        {avg_visual:.4f}")
    print(f"\n📋 Extraction Methods:")
    for method, count in method_counts.most_common():
        print(f"  {method}: {count} ({100*count/n_samples:.1f}%)")
    
    print("\n" + "=" * 80)
    
    # 6. Show some examples
    print("\n📝 Sample Predictions (first 3):")
    for i in range(min(3, n_samples)):
        print(f"\n--- Sample {i} ---")
        print(f"Ground Truth: {all_ground_truths[i]}")
        print(f"Correct: {reward_details['accuracy_rewards'][i] > 0}")
        print(f"Extraction: {reward_details['extraction_methods'][i]}")
        print(f"Completion: {all_completions[i][:300]}...")
    
    # 7. Log to wandb
    wandb_metrics = {
        "eval/accuracy": n_correct / n_samples,
        "eval/format_rate": n_formatted / n_samples,
        "eval/visual_rate": n_visual / n_samples,
        "eval/avg_reward": avg_reward,
        "eval/avg_accuracy_reward": avg_accuracy,
        "eval/avg_format_reward": avg_format,
        "eval/avg_visual_reward": avg_visual,
        "eval/n_samples": n_samples,
        "eval/n_correct": n_correct,
        "eval/n_formatted": n_formatted,
    }
    # Add extraction method counts
    for method, count in method_counts.items():
        wandb_metrics[f"eval/extraction_{method}"] = count
    
    wandb.log(wandb_metrics)
    
    # Log predictions table
    predictions_table = wandb.Table(columns=["idx", "ground_truth", "prediction", "correct", "extraction_method"])
    for i in range(min(50, n_samples)):  # Log first 50 samples
        predictions_table.add_data(
            i,
            all_ground_truths[i],
            all_completions[i][:500],  # Truncate for table
            reward_details['accuracy_rewards'][i] > 0,
            reward_details['extraction_methods'][i]
        )
    wandb.log({"predictions": predictions_table})
    
    # Finish wandb run
    wandb.finish()
    print(f"[EVAL] ✓ Results logged to wandb")
    
    return {
        'accuracy': n_correct / n_samples,
        'format_rate': n_formatted / n_samples,
        'visual_rate': n_visual / n_samples,
        'avg_reward': avg_reward,
    }


if __name__ == "__main__":
    results = evaluate()
    if results:
        print(f"\n✓ Evaluation complete! Accuracy: {results['accuracy']*100:.1f}%")
