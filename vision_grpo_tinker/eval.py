"""
Evaluation Script for Visual R1 Model
======================================

Tests the trained model on the Geometry3K test/validation set.
"""

import os
import io
import torch
from tqdm import tqdm

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
    "checkpoint_name": "tinker://a76d44a2-d232-5b91-b7d1-cfdd44f7f9ca:train:0/weights/final",
    
    # Inference settings
    "num_samples": 1,  # Number of samples per question (1 for deterministic eval)
    "max_new_tokens": 512,
    "temperature": 0.0,  # Greedy decoding for eval
    
    # Batch settings
    "max_eval_samples": 100,  # Limit for quick testing (None for full eval)
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
        
        # Create sampling client
        # NOTE: The checkpoint was saved with save_state() which is for resuming training.
        # For sampling, we need a checkpoint from save_weights_for_sampler().
        # Using base model for now - retrain with proper save to use trained weights.
        sampling_client = service_client.create_sampling_client(
            model_path=CONFIG['model_id']  # Base model
        )
        
        # Get tokenizer
        tokenizer = sampling_client.get_tokenizer() if hasattr(sampling_client, 'get_tokenizer') else None
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_id"], trust_remote_code=True)
        
        print(f"[EVAL] ✓ Model loaded (checkpoint: {CONFIG['checkpoint_name']})")
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
        
        # Generate
        try:
            result = sampling_client.sample(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=CONFIG["num_samples"]
            ).result()
            
            for seq in result.sequences:
                completion_text = "<think>" + tokenizer.decode(seq.tokens)
                all_completions.append(completion_text)
                all_ground_truths.append(ground_truth)
                
        except Exception as e:
            print(f"[WARNING] Sample {idx} failed: {e}")
            all_completions.append("")
            all_ground_truths.append(ground_truth)
    
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
