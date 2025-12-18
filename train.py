import torch
from tqdm import tqdm

def train_visual_r1(tinker_model, train_loader, epochs=1, group_size=8):
    """
    The Visual R1 Training Loop via GRPO.
    
    Args:
        tinker_model: The wrapper for Qwen2.5-VL (handles GPU inference).
        train_loader: Yields batches of (image, question, ground_truth).
        group_size (G): How many completions to sample per question (e.g., 8).
    """
    
    # Optimizer is handled inside Tinker usually, or you initialize it here
    # optimizer = torch.optim.AdamW(tinker_model.parameters(), lr=1e-6)
    
    print(f"Starting Training: G={group_size}")
    
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            images, questions, ground_truths = batch
            
            # -------------------------------------------------------
            # Step 1: Experience Collection (Sampling)
            # -------------------------------------------------------
            # We ask the model to attempt each problem 'G' times.
            # Tinker handles the kv-caching and batching efficiently.
            
            # Construct prompts with the <image> placeholder
            prompts = [
                f"User: <|image_pad|> {q} Assistant: <think>" 
                for q in questions
            ]
            
            # duplicate images G times for the batch? 
            # Usually Tinker handles the broadcasting, but let's assume we pass lists.
            rollouts = tinker_model.sample(
                images=images,      # [B, H, W, C]
                prompts=prompts,    # [B]
                n=group_size,       # G
                temperature=1.0,    # High temp to encourage exploration/diversity
                max_tokens=512
            )
            # rollouts is likely a list of B*G dicts: 
            # [{'text': "...", 'logprobs': ...}, ...]

            # -------------------------------------------------------
            # Step 2: Reward Calculation (Local CPU)
            # -------------------------------------------------------
            # Extract text completions for reward scoring
            completions = [r['text'] for r in rollouts]
            
            # We need to align completions with their ground truths.
            # Since Tinker expands B prompts into B*G completions, we expand truths too.
            expanded_truths = [t for t in ground_truths for _ in range(group_size)]
            
            # Compute raw rewards using our verified function
            # This returns a list of floats, e.g., [1.3, 0.0, 1.1, ...]
            raw_rewards = visual_cot_reward_fn(completions, expanded_truths)
            
            # Normalize rewards (GRPO Logic: Advantage = (R - Mean) / Std)
            # This happens PER QUESTION group.
            advantages = compute_group_advantages(raw_rewards, group_size)
            
            # Monitor: Log the average reward to see if the model is learning
            avg_reward = sum(raw_rewards) / len(raw_rewards)
            progress_bar.set_postfix({"Avg Reward": f"{avg_reward:.3f}"})

            # -------------------------------------------------------
            # Step 3: Policy Optimization (Update)
            # -------------------------------------------------------
            # We pass the calculated advantages back to Tinker.
            # Tinker calculates the GRPO loss: -Advantage * log_prob(token)
            
            loss_stats = tinker_model.forward_backward(
                inputs=rollouts,       # Contains the tokens/logprobs needed for grad
                advantages=advantages, # The signal we just computed
                loss_type="grpo_clip"  # Equation 33 from the assignment
            )
            
            # Optional: Gradient Accumulation or Stepping
            # tinker_model.step() 
            
            if step % 10 == 0:
                print(f"Step {step}: Reward={avg_reward:.3f}, Loss={loss_stats['loss']:.4f}")

    print("Training Complete.")

def compute_group_advantages(raw_rewards, G):
    """
    Normalizes rewards within each group of G samples.
    """
    rewards_tensor = torch.tensor(raw_rewards)
    # Reshape to [Batch, Group_Size]
    # Assuming standard ordering: q1_s1, q1_s2... q2_s1...
    reshaped_rewards = rewards_tensor.view(-1, G)
    
    # Calculate Mean and Std per row (per question)
    means = reshaped_rewards.mean(dim=1, keepdim=True)
    stds = reshaped_rewards.std(dim=1, keepdim=True) + 1e-4 # epsilon
    
    # Calculate Advantage
    advantages = (reshaped_rewards - means) / stds
    
    # Flatten back to match the rollout list
    return advantages.flatten().tolist()