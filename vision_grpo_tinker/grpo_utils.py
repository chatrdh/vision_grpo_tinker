import torch
import torch.nn.functional as F

def compute_group_advantages(raw_rewards, group_size, epsilon=1e-4):
    """
    Computes Group Relative Policy Optimization (GRPO) advantages.
    Normalizes rewards within each group (batch of samples for the same image).
    
    Args:
        raw_rewards (list[float]): Flat list of rewards. Length = Batch * Group_Size.
        group_size (int): Number of samples per prompt (G).
        epsilon (float): Small constant for stability.
        
    Returns:
        list[float]: Flat list of normalized advantages.
    """
    # 1. Validation
    if len(raw_rewards) % group_size != 0:
        raise ValueError(f"Rewards length {len(raw_rewards)} not divisible by group size {group_size}")
        
    # 2. Convert to Tensor
    rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32)
    
    # 3. Reshape to [Batch, Group_Size]
    # Each row is one "group" (one image, G attempts)
    reshaped_rewards = rewards_tensor.view(-1, group_size)
    
    # 4. Compute Statistics (Mean & Std per group)
    means = reshaped_rewards.mean(dim=1, keepdim=True)
    stds = reshaped_rewards.std(dim=1, keepdim=True)
    
    # 5. Normalize
    # Advantage = (Reward - Mean) / (Std + epsilon)
    advantages = (reshaped_rewards - means) / (stds + epsilon)
    
    return advantages.flatten().tolist()


def get_log_probs(logits, labels):
    """
    Utility to extract log probabilities of the actual ground truth tokens 
    from the raw logits.
    """
    # logits: [batch, seq_len, vocab_size]
    # labels: [batch, seq_len]
    
    # Shift so that logits at index t predict token at index t+1
    # (Standard Causal LM shift)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute Log Softmax
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather the log prob of the actual target token
    # gather_index needs same dims as log_probs (except last)
    gathered_log_probs = torch.gather(
        log_probs, 
        dim=-1, 
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    return gathered_log_probs


def grpo_loss_fn(model_outputs, batch, clip_epsilon=0.2, beta=0.04):
    """
    The GRPO-Clip Loss Function (Eq 33 in Assignment).
    
    Args:
        model_outputs: The object returned by the model forward pass (must contain logits).
        batch (dict): Dictionary containing 'input_ids', 'attention_mask', and 'advantage'.
        clip_epsilon (float): Clipping range (e.g., 0.2 means 0.8 to 1.2).
        beta (float): KL penalty coefficient (optional, usually 0.04 or 0.1 for R1-Zero).
    
    Returns:
        torch.Tensor: The scalar loss value.
    """
    logits = model_outputs.logits
    input_ids = batch["input_ids"] 
    advantages = batch["advantage"] # Shape: [batch_size]
    
    # 1. Get Log Probs of the generated tokens
    # Note: For strict GRPO, we only train on the 'completion' part, not the prompt.
    # Assuming 'batch' handles masking or we treat the whole sequence as training target.
    current_log_probs = get_log_probs(logits, input_ids)
    
    # 2. Get Old Log Probs (Reference)
    # In strict Off-Policy GRPO, these are passed in the batch. 
    # In simple On-Policy (1 step per sample), old_log_probs == current_log_probs (detached).
    if "old_log_probs" in batch:
        old_log_probs = batch["old_log_probs"]
    else:
        # If missing, assume On-Policy: ratio starts at 1.0
        # We detach to stop gradients flowing into the "old" policy
        old_log_probs = current_log_probs.detach()
        
    # Masking: We typically only calculate loss on the completion tokens, not the prompt.
    # If 'labels' or 'loss_mask' is provided, apply it here.
    if "loss_mask" in batch:
        mask = batch["loss_mask"][..., 1:] # Shift to match log_probs
        current_log_probs = current_log_probs * mask
        old_log_probs = old_log_probs * mask

    # 3. Calculate Ratio (pi_theta / pi_old)
    # exp(log_a - log_b) = a / b
    ratios = torch.exp(current_log_probs - old_log_probs)
    
    # 4. Reshape Advantages to match Sequence Length
    # Advantage is a scalar per sample, so we broadcast it across the sequence tokens.
    # advantages: [batch] -> [batch, 1]
    adv_broadcast = advantages.view(-1, 1).to(ratios.device)
    
    # 5. Compute Surrogate Objectives (PPO/GRPO Logic)
    # Obj1: Unclipped * Advantage
    surr1 = ratios * adv_broadcast
    
    # Obj2: Clipped * Advantage
    surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * adv_broadcast
    
    # 6. Take Minimum (Pessimistic Bound)
    # We use negative because optimizers minimize loss (we want to maximize reward)
    # GRPO Loss = - min(surr1, surr2)
    grpo_loss = -torch.min(surr1, surr2)
    
    # 7. Optional: KL Penalty (Reference to Eq 33 usually includes a KL term or beta)
    # For pure R1-Zero, we often rely on clipping, but adding a KL term helps stability.
    # approx_kl = (old_log_probs - current_log_probs)
    # loss = grpo_loss + beta * approx_kl
    
    # Averaging over the batch and sequence
    if "loss_mask" in batch:
        return (grpo_loss * mask).sum() / mask.sum()
    else:
        return grpo_loss.mean()