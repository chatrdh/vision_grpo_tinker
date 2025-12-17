import re
import math

# A dictionary of visual primitives common in Geometry3K/MathVista
# We categorize them to potentially weigh them differently later if needed.
VISUAL_PRIMITIVES = {
    "shapes": [
        "triangle", "circle", "square", "rectangle", "polygon", "trapezoid", 
        "parallelogram", "sector", "arc", "angle"
    ],
    "relationships": [
        "intersect", "parallel", "perpendicular", "adjacent", "opposite", 
        "tangent", "bisect", "midpoint", "overlap", "enclosed"
    ],
    "attributes": [
        "shaded", "dotted", "dashed", "solid", "vertex", "hypotenuse", 
        "radius", "diameter", "coordinate", "axis", "blue", "red", "green" 
    ]
}

# Flatten for easy searching
ALL_VISUAL_KEYWORDS = set([word for category in VISUAL_PRIMITIVES.values() for word in category])



def extract_content(text, tag):
    """
    Extracts content between <tag> and </tag>.
    Returns None if the tag is missing or malformed.
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def is_numeric_match(pred, truth, tolerance=1e-2):
    """
    Compares two strings numerically.
    Example: pred="30.0", truth="30" -> True
    """
    try:
        p_val = float(re.sub(r"[^\d.-]", "", pred)) # Remove non-numeric chars like $ or degrees
        t_val = float(re.sub(r"[^\d.-]", "", truth))
        return abs(p_val - t_val) <= tolerance
    except ValueError:
        return False

def is_option_match(pred, truth):
    """
    Compares multiple choice options.
    Example: pred="B", truth="B" -> True
    """
    # Normalize: remove parens, spaces, lowercasing
    clean_pred = pred.strip().lower().replace("(", "").replace(")", "")
    clean_truth = truth.strip().lower().replace("(", "").replace(")", "")
    return clean_pred == clean_truth


def visual_cot_reward_fn(completions, ground_truth, **kwargs):
    """
    Calculates rewards for a batch of completions.
    
    Args:
        completions (list[str]): The generated outputs from the model.
        ground_truth (list[str]): The correct answers (numeric or option).
        
    Returns:
        list[float]: A list of reward scores corresponding to each completion.
    """
    rewards = []

    for pred_text, truth in zip(completions, ground_truth):
        score = 0.0
        
        # --- Component 1: Format Reward (+0.1) ---
        # We strictly require the <think>...</think><answer>...</answer> structure.
        think_content = extract_content(pred_text, "think")
        answer_content = extract_content(pred_text, "answer")
        
        if think_content and answer_content:
            score += 0.1
        else:
            # If format is broken, we stop here (hard penalty) or return 0.
            # For GRPO, returning 0 is usually sufficient to discourage the behavior.
            rewards.append(0.0)
            continue

        # --- Component 2: Visual Grounding Reward (+0.2) ---
        # We check if the reasoning trace mentions visual features.
        # Strategy: Count unique visual keywords found in the thought trace.
        
        found_keywords = 0
        normalized_think = think_content.lower()
        
        for word in ALL_VISUAL_KEYWORDS:
            if word in normalized_think:
                found_keywords += 1
        
        # We cap the bonus to prevent "keyword stuffing" (listing dictionary words).
        # Logic: If they mention at least 3 distinct visual concepts, they get full visual credit.
        grounding_score = min(0.2, (found_keywords / 3) * 0.2)
        if is_correct:
            score += grounding_score  # Only reward looking if it led to the right answer!
        else:
            score += 0.0 # No participation award for hallucinating features

        # --- Component 3: Accuracy Reward (+1.0) ---
        # Check if the final answer matches the ground truth.
        
        is_correct = False
        
        # Check 1: Is it a numeric match?
        if is_numeric_match(answer_content, truth):
            is_correct = True
        # Check 2: Is it an option match (e.g., "A", "B")?
        elif is_option_match(answer_content, truth):
            is_correct = True
            
        if is_correct:
            score += 1.0
        
        rewards.append(score)

    return rewards