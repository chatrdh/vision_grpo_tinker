import re
from grader import grade, extract_answer

# --- Configuration: Visual Primitives ---
# These are the keywords we want to see in the <think> block
VISUAL_PRIMITIVES = {
    "shapes": [
        "triangle", "circle", "square", "rectangle", "polygon", "trapezoid", 
        "parallelogram", "sector", "arc", "angle", "rhombus", "ellipse"
    ],
    "relationships": [
        "intersect", "parallel", "perpendicular", "adjacent", "opposite", 
        "tangent", "bisect", "midpoint", "overlap", "enclosed", "congruent", 
        "similar", "corresponding"
    ],
    "attributes": [
        "shaded", "dotted", "dashed", "solid", "vertex", "hypotenuse", 
        "radius", "diameter", "coordinate", "axis", "blue", "red", "green", 
        "perimeter", "area", "volume", "slope"
    ]
}

# Flatten for efficient searching
ALL_VISUAL_KEYWORDS = set([w for cat in VISUAL_PRIMITIVES.values() for w in cat])


def extract_xml_tag(text, tag):
    """
    Extracts the content inside <tag>...</tag>.
    Returns None if the tag is missing or the structure is broken.
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_answer_flexible(text):
    """
    Try multiple strategies to extract an answer from text.
    
    Priority:
    1. <answer>...</answer> tags
    2. \\boxed{...} LaTeX
    3. "answer is X" pattern
    4. "= X" at end of text
    5. Last number in text
    """
    # 1. Try <answer> tags
    answer = extract_xml_tag(text, "answer")
    if answer:
        return answer, "answer_tag"
    
    # 2. Try \boxed{}
    boxed = extract_answer(text)  # From grader.py
    if boxed:
        return boxed, "boxed"
    
    # 3. Try "answer is X" or "answer: X" pattern
    patterns = [
        r"(?:the\s+)?answer\s+is\s*[:\s]*([A-Za-z0-9\.\-\+\/\\\{\}\^\s]+)",
        r"(?:final\s+)?answer\s*[:\s]+([A-Za-z0-9\.\-\+\/\\\{\}\^\s]+)",
        r"=\s*([A-Za-z0-9\.\-\+\/\\\{\}\^\s]+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            ans = match.group(1).strip().rstrip('.')
            if ans:
                return ans, "pattern"
    
    # 4. Try last number
    numbers = re.findall(r'[\d\.\-\+]+', text)
    if numbers:
        return numbers[-1], "last_number"
    
    return None, "none"


def visual_cot_reward_fn(completions, ground_truths, **kwargs):
    """
    The Core Reward Function for Visual R1.
    
    Components:
    1. Format Reward (+0.3): Did the model use <think>...</think> and <answer>...</answer> tags properly?
    2. Accuracy Reward (+1.0): Is the math mathematically equivalent to the truth?
    3. Visual Reward (+0.2): Did the model describe image features? (Gated by Accuracy)
    
    Args:
        completions (list[str]): The generated strings from the model.
        ground_truths (list[str]): The verifiable answers (LaTeX/Numbers) from the dataset.
        
    Returns:
        list[float]: A list of scalar rewards for the GRPO loop.
    """
    rewards = []

    for pred_text, truth_text in zip(completions, ground_truths):
        score = 0.0
        
        # --- 1. Format Check ---
        think_content = extract_xml_tag(pred_text, "think")
        answer_content = extract_xml_tag(pred_text, "answer")
        
        # We strictly require BOTH opening and closing tags for proper format
        # Check for </think> explicitly - the model often forgets to close it
        has_think_close = "</think>" in pred_text
        
        if think_content and answer_content and has_think_close:
            score += 0.3  # Full format reward for proper structure
        elif answer_content:
            score += 0.05  # Minimal credit if answer tag exists but format is broken
        else:
            # If no answer tags at all, penalize heavily
            rewards.append(0.0)
            continue

        # --- 2. Accuracy Check (Using grader.py) ---
        # Note: grade() handles the heavy lifting of LaTeX parsing and symbolic equality.
        # It returns True if the answer is mathematically correct.
        is_correct = grade(answer_content, truth_text)
        
        if is_correct:
            score += 1.0
            
            # --- 3. Visual Grounding Bonus (Gated) ---
            # We ONLY check for visual keywords if the answer is correct.
            # This prevents the model from "reward hacking" (babbling about shapes to get +0.2).
            
            found_keywords = 0
            normalized_think = think_content.lower()
            
            for word in ALL_VISUAL_KEYWORDS:
                if word in normalized_think:
                    found_keywords += 1
            
            # Bonus Logic:
            # We cap the bonus at 0.2. 
            # 1 keyword = +0.05, 4+ keywords = +0.20
            visual_bonus = min(0.2, found_keywords * 0.05)
            score += visual_bonus
            
        rewards.append(score)

    return rewards


def visual_cot_reward_fn_detailed(completions, ground_truths, lenient=True, **kwargs):
    """
    Same as visual_cot_reward_fn but returns detailed breakdown of rewards.
    
    Args:
        completions: Model outputs
        ground_truths: Expected answers
        lenient: If True, try flexible answer extraction when tags are missing
    
    Returns:
        dict: {
            'total_rewards': list[float],
            'format_rewards': list[float],
            'accuracy_rewards': list[float],
            'visual_rewards': list[float],
            'found_keywords': list[int],
            'extraction_methods': list[str],  # How the answer was extracted
        }
    """
    total_rewards = []
    format_rewards = []
    accuracy_rewards = []
    visual_rewards = []
    found_keywords_list = []
    extraction_methods = []

    for pred_text, truth_text in zip(completions, ground_truths):
        format_reward = 0.0
        accuracy_reward = 0.0
        visual_reward = 0.0
        found_keywords = 0
        extraction_method = "none"
        
        # --- 1. Format Check ---
        think_content = extract_xml_tag(pred_text, "think")
        answer_content = extract_xml_tag(pred_text, "answer")
        
        # Check for proper format - require BOTH opening and closing tags
        has_think_close = "</think>" in pred_text
        has_proper_format = think_content and answer_content and has_think_close
        
        if has_proper_format:
            format_reward = 0.3  # Full reward for proper structure
            extraction_method = "answer_tag"
        elif think_content and answer_content:
            # Has both tags but missing </think> closing
            format_reward = 0.1  # Reduced reward
            extraction_method = "answer_tag"
        elif lenient:
            # Try flexible extraction
            answer_content, extraction_method = extract_answer_flexible(pred_text)
            # Partial format credit if we have some structure
            if "<think>" in pred_text:
                format_reward = 0.02  # Minimal credit for trying
                # Use everything after <think> as thinking content
                think_start = pred_text.find("<think>") + len("<think>")
                think_content = pred_text[think_start:]
        
        # If no answer found at all, skip
        if not answer_content:
            total_rewards.append(0.0)
            format_rewards.append(0.0)
            accuracy_rewards.append(0.0)
            visual_rewards.append(0.0)
            found_keywords_list.append(0)
            extraction_methods.append("none")
            continue

        # --- 2. Accuracy Check ---
        try:
            is_correct = grade(answer_content, truth_text)
        except Exception:
            is_correct = False
        
        if is_correct:
            accuracy_reward = 1.0
            
            # --- 3. Visual Grounding Bonus (Gated) ---
            if think_content:
                normalized_think = think_content.lower()
                
                for word in ALL_VISUAL_KEYWORDS:
                    if word in normalized_think:
                        found_keywords += 1
                
                visual_reward = min(0.2, found_keywords * 0.05)
        
        total = format_reward + accuracy_reward + visual_reward
        total_rewards.append(total)
        format_rewards.append(format_reward)
        accuracy_rewards.append(accuracy_reward)
        visual_rewards.append(visual_reward)
        found_keywords_list.append(found_keywords)
        extraction_methods.append(extraction_method)

    return {
        'total_rewards': total_rewards,
        'format_rewards': format_rewards,
        'accuracy_rewards': accuracy_rewards,
        'visual_rewards': visual_rewards,
        'found_keywords': found_keywords_list,
        'extraction_methods': extraction_methods,
    }