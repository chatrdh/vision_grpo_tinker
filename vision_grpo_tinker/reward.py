import re
import math
from fractions import Fraction
from typing import List, Dict, Any, Optional, Tuple

# Import grader functions - some may fail if sympy not available
try:
    from grader import grade, extract_answer, _normalize, are_equal_under_sympy
    GRADER_AVAILABLE = True
except ImportError:
    # Fallback if grader dependencies not installed
    GRADER_AVAILABLE = False
    def grade(model_answer, gt_answer, fast=True):
        return model_answer.strip().lower() == gt_answer.strip().lower()
    def extract_answer(passage):
        return None
    def _normalize(expr):
        return expr.strip().lower() if expr else None
    def are_equal_under_sympy(a, b):
        return False

# --- Configuration: Visual Keywords with Weights ---
# Keywords are weighted based on how strongly they indicate actual visual grounding
# Higher weight = stronger evidence the model looked at the image

VISUAL_KEYWORDS_WEIGHTED = {
    # === HIGH WEIGHT (2.0): Explicit visual observations ===
    # Colors - must be looking at the image to identify these
    "blue": 2.0, "red": 2.0, "green": 2.0, "yellow": 2.0, "orange": 2.0,
    "purple": 2.0, "pink": 2.0, "black": 1.5, "white": 1.5, "gray": 1.5, "grey": 1.5,
    
    # Visual styling - indicates looking at diagram
    "shaded": 2.0, "dotted": 2.0, "dashed": 2.0, "highlighted": 2.0,
    "marked": 1.5, "labeled": 1.5, "shown": 1.5, "drawn": 1.5,
    
    # Explicit image references
    "in the figure": 2.5, "in the diagram": 2.5, "in the image": 2.5,
    "from the figure": 2.5, "from the diagram": 2.5, "from the image": 2.5,
    "looking at": 2.0, "we can see": 2.0, "as shown": 2.0, "depicted": 2.0,
    
    # === MEDIUM WEIGHT (1.0-1.5): Geometric relationships ===
    "inscribed": 1.5, "circumscribed": 1.5, "tangent": 1.5,
    "intersect": 1.2, "intersection": 1.2, "overlap": 1.2, "enclosed": 1.2,
    "parallel": 1.0, "perpendicular": 1.0, "adjacent": 1.0, "opposite": 1.0,
    "bisect": 1.0, "midpoint": 1.0, "collinear": 1.2, "coplanar": 1.2,
    "congruent": 1.0, "similar": 1.0, "corresponding": 1.0,
    
    # Geometric attributes
    "vertex": 1.0, "vertices": 1.0, "hypotenuse": 1.0, "diagonal": 1.0,
    "radius": 1.0, "diameter": 1.0, "chord": 1.2, "secant": 1.2,
    
    # === LOW WEIGHT (0.3-0.7): Common geometric terms ===
    "triangle": 0.5, "circle": 0.5, "square": 0.5, "rectangle": 0.5,
    "polygon": 0.5, "trapezoid": 0.7, "parallelogram": 0.7,
    "angle": 0.3, "side": 0.3, "perimeter": 0.5, "area": 0.5,
}

# All keywords for backward compatibility
ALL_VISUAL_KEYWORDS = set(VISUAL_KEYWORDS_WEIGHTED.keys())


def extract_xml_tag(text: str, tag: str) -> Optional[str]:
    """
    Extracts the content inside <tag>...</tag>.
    Returns None if the tag is missing or the structure is broken.
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def normalize_numeric_answer(text: str) -> str:
    r"""
    Strip non-numeric characters from an answer string while preserving:
    - Digits (0-9)
    - Decimal points (.)
    - Fractions (/)
    - Negative signs (-)
    - Plus signs (+)
    - LaTeX elements like \sqrt, \frac, etc.
    
    This helps match answers like "45 degrees" to "45" or "$12.50" to "12.50".
    
    Args:
        text: The extracted answer string
        
    Returns:
        str: Cleaned string, or original text if no numeric content found
    """
    if not text:
        return text
    
    text = text.strip()
    
    # First, check if it looks like a LaTeX expression (keep as-is for sympy)
    if any(s in text for s in ['\\frac', '\\sqrt', 'frac{', 'sqrt{']):
        # Normalize LaTeX spacing: remove spaces around elements
        text = re.sub(r'\s+', '', text)
        # Normalize curly braces
        text = text.replace('{ ', '{').replace(' }', '}')
        return text
    
    # Remove common units and suffixes
    text = re.sub(r'\s*(degrees?|°|ft²?|cm²?|m²?|in²?|yd²?|sq\.?\s*\w+|square\s*\w+)$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*(ft|cm|m|mm|km|in|yd|mi|miles?)$', '', text, flags=re.IGNORECASE)
    
    # Keep only: digits, decimal point, fraction slash, minus, plus
    cleaned = re.sub(r'[^\d.\-+/]', '', text)
    
    # If we extracted something meaningful, return it
    # Otherwise return the original (might be a letter answer like "A" or "B")
    if cleaned and any(c.isdigit() for c in cleaned):
        return cleaned
    
    return text.strip()


def normalize_latex_answer(text: str) -> str:
    """
    Normalize LaTeX expressions for comparison.
    Handles spacing differences and common variations.
    """
    if not text:
        return text
    
    # Remove all spaces
    text = re.sub(r'\s+', '', text)
    
    # Normalize sqrt: \sqrt { 3 } -> \sqrt{3}
    text = re.sub(r'\\sqrt\s*\{\s*', r'\\sqrt{', text)
    text = re.sub(r'\s*\}', '}', text)
    
    # Normalize frac: \frac { 4 } { 11 } -> \frac{4}{11}
    text = re.sub(r'\\frac\s*\{\s*', r'\\frac{', text)
    
    # Normalize mixed numbers: 4 \frac{4}{11} -> 4+\frac{4}{11} for evaluation
    # Match: integer followed by \frac
    text = re.sub(r'^(\d+)\s*\\frac', r'\1+\\frac', text)
    
    return text


def try_parse_numeric(text: str) -> Optional[float]:
    """
    Try to parse text as a numeric value.
    Handles fractions, decimals, LaTeX fractions, and mixed numbers.
    """
    if not text:
        return None
    
    text = text.strip()
    
    try:
        # Direct float
        return float(text)
    except ValueError:
        pass
    
    # Try simple fraction (e.g., "3/4")
    if '/' in text and '\\' not in text:
        try:
            parts = text.split('/')
            if len(parts) == 2:
                return float(Fraction(int(float(parts[0])), int(float(parts[1]))))
        except (ValueError, ZeroDivisionError):
            pass
    
    # Try LaTeX fraction: \frac{48}{11} or \\frac{48}{11}
    frac_match = re.match(r'\\*frac\s*\{?\s*(\d+)\s*\}?\s*\{?\s*(\d+)\s*\}?', text)
    if frac_match:
        try:
            num = int(frac_match.group(1))
            denom = int(frac_match.group(2))
            return float(Fraction(num, denom))
        except (ValueError, ZeroDivisionError):
            pass
    
    # Try mixed number (e.g., "4 3/4")
    mixed_match = re.match(r'^(\d+)\s+(\d+)/(\d+)$', text)
    if mixed_match:
        try:
            whole = int(mixed_match.group(1))
            num = int(mixed_match.group(2))
            denom = int(mixed_match.group(3))
            return float(whole + Fraction(num, denom))
        except (ValueError, ZeroDivisionError):
            pass
    
    # Try LaTeX mixed number: "4 \frac{4}{11}" or "4\frac{4}{11}"
    latex_mixed_match = re.match(r'^(\d+)\s*\\*frac\s*\{?\s*(\d+)\s*\}?\s*\{?\s*(\d+)\s*\}?', text)
    if latex_mixed_match:
        try:
            whole = int(latex_mixed_match.group(1))
            num = int(latex_mixed_match.group(2))
            denom = int(latex_mixed_match.group(3))
            return float(whole + Fraction(num, denom))
        except (ValueError, ZeroDivisionError):
            pass
    
    return None


def are_numerically_equivalent(a: str, b: str, rel_tol: float = 1e-4) -> bool:
    """
    Check if two answers are numerically equivalent.
    Handles fractions, decimals, and simple expressions.
    """
    val_a = try_parse_numeric(a)
    val_b = try_parse_numeric(b)
    
    if val_a is not None and val_b is not None:
        return math.isclose(val_a, val_b, rel_tol=rel_tol)
    
    return False


def grade_answer_lenient(model_answer: str, ground_truth: str) -> bool:
    """
    More lenient answer grading that handles common equivalent forms.
    
    Priority:
    1. Standard grade() - uses mathd normalization + sympy
    2. Normalized comparison (strip units like ft, cm, degrees)
    3. Numeric equivalence (fraction/decimal conversion)
    4. LaTeX normalized comparison
    5. Sympy equivalence on normalized forms
    """
    if not model_answer or not ground_truth:
        return False
    
    # 1. Try standard grader first (fast mode)
    try:
        if grade(model_answer, ground_truth, fast=True):
            return True
    except Exception:
        pass
    
    # 2. Try normalized comparison (strips units like "78 ft" -> "78")
    norm_model_numeric = normalize_numeric_answer(model_answer)
    norm_truth_numeric = normalize_numeric_answer(ground_truth)
    if norm_model_numeric == norm_truth_numeric:
        return True
    
    # 3. Try numeric equivalence (handles 0.96 vs 24/25, 4 4/11 vs 48/11)
    if are_numerically_equivalent(norm_model_numeric, norm_truth_numeric):
        return True
    if are_numerically_equivalent(model_answer, ground_truth):
        return True
    
    # 4. Try LaTeX normalized comparison
    norm_model = normalize_latex_answer(model_answer)
    norm_truth = normalize_latex_answer(ground_truth)
    
    if norm_model and norm_truth and norm_model == norm_truth:
        return True
    
    # 5. Try grader _normalize and compare
    try:
        grader_norm_model = _normalize(model_answer)
        grader_norm_truth = _normalize(ground_truth)
        if grader_norm_model and grader_norm_truth and grader_norm_model == grader_norm_truth:
            return True
    except Exception:
        pass
    
    # 6. Try sympy equivalence
    try:
        if are_equal_under_sympy(norm_model, norm_truth):
            return True
    except Exception:
        pass
    
    # 7. Try slow mode with math_verify (more thorough but slower)
    try:
        if grade(model_answer, ground_truth, fast=False):
            return True
    except Exception:
        pass
    
    return False




def extract_answer_flexible(text: str) -> Tuple[Optional[str], str]:
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


def compute_visual_score(think_content: str) -> Tuple[float, int, List[str]]:
    """
    Compute visual grounding score using weighted keyword matching.
    
    Uses word boundaries and phrase matching to avoid false positives.
    Higher-weight keywords (colors, explicit image references) indicate
    stronger visual grounding than generic geometry terms.
    
    Args:
        think_content: The content from the <think> block
        
    Returns:
        tuple: (weighted_score, keyword_count, matched_keywords)
    """
    if not think_content:
        return 0.0, 0, []
    
    normalized = think_content.lower()
    total_score = 0.0
    matched = []
    
    for keyword, weight in VISUAL_KEYWORDS_WEIGHTED.items():
        # Phrases (contain spaces) - use simple substring match
        if ' ' in keyword:
            if keyword in normalized:
                total_score += weight
                matched.append(keyword)
        else:
            # Single words - use word boundary regex
            pattern = rf'\b{re.escape(keyword)}\b'
            if re.search(pattern, normalized):
                total_score += weight
                matched.append(keyword)
    
    return total_score, len(matched), matched


# =============================================================================
# VISION-SR1 "BLINDFOLDED TEST" REWARD SYSTEM
# =============================================================================
# The key insight: If a model truly understands an image, it should describe
# it well enough that it (or another model) can solve the problem without
# looking at the image again.
#
# Pass 1 (with image): Generate <scan> + <think> + <answer>
# Pass 2 (blind):      Using only question + <scan>, solve again
#
# If Pass 2 succeeds, the <scan> contained sufficient visual information.
# =============================================================================

# --- Reward Weight Configuration ---
# Vision-SR1 "Blindfolded Test" reward structure
REWARD_WEIGHTS = {
    # Format rewards
    "format_full": 0.1,            # Proper <scan> + <think> + <answer> tags
    "format_partial": 0.02,        # Only answer tag
    
    # Accuracy (Pass 1 - with image)
    "accuracy": 1.0,               # Pass 1 correct answer
    
    # Visual Grounding (Pass 2 - blind verification, only if Pass 1 correct)
    "blind_correct": 1.0,          # Blind pass correct = scan was sufficient
    "blind_wrong": -0.5,           # Blind pass wrong = scan was insufficient/hallucinated
    
    # Legacy keyword-based visual (fallback when blind pass not available)
    "visual_score_scale": 0.02,
    "visual_max": 0.20,
}
# Total possible: 1.0 (accuracy) + 0.1 (format) + 1.0 (blind) = 2.1
# Worst case for correct Pass 1: 1.0 + 0.1 - 0.5 = 0.6


def build_blind_verification_prompt(question: str, scan_content: str) -> str:
    """
    Build prompt for Pass 2 "Blindfolded Test" (no image, just question + scan).
    
    The model must solve the problem using ONLY the visual description it
    generated in Pass 1. If it can do this, the scan was sufficient.
    
    Args:
        question: The original problem question
        scan_content: The <scan> content from Pass 1
        
    Returns:
        str: The blind verification prompt (text only, no image)
    """
    return f"""<|im_start|>user
Based on this visual description of a geometry diagram, solve the problem:

=== Visual Description ===
{scan_content}
=== End Description ===

Problem: {question}

Think step by step in <think>...</think> and provide your final answer in <answer>...</answer>.<|im_end|>
<|im_start|>assistant
<think>"""


def vision_sr1_reward_fn(
    completions: List[str],
    ground_truths: List[str],
    blind_answers: Optional[List[Optional[str]]] = None,
    **kwargs
) -> List[float]:
    """
    Vision-SR1 "Blindfolded Test" Reward Function.
    
    This implements the dual-pass reward system:
    - Pass 1 reward: Standard accuracy check (was the answer correct with image?)
    - Pass 2 reward: Blind verification (can the model solve using only its scan?)
    
    Pass 2 is ONLY run for Pass 1 correct answers (cost optimization).
    
    Args:
        completions: Model outputs from Pass 1 (with image)
        ground_truths: Expected correct answers
        blind_answers: Optional Pass 2 answers. If provided, used for visual reward.
                      If None, falls back to keyword-based visual reward.
    
    Returns:
        list[float]: Reward for each completion
    """
    rewards = []
    
    for idx, (pred_text, truth_text) in enumerate(zip(completions, ground_truths)):
        score = 0.0
        
        # --- 1. Format Check ---
        scan_content = extract_xml_tag(pred_text, "scan")
        think_content = extract_xml_tag(pred_text, "think")
        answer_content = extract_xml_tag(pred_text, "answer")
        
        has_scan_close = "</scan>" in pred_text
        has_think_close = "</think>" in pred_text
        
        # Full format: <scan> + <think> + <answer> all properly closed
        if scan_content and think_content and answer_content and has_scan_close and has_think_close:
            score += REWARD_WEIGHTS["format_full"]
        elif think_content and answer_content and has_think_close:
            # Accept old format (no scan) with partial reward
            score += REWARD_WEIGHTS["format_full"] * 0.8
        elif answer_content:
            score += REWARD_WEIGHTS["format_partial"]
        else:
            # No answer found at all
            rewards.append(0.0)
            continue
        
        # --- 2. Accuracy Check (Pass 1) ---
        normalized_answer = normalize_numeric_answer(answer_content)
        is_correct = grade_answer_lenient(normalized_answer, truth_text)
        
        if is_correct:
            score += REWARD_WEIGHTS["accuracy"]
            
            # --- 3. Visual Grounding (Pass 2 or Fallback) ---
            if blind_answers is not None and idx < len(blind_answers):
                blind_answer = blind_answers[idx]
                if blind_answer is not None:
                    # We have a blind pass answer - use it for visual reward
                    normalized_blind = normalize_numeric_answer(blind_answer)
                    blind_correct = grade_answer_lenient(normalized_blind, truth_text)
                    if blind_correct:
                        score += REWARD_WEIGHTS["blind_correct"]
                    else:
                        score += REWARD_WEIGHTS["blind_wrong"]
                else:
                    # No blind answer (e.g., scan extraction failed)
                    # Use keyword-based fallback
                    content_for_visual = scan_content or think_content
                    if content_for_visual:
                        visual_score, _, _ = compute_visual_score(content_for_visual)
                        visual_bonus = min(REWARD_WEIGHTS["visual_max"], 
                                          visual_score * REWARD_WEIGHTS["visual_score_scale"])
                        score += visual_bonus
            else:
                # No blind_answers provided - use keyword-based fallback
                content_for_visual = scan_content or think_content
                if content_for_visual:
                    visual_score, _, _ = compute_visual_score(content_for_visual)
                    visual_bonus = min(REWARD_WEIGHTS["visual_max"], 
                                      visual_score * REWARD_WEIGHTS["visual_score_scale"])
                    score += visual_bonus
        
        rewards.append(score)
    
    return rewards


def vision_sr1_reward_fn_detailed(
    completions: List[str],
    ground_truths: List[str],
    blind_answers: Optional[List[Optional[str]]] = None,
    lenient: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Vision-SR1 reward function with detailed breakdown.
    
    Returns detailed metrics for logging and analysis.
    
    Args:
        completions: Model outputs from Pass 1
        ground_truths: Expected correct answers
        blind_answers: Optional Pass 2 answers for blind verification
        lenient: If True, try flexible answer extraction when tags are missing
    
    Returns:
        dict with:
            'total_rewards': list[float]
            'format_rewards': list[float]
            'accuracy_rewards': list[float]
            'visual_rewards': list[float]
            'blind_results': list[str] - 'correct', 'wrong', 'skipped', 'no_scan'
            'extraction_methods': list[str]
            'has_scan': list[bool]
    """
    total_rewards = []
    format_rewards = []
    accuracy_rewards = []
    visual_rewards = []
    blind_results = []
    extraction_methods = []
    has_scan_list = []
    extracted_answers = []  # Raw extracted answers
    normalized_answers = []  # Normalized answers for grading
    
    for idx, (pred_text, truth_text) in enumerate(zip(completions, ground_truths)):
        format_reward = 0.0
        accuracy_reward = 0.0
        visual_reward = 0.0
        extraction_method = "none"
        blind_result = "skipped"
        
        # --- 1. Format Check ---
        scan_content = extract_xml_tag(pred_text, "scan")
        think_content = extract_xml_tag(pred_text, "think")
        answer_content = extract_xml_tag(pred_text, "answer")
        
        has_scan = scan_content is not None and "</scan>" in pred_text
        has_scan_list.append(has_scan)
        
        has_think_close = "</think>" in pred_text
        
        if has_scan and think_content and answer_content and has_think_close:
            format_reward = REWARD_WEIGHTS["format_full"]
            extraction_method = "answer_tag"
        elif think_content and answer_content and has_think_close:
            format_reward = REWARD_WEIGHTS["format_full"] * 0.8
            extraction_method = "answer_tag"
        elif answer_content:
            format_reward = REWARD_WEIGHTS["format_partial"]
            extraction_method = "answer_tag"
        elif lenient:
            answer_content, extraction_method = extract_answer_flexible(pred_text)
            if "<think>" in pred_text:
                format_reward = 0.02
                think_start = pred_text.find("<think>") + len("<think>")
                think_content = pred_text[think_start:]
        
        # Skip if no answer
        if not answer_content:
            total_rewards.append(0.0)
            format_rewards.append(0.0)
            accuracy_rewards.append(0.0)
            visual_rewards.append(0.0)
            blind_results.append("no_answer")
            extraction_methods.append("none")
            extracted_answers.append(None)
            normalized_answers.append(None)
            continue
        
        # --- 2. Accuracy Check ---
        try:
            # Normalize answer to strip non-numeric characters (e.g., "45 degrees" -> "45")
            normalized_answer = normalize_numeric_answer(answer_content)
            is_correct = grade_answer_lenient(normalized_answer, truth_text)
        except Exception:
            normalized_answer = answer_content
            is_correct = False
        
        extracted_answers.append(answer_content)
        normalized_answers.append(normalized_answer)
        
        if is_correct:
            accuracy_reward = REWARD_WEIGHTS["accuracy"]
            
            # --- 3. Visual Grounding ---
            if blind_answers is not None and idx < len(blind_answers):
                blind_answer = blind_answers[idx]
                
                if not has_scan:
                    blind_result = "no_scan"
                    # Keyword fallback
                    if think_content:
                        vs, _, _ = compute_visual_score(think_content)
                        visual_reward = min(REWARD_WEIGHTS["visual_max"], 
                                           vs * REWARD_WEIGHTS["visual_score_scale"])
                elif blind_answer is not None:
                    try:
                        normalized_blind = normalize_numeric_answer(blind_answer)
                        blind_correct = grade_answer_lenient(normalized_blind, truth_text)
                        if blind_correct:
                            visual_reward = REWARD_WEIGHTS["blind_correct"]
                            blind_result = "correct"
                        else:
                            visual_reward = REWARD_WEIGHTS["blind_wrong"]
                            blind_result = "wrong"
                    except Exception:
                        blind_result = "error"
                else:
                    blind_result = "no_blind_answer"
            else:
                blind_result = "not_provided"
                # Keyword fallback
                content_for_visual = scan_content or think_content
                if content_for_visual:
                    vs, _, _ = compute_visual_score(content_for_visual)
                    visual_reward = min(REWARD_WEIGHTS["visual_max"], 
                                       vs * REWARD_WEIGHTS["visual_score_scale"])
        
        total = format_reward + accuracy_reward + visual_reward
        total_rewards.append(total)
        format_rewards.append(format_reward)
        accuracy_rewards.append(accuracy_reward)
        visual_rewards.append(visual_reward)
        blind_results.append(blind_result)
        extraction_methods.append(extraction_method)
    
    return {
        'total_rewards': total_rewards,
        'format_rewards': format_rewards,
        'accuracy_rewards': accuracy_rewards,
        'visual_rewards': visual_rewards,
        'blind_results': blind_results,
        'extraction_methods': extraction_methods,
        'has_scan': has_scan_list,
        'extracted_answers': extracted_answers,
        'normalized_answers': normalized_answers,
    }


# =============================================================================
# LEGACY REWARD FUNCTIONS (for backward compatibility)
# =============================================================================

def visual_cot_reward_fn(completions, ground_truths, **kwargs):
    """
    Legacy reward function - calls Vision-SR1 without blind answers (keyword fallback).
    """
    return vision_sr1_reward_fn(completions, ground_truths, blind_answers=None, **kwargs)


def visual_cot_reward_fn_detailed(completions, ground_truths, lenient=True, **kwargs):
    """
    Legacy detailed reward function - calls Vision-SR1 without blind answers.
    """
    return vision_sr1_reward_fn_detailed(
        completions, ground_truths, 
        blind_answers=None, 
        lenient=lenient, 
        **kwargs
    )