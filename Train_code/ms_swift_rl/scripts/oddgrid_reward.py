import re
from typing import List, Optional, Sequence, Set

from swift.rewards import ORM, orms


def extract_boxed_content(text: str) -> Optional[str]:
    matches = list(re.finditer(r"\\boxed\{", text or ""))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    content = []
    for char in text[start:]:
        if char == "{":
            depth += 1
            content.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(content).strip()
            content.append(char)
        else:
            content.append(char)
    return None


def parse_to_set(text: str) -> Set[str]:
    if text is None:
        return set()
    cleaned = str(text).strip()
    if not cleaned:
        return set()

    coord_matches = re.findall(r"\((\d+)\s*,\s*(\d+)\)", cleaned)
    if coord_matches:
        return {f"({row},{col})" for row, col in coord_matches}

    image_matches = re.findall(r"image\s*(\d+)", cleaned, flags=re.IGNORECASE)
    if image_matches:
        return {f"image{idx}" for idx in image_matches}

    parts = [part.strip() for part in re.split(r"[,;\n]+", cleaned) if part.strip()]
    return set(parts)


def em_reward(pred: str, gold: str) -> float:
    boxed = extract_boxed_content(pred)
    if boxed is None:
        return 0.0
    return 1.0 if parse_to_set(boxed) == parse_to_set(gold) else 0.0


def f1_reward(pred: str, gold: str) -> float:
    boxed = extract_boxed_content(pred)
    if boxed is None:
        return 0.0
    pred_set = parse_to_set(boxed)
    gold_set = parse_to_set(gold)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    true_positive = len(pred_set & gold_set)
    precision = true_positive / len(pred_set)
    recall = true_positive / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def format_reward(pred: str) -> float:
    return 1.0 if extract_boxed_content(pred) is not None else 0.0


def combine_with_format(score: float, pred: str, format_weight: float = 0.2) -> float:
    return score * (1.0 - format_weight) + format_reward(pred) * format_weight


def _normalize_solution(solution: Optional[Sequence[str]], completions: Sequence[str]) -> List[str]:
    if solution is None:
        return [""] * len(completions)
    if isinstance(solution, str):
        return [solution] * len(completions)
    return ["" if item is None else str(item) for item in solution]


class OddGridEM(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        solutions = _normalize_solution(solution, completions)
        return [combine_with_format(em_reward(pred, gold), pred) for pred, gold in zip(completions, solutions)]


class OddGridF1(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        solutions = _normalize_solution(solution, completions)
        return [combine_with_format(f1_reward(pred, gold), pred) for pred, gold in zip(completions, solutions)]


orms["oddgrid_em"] = OddGridEM
orms["oddgrid_f1"] = OddGridF1
