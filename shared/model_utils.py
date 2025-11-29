# shared/model_utils.py
import math
import random
from typing import Dict, List, Tuple

def init_model(dim: int = 4) -> List[float]:
    return [0.0 for _ in range(dim)]

def average_models(models: List[List[float]]) -> List[float]:
    if not models:
        raise ValueError("No models provided for averaging")
    dim = len(models[0])
    avg = [0.0] * dim
    for m in models:
        for i, v in enumerate(m):
            avg[i] += v
    return [v / len(models) for v in avg]

def l2_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))