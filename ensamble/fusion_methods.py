import numpy as np


def avrg_fusion(decisions: list) -> np.ndarray:
    decisions = list(map(lambda x: np.expand_dims(x, axis=2), decisions))
    decisions_array = np.concatenate(decisions, axis=2)
    return np.mean(decisions_array, axis=2)
