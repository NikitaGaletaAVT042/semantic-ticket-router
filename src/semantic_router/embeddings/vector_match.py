from __future__ import annotations

import numpy as np


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)

    if matrix.ndim == 1:
        norm = np.linalg.norm(matrix)
        return matrix / max(norm, eps)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)

    if query.ndim != 1:
        raise ValueError("query must be 1D vector")

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")

    query_norm = l2_normalize(query)
    matrix_norm = l2_normalize(matrix)

    return matrix_norm @ query_norm