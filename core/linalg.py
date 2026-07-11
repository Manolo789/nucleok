"""
nucleok.core.linalg
===================
Vetores e utilidades de álgebra linear geométrica (Camada 1).

Convenção do kernel: pontos e vetores são ``numpy.ndarray`` float64 de
shape ``(3,)`` (ou ``(2,)`` no plano paramétrico). Estas funções são finas
por design — a robustez fica em ``predicates``; aqui, só conveniência.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

Vec = np.ndarray


def v(*coords: float) -> Vec:
    """Construtor curto de vetor/ponto float64."""
    return np.asarray(coords, dtype=np.float64)


def asvec(x: Sequence[float]) -> Vec:
    return np.asarray(x, dtype=np.float64)


def norm(a: Vec) -> float:
    return float(np.linalg.norm(a))


def unit(a: Vec) -> Vec:
    """Vetor unitário; lança em vetor nulo (erro de modelagem, não NaN)."""
    n = np.linalg.norm(a)
    if n == 0.0:
        raise ValueError("vetor nulo não pode ser normalizado")
    return a / n


def cross(a: Vec, b: Vec) -> Vec:
    return np.cross(a, b)


def dot(a: Vec, b: Vec) -> float:
    return float(np.dot(a, b))


def dist(a: Vec, b: Vec) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def angle_between(a: Vec, b: Vec) -> float:
    """Ângulo em radianos, numericamente estável (atan2 de |cross| e dot)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


def make_frame(normal: Vec) -> Tuple[Vec, Vec, Vec]:
    """
    Base ortonormal destro ``(x, y, z)`` com ``z`` na direção dada —
    determinística (mesma normal => mesmo frame), essencial para
    parametrizações reproduzíveis de planos e superfícies.
    """
    z = unit(np.asarray(normal, float))
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, helper)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    x = unit(np.cross(helper, z))
    y = np.cross(z, x)
    return x, y, z


def newell_normal(points: np.ndarray) -> Vec:
    """Normal de um polígono 3D (método de Newell — robusto a
    não-planaridade leve e vértices quase colineares)."""
    p = np.asarray(points, float)
    q = np.roll(p, -1, axis=0)
    n = np.array([
        np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])),
        np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])),
        np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])),
    ])
    return n


def polygon_area_2d(pts: np.ndarray) -> float:
    """Área ASSINADA de polígono 2D (positiva = anti-horário)."""
    p = np.asarray(pts, float)
    q = np.roll(p, -1, axis=0)
    return float(0.5 * np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]))
