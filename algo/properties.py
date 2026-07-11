"""
nucleok.algo.properties
=======================
Propriedades globais (Camada 4) calculadas sobre a tesselação da
fronteira pelo teorema da divergência: volume ASSINADO (positivo quando
as normais apontam para fora — também serve de checagem de orientação),
área, centroide e comprimento total de arestas.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from ..topo.entities import Solid
from .tessellate import Tessellation, tessellate


def _tess(obj: Union[Solid, Tessellation], deflection: float):
    return obj if isinstance(obj, Tessellation) else tessellate(obj,
                                                                deflection)


def signed_volume(obj, deflection: float = 0.02) -> float:
    """Volume assinado: ∑ det(p0, p1, p2)/6 sobre os triângulos."""
    t = _tess(obj, deflection)
    p0 = t.vertices[t.triangles[:, 0]]
    p1 = t.vertices[t.triangles[:, 1]]
    p2 = t.vertices[t.triangles[:, 2]]
    return float(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0)


def volume(obj, deflection: float = 0.02) -> float:
    return abs(signed_volume(obj, deflection))


def surface_area(obj, deflection: float = 0.02) -> float:
    t = _tess(obj, deflection)
    p0 = t.vertices[t.triangles[:, 0]]
    p1 = t.vertices[t.triangles[:, 1]]
    p2 = t.vertices[t.triangles[:, 2]]
    return float(np.linalg.norm(np.cross(p1 - p0, p2 - p0),
                                axis=1).sum() / 2.0)


def centroid(obj, deflection: float = 0.02) -> np.ndarray:
    t = _tess(obj, deflection)
    p0 = t.vertices[t.triangles[:, 0]]
    p1 = t.vertices[t.triangles[:, 1]]
    p2 = t.vertices[t.triangles[:, 2]]
    v6 = np.einsum("ij,ij->i", p0, np.cross(p1, p2))
    tot = v6.sum()
    if abs(tot) < 1e-14:
        return t.vertices.mean(axis=0)
    return ((p0 + p1 + p2) / 4.0 * v6[:, None]).sum(axis=0) / tot


def edge_length_total(solid: Solid) -> float:
    return float(sum(e.length() for e in solid.unique_edges()))
