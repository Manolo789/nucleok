"""
nucleok.algo.classify
=====================
Classificação ponto×sólido (Camada 4) por paridade de cruzamentos de raio
contra a tesselação da fronteira, com direção de raio "irracional" para
evitar degenerescências (raio passando por arestas/vértices).
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from ..topo.entities import Solid
from .intersect import ray_triangle
from .tessellate import Tessellation, tessellate


class Location(Enum):
    INSIDE = "inside"
    OUTSIDE = "outside"
    ON_BOUNDARY = "on_boundary"


_RAY_DIR = np.array([0.577350269189626, 0.211324865405187,
                     0.788675134594813])
_RAY_DIR = _RAY_DIR / np.linalg.norm(_RAY_DIR)


def classify_point(solid_or_tess, point, deflection: float = 0.05,
                   boundary_tol: float = 1e-6) -> Location:
    """Classifica o ponto em relação ao sólido (via tesselação)."""
    if isinstance(solid_or_tess, Tessellation):
        tess = solid_or_tess
    else:
        tess = tessellate(solid_or_tess, deflection)
    p = np.asarray(point, float)
    v = tess.vertices
    t = tess.triangles

    count = 0
    for tri in t:
        hit = ray_triangle(p, _RAY_DIR, v[tri[0]], v[tri[1]], v[tri[2]])
        if hit is None:
            continue
        if hit < boundary_tol:
            return Location.ON_BOUNDARY
        count += 1
    return Location.INSIDE if count % 2 == 1 else Location.OUTSIDE


def is_inside(solid: Solid, point, deflection: float = 0.05) -> bool:
    return classify_point(solid, point, deflection) is Location.INSIDE
