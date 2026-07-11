"""
nucleok.algo.bvh
================
BVH (Bounding Volume Hierarchy) de AABBs sobre triângulos (Camada 4):
acelera consultas de raio de O(n) para ~O(log n) — usada pela
classificação ponto×sólido e disponível para picking/medições.

Construção por mediana no eixo de maior extensão dos centroides;
travessia iterativa com teste de laje (slab test) vetor-seguro.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .intersect import ray_triangle


class BVH:
    """BVH binária de triângulos: ``BVH(vertices (V,3), triangles (T,3))``."""

    __slots__ = ("v", "t", "lo", "hi", "left", "right", "tri_ids", "order")

    LEAF = 8

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        self.v = np.asarray(vertices, float)
        self.t = np.asarray(triangles, np.int64)
        tri = self.v[self.t]                       # (T, 3, 3)
        lo_all = tri.min(axis=1)
        hi_all = tri.max(axis=1)
        cent = tri.mean(axis=1)

        # nós em arrays paralelos (árvore achatada)
        self.lo: List[np.ndarray] = []
        self.hi: List[np.ndarray] = []
        self.left: List[int] = []                  # -1 = folha
        self.right: List[int] = []
        self.tri_ids: List[Optional[np.ndarray]] = []

        def build(ids: np.ndarray) -> int:
            node = len(self.lo)
            self.lo.append(lo_all[ids].min(axis=0))
            self.hi.append(hi_all[ids].max(axis=0))
            self.left.append(-1)
            self.right.append(-1)
            self.tri_ids.append(None)
            if len(ids) <= self.LEAF:
                self.tri_ids[node] = ids
                return node
            c = cent[ids]
            axis = int(np.argmax(c.max(axis=0) - c.min(axis=0)))
            order = ids[np.argsort(c[:, axis])]
            mid = len(order) // 2
            self.left[node] = build(order[:mid])
            self.right[node] = build(order[mid:])
            return node

        build(np.arange(len(self.t)))
        self.lo = np.asarray(self.lo)
        self.hi = np.asarray(self.hi)

    # ------------------------------------------------------------- raio
    def _hit_box(self, node: int, o: np.ndarray, inv_d: np.ndarray) -> bool:
        t1 = (self.lo[node] - o) * inv_d
        t2 = (self.hi[node] - o) * inv_d
        tmin = np.minimum(t1, t2).max()
        tmax = np.maximum(t1, t2).min()
        return tmax >= max(tmin, 0.0)

    def ray_hits(self, origin, direction) -> List[float]:
        """Todos os parâmetros ``t ≥ 0`` de interseção do raio com os
        triângulos (para paridade/picking)."""
        o = np.asarray(origin, float)
        d = np.asarray(direction, float)
        with np.errstate(divide="ignore"):
            inv_d = np.where(d != 0.0, 1.0 / d, np.copysign(1e300, d + 1.0))
        hits: List[float] = []
        stack = [0]
        while stack:
            node = stack.pop()
            if not self._hit_box(node, o, inv_d):
                continue
            ids = self.tri_ids[node]
            if ids is None:
                stack.append(self.left[node])
                stack.append(self.right[node])
                continue
            for k in ids:
                a, b, c = self.t[k]
                t = ray_triangle(o, d, self.v[a], self.v[b], self.v[c])
                if t is not None:
                    hits.append(t)
        return hits
