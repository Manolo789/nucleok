"""
nucleok.algo.tessellate
=======================
Tesselação (Camada 4): converte faces B-Rep em triângulos com controle de
DEFLEXÃO DE CORDA (desvio máximo entre a malha e a superfície exata).

Duas rotas, escolhidas pela natureza da face:
- **Patch retangular** (``face.rect_domain`` presente — laterais de
  cilindros, esferas, toros, superfícies de revolução): grade paramétrica
  com subdivisões dadas por ``surface.divisions``.
- **Face plana recortada**: loops amostrados no espaço (u, v) do plano e
  triangulados por *ear clipping* com furos (``triangulate``).

Os triângulos saem orientados com a normal MATERIAL da face (respeitando
``same_sense``) — pré-requisito para volume assinado e STL correto.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..core.linalg import polygon_area_2d
from ..geom.surfaces import Plane
from ..topo.entities import Face, Solid
from .triangulate import triangulate_polygon


class Tessellation:
    """Malha triangular indexada: ``vertices (V,3)``, ``triangles (T,3)``."""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        self.vertices = np.asarray(vertices, np.float64).reshape(-1, 3)
        self.triangles = np.asarray(triangles, np.int64).reshape(-1, 3)

    @classmethod
    def merge(cls, parts: List["Tessellation"]) -> "Tessellation":
        vs, ts, off = [], [], 0
        for p in parts:
            vs.append(p.vertices)
            ts.append(p.triangles + off)
            off += len(p.vertices)
        if not vs:
            return cls(np.zeros((0, 3)), np.zeros((0, 3), np.int64))
        return cls(np.vstack(vs), np.vstack(ts))

    def weld(self, decimals: int = 9) -> "Tessellation":
        """Solda vértices coincidentes (importante após merge de faces)."""
        uniq, inv = np.unique(self.vertices.round(decimals), axis=0,
                              return_inverse=True)
        tris = inv[self.triangles]
        # remove triângulos degenerados criados pela solda
        keep = ((tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2])
                & (tris[:, 2] != tris[:, 0]))
        return Tessellation(uniq, tris[keep])

    def face_normals(self) -> np.ndarray:
        v = self.vertices
        t = self.triangles
        n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln[ln == 0] = 1.0
        return n / ln


# ------------------------------------------------------------------- faces
def _tessellate_grid_face(face: Face, deflection: float) -> Tessellation:
    (u0, u1), (v0, v1) = face.rect_domain
    nu, nv = face.surface.divisions(face.rect_domain, deflection)
    us = np.linspace(u0, u1, nu + 1)
    vs = np.linspace(v0, v1, nv + 1)
    UU, VV = np.meshgrid(us, vs, indexing="ij")
    pts = face.surface.evaluate(UU.ravel(), VV.ravel())

    tris = []
    for i in range(nu):
        for j in range(nv):
            a = i * (nv + 1) + j
            b = (i + 1) * (nv + 1) + j
            tris.append((a, b, b + 1))
            tris.append((a, b + 1, a + 1))
    tris = np.asarray(tris, np.int64)

    # remove degenerados (polos de esfera etc.)
    v = pts
    n = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]])
    tris = tris[np.linalg.norm(n, axis=1) > 1e-14]

    if not face.same_sense:                       # inverte orientação
        tris = tris[:, ::-1]
    return Tessellation(pts, tris)


def _loop_to_uv(face: Face, loop, deflection: float) -> np.ndarray:
    """Amostra o loop como polilinha FECHADA no (u, v) do plano da face."""
    plane: Plane = face.surface
    uv: List[np.ndarray] = []
    for e, fwd in loop.edges:
        pts = e.sample(deflection)
        if not fwd:
            pts = pts[::-1]
        for p in pts[:-1]:                        # evita duplicar juntas
            uv.append(np.asarray(plane.parameters_of(p)))
    return np.asarray(uv)


def _tessellate_planar_face(face: Face, deflection: float) -> Tessellation:
    plane: Plane = face.surface
    loops_uv = [_loop_to_uv(face, lp, deflection) for lp in face.loops]

    # o loop EXTERNO é o de maior área — a ordem dos loops na face pode
    # vir de qualquer construtor
    areas = [abs(polygon_area_2d(l)) for l in loops_uv]
    k = int(np.argmax(areas))
    outer = loops_uv[k]
    holes = [l for i, l in enumerate(loops_uv) if i != k]

    uv, tris = triangulate_polygon(outer, holes)
    pts = plane.evaluate(uv[:, 0], uv[:, 1])
    # triangulate_polygon devolve CCW no plano ⇒ normal 3D = normal do
    # plano; inverte se a normal material for oposta
    if not face.same_sense:
        tris = tris[:, ::-1]
    return Tessellation(pts, tris)


def tessellate_face(face: Face, deflection: float = 0.1) -> Tessellation:
    if face.rect_domain is not None:
        return _tessellate_grid_face(face, deflection)
    if isinstance(face.surface, Plane):
        return _tessellate_planar_face(face, deflection)
    raise NotImplementedError(
        f"tesselação de face recortada sobre {type(face.surface).__name__} "
        "— recortes genéricos em superfícies curvas chegam com as "
        "booleanas (Camada 5)")


def tessellate(solid: Solid, deflection: float = 0.1) -> Tessellation:
    """Tessela o sólido inteiro (faces fundidas e soldadas)."""
    parts = [tessellate_face(f, deflection) for f in solid.faces]
    return Tessellation.merge(parts).weld()
