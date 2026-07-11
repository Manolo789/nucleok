"""
nucleok.model.boolean
=====================
Operações booleanas (Camada 5): ``fuse`` (união), ``common`` (interseção)
e ``cut`` (diferença) — IMPLEMENTADAS, sem OCCT.

Estratégia (v0.2): **booleanas facetadas por BSP** + reconstrução B-Rep.

1. Os operandos são tesselados com deflexão controlada (superfícies
   analíticas viram facetas — precisão escolhida pelo chamador).
2. A booleana é resolvida por árvores BSP (algoritmo clássico de CSG:
   particionamento por planos, recorte mútuo e inversão), robusto para
   malhas fechadas.
3. O resultado volta a ser um :class:`~nucleok.topo.entities.Solid`
   B-Rep VÁLIDO por :func:`nucleok.model.ops.solid_from_tessellation`
   (faces planas com furos, T-vértices cicatrizados) — exportável para
   STEP/STL e utilizável por todas as camadas.

Limitação documentada: o resultado é FACETADO — as superfícies curvas
dos operandos viram conjuntos de faces planas na região tocada (e no
sólido todo). Booleanas ANALÍTICAS (curvas de interseção exatas +
imprint via operadores de Euler, plano em ``_ANALYTIC_PLAN``) continuam
no roadmap como refinamento, agora com todos os pré-requisitos prontos
(interseções, tesselação de recortes, transformações profundas).
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from ..algo.tessellate import Tessellation, tessellate
from ..topo.entities import Solid
from .ops import solid_from_tessellation

_ANALYTIC_PLAN = """
Plano das booleanas analíticas (próximo refinamento):
1. interseção face×face por par de superfícies (analítica p/ plano×
   quádrica; marching por continuação no geral) -> curvas de interseção;
2. imprint das curvas nas duas faces (operadores de Euler mev/mef);
3. classificação dos fragmentos dentro/fora (algo.classify);
4. seleção por operação e costura + validação (topo.validate).
"""


# ------------------------------------------------------------- BSP CSG
class _Poly:
    __slots__ = ("verts", "n", "w")

    def __init__(self, verts: List[np.ndarray],
                 n: Optional[np.ndarray] = None,
                 w: Optional[float] = None):
        self.verts = verts
        if n is None:
            # Newell (robusto p/ polígonos quase degenerados)
            nn = np.zeros(3)
            for i in range(len(verts)):
                a, b = verts[i], verts[(i + 1) % len(verts)]
                nn += np.cross(a, b)
            ln = np.linalg.norm(nn)
            n = nn / ln if ln > 0 else np.array([0.0, 0.0, 1.0])
            w = float(np.dot(n, verts[0]))
        self.n = n
        self.w = w

    def flip(self):
        self.verts = self.verts[::-1]
        self.n = -self.n
        self.w = -self.w

    def clone(self):
        return _Poly([v.copy() for v in self.verts], self.n.copy(),
                     self.w)


_COPLANAR, _FRONT, _BACK, _SPAN = 0, 1, 2, 3


def _split(plane_n, plane_w, poly: _Poly, eps: float,
           cofront: list, coback: list, front: list, back: list):
    types = []
    ptype = 0
    for v in poly.verts:
        t = float(np.dot(plane_n, v)) - plane_w
        ty = _COPLANAR if -eps < t < eps else (_FRONT if t > 0 else _BACK)
        ptype |= ty
        types.append(ty)
    if ptype == _COPLANAR:
        (cofront if np.dot(plane_n, poly.n) > 0 else coback).append(poly)
    elif ptype == _FRONT:
        front.append(poly)
    elif ptype == _BACK:
        back.append(poly)
    else:
        f, b = [], []
        nv = len(poly.verts)
        for i in range(nv):
            j = (i + 1) % nv
            ti, tj = types[i], types[j]
            vi, vj = poly.verts[i], poly.verts[j]
            if ti != _BACK:
                f.append(vi)
            if ti != _FRONT:
                b.append(vi.copy() if ti != _BACK else vi)
            if (ti | tj) == _SPAN:
                t = (plane_w - float(np.dot(plane_n, vi))) \
                    / float(np.dot(plane_n, vj - vi))
                v = vi + t * (vj - vi)
                f.append(v)
                b.append(v.copy())
        if len(f) >= 3:
            front.append(_Poly(f, poly.n, poly.w))
        if len(b) >= 3:
            back.append(_Poly(b, poly.n, poly.w))


class _Node:
    __slots__ = ("n", "w", "front", "back", "polys")

    def __init__(self, polys: Optional[List[_Poly]] = None,
                 eps: float = 1e-5):
        self.n = None
        self.w = 0.0
        self.front: Optional[_Node] = None
        self.back: Optional[_Node] = None
        self.polys: List[_Poly] = []
        if polys:
            self.build(polys, eps)

    def build(self, polys: List[_Poly], eps: float):
        # iterativo (árvores BSP de malhas curvas ficam profundas)
        stack = [(self, polys)]
        while stack:
            node, ps = stack.pop()
            if not ps:
                continue
            if node.n is None:
                node.n, node.w = ps[0].n, ps[0].w
            front, back = [], []
            for p in ps:
                _split(node.n, node.w, p, eps, node.polys, node.polys,
                       front, back)
            if front:
                if node.front is None:
                    node.front = _Node()
                stack.append((node.front, front))
            if back:
                if node.back is None:
                    node.back = _Node()
                stack.append((node.back, back))

    def invert(self):
        stack = [self]
        while stack:
            node = stack.pop()
            for p in node.polys:
                p.flip()
            if node.n is not None:
                node.n = -node.n
                node.w = -node.w
            node.front, node.back = node.back, node.front
            if node.front:
                stack.append(node.front)
            if node.back:
                stack.append(node.back)

    def clip_polygons(self, polys: List[_Poly], eps: float) -> List[_Poly]:
        out: List[_Poly] = []
        stack = [(self, polys)]
        while stack:
            node, ps = stack.pop()
            if node.n is None:
                out.extend(p.clone() for p in ps)
                continue
            front, back = [], []
            for p in ps:
                _split(node.n, node.w, p, eps, front, back, front, back)
            if node.front:
                stack.append((node.front, front))
            else:
                out.extend(front)
            if node.back:
                stack.append((node.back, back))
            # sem node.back: polígonos "back" estão dentro -> descartados
        return out

    def clip_to(self, other: "_Node", eps: float):
        stack = [self]
        while stack:
            node = stack.pop()
            node.polys = other.clip_polygons(node.polys, eps)
            if node.front:
                stack.append(node.front)
            if node.back:
                stack.append(node.back)

    def all_polygons(self) -> List[_Poly]:
        out: List[_Poly] = []
        stack = [self]
        while stack:
            node = stack.pop()
            out.extend(node.polys)
            if node.front:
                stack.append(node.front)
            if node.back:
                stack.append(node.back)
        return out


def _to_polys(t: Tessellation) -> List[_Poly]:
    v, f = t.vertices, t.triangles
    return [_Poly([v[a].copy(), v[b].copy(), v[c].copy()])
            for a, b, c in f]


def _to_tess(polys: List[_Poly]) -> Tessellation:
    verts, tris = [], []
    for p in polys:
        base = len(verts)
        verts.extend(p.verts)
        for k in range(1, len(p.verts) - 1):
            tris.append((base, base + k, base + k + 1))
    return Tessellation(np.asarray(verts), np.asarray(tris, np.int64))


def _prep(obj: Union[Solid, Tessellation],
          deflection: float) -> Tessellation:
    return obj if isinstance(obj, Tessellation) else tessellate(
        obj, deflection)


def _csg(a, b, op: str, deflection: float,
         rebuild: bool) -> Union[Solid, Tessellation]:
    ta, tb = _prep(a, deflection), _prep(b, deflection)
    allv = np.vstack([ta.vertices, tb.vertices])
    diag = float(np.linalg.norm(allv.max(axis=0) - allv.min(axis=0)))
    eps = max(1e-9, 1e-7 * diag)

    A = _Node(_to_polys(ta), eps)
    B = _Node(_to_polys(tb), eps)

    if op == "fuse":
        A.clip_to(B, eps)
        B.clip_to(A, eps)
        B.invert()
        B.clip_to(A, eps)
        B.invert()
        A.build(B.all_polygons(), eps)
    elif op == "cut":
        A.invert()
        A.clip_to(B, eps)
        B.clip_to(A, eps)
        B.invert()
        B.clip_to(A, eps)
        B.invert()
        A.build(B.all_polygons(), eps)
        A.invert()
    elif op == "common":
        A.invert()
        B.clip_to(A, eps)
        B.invert()
        A.clip_to(B, eps)
        B.clip_to(A, eps)
        A.build(B.all_polygons(), eps)
        A.invert()
    out = _to_tess(A.all_polygons()).weld(7)
    if not rebuild:
        return out
    return solid_from_tessellation(out)


def fuse(a, b, deflection: float = 0.05, rebuild: bool = True):
    """União ``a ∪ b``. Devolve Solid B-Rep (``rebuild=True``, padrão)
    ou a Tessellation crua (``rebuild=False``)."""
    return _csg(a, b, "fuse", deflection, rebuild)


def common(a, b, deflection: float = 0.05, rebuild: bool = True):
    """Interseção ``a ∩ b``."""
    return _csg(a, b, "common", deflection, rebuild)


def cut(a, b, deflection: float = 0.05, rebuild: bool = True):
    """Diferença ``a − b``."""
    return _csg(a, b, "cut", deflection, rebuild)
