"""
nucleok.topo.entities
=====================
Entidades topológicas do B-Rep (Camada 3), com separação RIGOROSA entre
geometria e topologia:

    GEOMETRIA (Camada 2)          TOPOLOGIA (esta camada)
    ------------------------      ---------------------------------------
    ponto 3D                      Vertex        (referencia o ponto)
    Curve (ilimitada/completa)    Edge          (curva + intervalo [t0,t1]
                                                 + 2 Vertex)
    Surface (ilimitada/completa)  Face          (superfície + loops de
                                                 recorte + same_sense)
                                  Loop          (ciclo de arestas
                                                 orientadas)
                                  Shell         (conjunto fechado de faces)
                                  Solid         (região limitada por
                                                 cascas)

Uma MESMA curva/superfície pode ser compartilhada por várias entidades
topológicas; a topologia diz "que pedaço" e "como conecta", nunca "onde
está" — quem diz onde é a geometria.
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..geom.curves import Curve
from ..geom.surfaces import Surface

_ids = itertools.count(1)


class TopoEntity:
    """Base: identidade topológica (id estável, comparação por identidade)."""

    def __init__(self):
        self.id: int = next(_ids)

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self is other


class Vertex(TopoEntity):
    """Ponto topológico."""

    def __init__(self, point):
        super().__init__()
        self.point = np.asarray(point, dtype=np.float64)

    def __repr__(self):
        return f"Vertex#{self.id}({self.point.round(6).tolist()})"


class Edge(TopoEntity):
    """
    Aresta: RECORTE ``[t0, t1]`` de uma curva, entre dois vértices.
    ``start``/``end`` correspondem a ``curve(t0)``/``curve(t1)``.
    Arestas fechadas (círculo completo) têm ``start is end``.
    """

    def __init__(self, curve: Curve, t0: float, t1: float,
                 start: Vertex, end: Vertex):
        super().__init__()
        self.curve = curve
        self.t0, self.t1 = float(t0), float(t1)
        self.start, self.end = start, end

    @property
    def is_closed(self) -> bool:
        return self.start is self.end

    def evaluate(self, t):
        return self.curve.evaluate(t)

    def sample(self, deflection: float, min_pts: int = 2) -> np.ndarray:
        """Polilinha da aresta (de start a end) com desvio de corda
        ≤ deflection. Círculos usam a MESMA fórmula de subdivisão da
        tesselação em grade (garante malhas estanques entre tampas
        planas e laterais curvas)."""
        from ..geom.curves import Circle
        from ..geom.surfaces import _adaptive_curve_params, _chord_divs
        if isinstance(self.curve, Circle):
            n = _chord_divs(self.curve.radius, abs(self.t1 - self.t0),
                            deflection) + 1
            ts = np.linspace(self.t0, self.t1, max(n, min_pts))
        else:
            ts = _adaptive_curve_params(self.curve, self.t0, self.t1,
                                        deflection)
            if len(ts) < min_pts:
                ts = np.linspace(self.t0, self.t1, min_pts)
        return self.curve.evaluate(ts)

    def length(self) -> float:
        return self.curve.length(self.t0, self.t1)

    def __repr__(self):
        return (f"Edge#{self.id}({type(self.curve).__name__}, "
                f"[{self.t0:.4g},{self.t1:.4g}])")


class Loop(TopoEntity):
    """Ciclo orientado de arestas: lista de ``(Edge, forward)``.
    ``forward=False`` percorre a aresta de ``end`` para ``start``."""

    def __init__(self, oriented_edges: Sequence[Tuple[Edge, bool]]):
        super().__init__()
        self.edges: List[Tuple[Edge, bool]] = list(oriented_edges)

    def vertices(self) -> List[Vertex]:
        out = []
        for e, fwd in self.edges:
            out.append(e.start if fwd else e.end)
        return out

    def is_connected(self) -> bool:
        """Cada aresta termina onde a próxima começa?"""
        n = len(self.edges)
        for i in range(n):
            e, fwd = self.edges[i]
            nxt, nfwd = self.edges[(i + 1) % n]
            head = e.end if fwd else e.start
            tail = nxt.start if nfwd else nxt.end
            if head is not tail:
                return False
        return True

    def __repr__(self):
        return f"Loop#{self.id}({len(self.edges)} arestas)"


class Face(TopoEntity):
    """
    Face: RECORTE de uma superfície pelos loops (o primeiro é a fronteira
    externa; os demais são furos/anéis).

    ``same_sense``: a normal MATERIAL da face (apontando para fora do
    sólido) coincide com a normal da superfície? ``rect_domain`` é uma
    dica de construção: quando presente, a face cobre exatamente esse
    retângulo do espaço paramétrico (patch completo) — a tesselação usa
    grade em vez de triangulação de recorte.
    """

    def __init__(self, surface: Surface, loops: Sequence[Loop],
                 same_sense: bool = True,
                 rect_domain: Optional[Tuple[Tuple[float, float],
                                             Tuple[float, float]]] = None):
        super().__init__()
        self.surface = surface
        self.loops: List[Loop] = list(loops)
        self.same_sense = bool(same_sense)
        self.rect_domain = rect_domain

    @property
    def outer_loop(self) -> Loop:
        return self.loops[0]

    def material_normal(self, u: float, v: float) -> np.ndarray:
        n = self.surface.normal(u, v)
        return n if self.same_sense else -n

    def __repr__(self):
        return (f"Face#{self.id}({type(self.surface).__name__}, "
                f"{len(self.loops)} loop(s))")


class Shell(TopoEntity):
    """Conjunto conexo de faces; ``closed=True`` = limita volume."""

    def __init__(self, faces: Sequence[Face], closed: bool = True):
        super().__init__()
        self.faces: List[Face] = list(faces)
        self.closed = bool(closed)

    def __repr__(self):
        return f"Shell#{self.id}({len(self.faces)} faces)"


class Solid(TopoEntity):
    """Sólido: casca externa + cascas internas (cavidades)."""

    def __init__(self, outer: Shell,
                 voids: Sequence[Shell] = ()):
        super().__init__()
        self.outer = outer
        self.voids: List[Shell] = list(voids)

    @property
    def shells(self) -> List[Shell]:
        return [self.outer] + self.voids

    @property
    def faces(self) -> List[Face]:
        return [f for s in self.shells for f in s.faces]

    # ------------------------------------------------------- agregações
    def unique_edges(self) -> List[Edge]:
        seen, out = set(), []
        for f in self.faces:
            for lp in f.loops:
                for e, _ in lp.edges:
                    if e.id not in seen:
                        seen.add(e.id)
                        out.append(e)
        return out

    def unique_vertices(self) -> List[Vertex]:
        seen, out = set(), []
        for e in self.unique_edges():
            for vtx in (e.start, e.end):
                if vtx.id not in seen:
                    seen.add(vtx.id)
                    out.append(vtx)
        return out

    def transformed(self, trsf) -> "Solid":
        """Cópia profundamente transformada (similaridades: translação,
        rotação, escala uniforme, espelho). Implementação em
        ``nucleok.model.ops.transform_solid``."""
        from ..model.ops import transform_solid
        return transform_solid(self, trsf)

    def __repr__(self):
        return (f"Solid#{self.id}({len(self.faces)} faces, "
                f"{len(self.unique_edges())} arestas, "
                f"{len(self.unique_vertices())} vértices)")
