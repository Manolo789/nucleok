"""
nucleok.topo.euler
==================
Operadores de Euler (Camada 4) sobre uma estrutura half-edge dedicada.

Os operadores de Euler são as "instruções atômicas" de edição topológica:
cada um altera V, E, F, S (e anéis/gênero) preservando a fórmula de
Euler–Poincaré ``V − E + F = 2(S − H) + R``. Qualquer B-Rep válido pode
ser construído por uma sequência deles — o teste clássico (reproduzido em
``tests``) constrói um tetraedro com ``mvfs`` + 3×``mev`` + 3×``mef``.

Implementados (com inversos como próximos passos do roadmap):
    mvfs(p)          make vertex, face, solid   (V+1, F+1, S+1)
    mev(he|vertex)   make edge, vertex          (V+1, E+1)
    mef(he1, he2)    make edge, face            (E+1, F+1)

A conversão :func:`to_solid` produz as entidades B-Rep da Camada 3 com
faces planas — a ponte entre modelagem topológica e o resto do kernel.
"""

from __future__ import annotations

import itertools
from typing import List, Optional

import numpy as np

_ids = itertools.count(1)


class HVertex:
    def __init__(self, point):
        self.id = next(_ids)
        self.point = np.asarray(point, float)

    def __repr__(self):
        return f"v{self.id}"


class HalfEdge:
    __slots__ = ("id", "origin", "twin", "next", "prev", "loop")

    def __init__(self, origin: HVertex):
        self.id = next(_ids)
        self.origin = origin
        self.twin: Optional["HalfEdge"] = None
        self.next: Optional["HalfEdge"] = None
        self.prev: Optional["HalfEdge"] = None
        self.loop: Optional["HLoop"] = None

    def __repr__(self):
        d = self.twin.origin if self.twin else None
        return f"he{self.id}({self.origin}->{d})"


class HLoop:
    def __init__(self, face: "HFace"):
        self.id = next(_ids)
        self.face = face
        self.he: Optional[HalfEdge] = None       # None = loop vazio (só mvfs)
        self.anchor: Optional[HVertex] = None    # vértice do loop vazio

    def halfedges(self) -> List[HalfEdge]:
        out = []
        if self.he is None:
            return out
        h = self.he
        while True:
            out.append(h)
            h = h.next
            if h is self.he:
                break
        return out

    def vertices(self) -> List[HVertex]:
        if self.he is None:
            return [self.anchor] if self.anchor else []
        return [h.origin for h in self.halfedges()]


class HFace:
    def __init__(self, model: "HEModel"):
        self.id = next(_ids)
        self.model = model
        self.loops: List[HLoop] = []

    @property
    def outer(self) -> HLoop:
        return self.loops[0]


class HEModel:
    """Modelo half-edge manipulado exclusivamente pelos operadores de
    Euler — invariantes topológicos garantidos por construção."""

    def __init__(self):
        self.vertices: List[HVertex] = []
        self.faces: List[HFace] = []

    # ------------------------------------------------------------ censos
    @property
    def V(self) -> int:
        return len(self.vertices)

    @property
    def E(self) -> int:
        seen = set()
        for f in self.faces:
            for lp in f.loops:
                for h in lp.halfedges():
                    seen.add(frozenset((h.id, h.twin.id)))
        return len(seen)

    @property
    def F(self) -> int:
        return len(self.faces)

    def euler_characteristic(self) -> int:
        return self.V - self.E + self.F

    # ------------------------------------------------------------- mvfs
    def mvfs(self, point) -> tuple:
        """Make Vertex, Face, Solid: cria o sólido mínimo — um vértice e
        uma face cujo loop (vazio) ancora nele. χ = 1 − 0 + 1 = 2. ✓"""
        v = HVertex(point)
        f = HFace(self)
        lp = HLoop(f)
        lp.anchor = v
        f.loops.append(lp)
        self.vertices.append(v)
        self.faces.append(f)
        return v, f

    # -------------------------------------------------------------- mev
    def mev(self, at: HVertex, point, loop: HLoop) -> HVertex:
        """Make Edge, Vertex: brota uma aresta-espora de ``at`` (dentro de
        ``loop``) até um vértice novo. V+1, E+1 ⇒ χ inalterado. ✓"""
        w = HVertex(point)
        self.vertices.append(w)
        h1 = HalfEdge(at)      # at -> w
        h2 = HalfEdge(w)       # w  -> at
        h1.twin, h2.twin = h2, h1
        h1.loop = h2.loop = loop

        if loop.he is None:                       # loop vazio (pós-mvfs)
            if loop.anchor is not at:
                raise ValueError("mev: vértice não pertence ao loop vazio")
            h1.next, h1.prev = h2, h2
            h2.next, h2.prev = h1, h1
            loop.he = h1
            loop.anchor = None
            return w

        # insere antes da primeira half-edge que SAI de `at` neste loop
        target = None
        for h in loop.halfedges():
            if h.origin is at:
                target = h
                break
        if target is None:
            raise ValueError("mev: vértice não pertence ao loop")
        p = target.prev
        p.next, h1.prev = h1, p
        h1.next, h2.prev = h2, h1
        h2.next, target.prev = target, h2
        return w

    # -------------------------------------------------------------- mef
    def mef(self, he_from: HalfEdge, he_to: HalfEdge) -> HFace:
        """Make Edge, Face: liga ``he_from.origin`` a ``he_to.origin``
        (mesmo loop), dividindo-o em dois — o novo recebe face nova.
        E+1, F+1 ⇒ χ inalterado. ✓"""
        if he_from.loop is not he_to.loop:
            raise ValueError("mef: half-edges devem estar no mesmo loop")
        loop1 = he_from.loop
        A, B = he_from.origin, he_to.origin
        n1 = HalfEdge(A)      # A -> B, fica no loop original
        n2 = HalfEdge(B)      # B -> A, vai para o loop novo
        n1.twin, n2.twin = n2, n1

        pa, pb = he_from.prev, he_to.prev
        # loop 1 (mantém a face): ... pa -> n1 -> he_to ...
        pa.next, n1.prev = n1, pa
        n1.next, he_to.prev = he_to, n1
        # loop 2 (face nova): ... pb -> n2 -> he_from ...
        pb.next, n2.prev = n2, pb
        n2.next, he_from.prev = he_from, n2

        loop1.he = n1
        f2 = HFace(self)
        loop2 = HLoop(f2)
        loop2.he = n2
        f2.loops.append(loop2)
        self.faces.append(f2)
        for h in loop1.halfedges():
            h.loop = loop1
        for h in loop2.halfedges():
            h.loop = loop2
        return f2

    # -------------------------------------------------- ponte p/ Camada 3
    def to_solid(self):
        """Converte o modelo (faces planas poligonais) para as entidades
        B-Rep da Camada 3, compartilhando arestas e vértices."""
        from ..geom.curves import Line
        from ..geom.surfaces import Plane
        from ..core.linalg import newell_normal, unit
        from .entities import Edge, Face, Loop, Shell, Solid, Vertex

        vmap = {v.id: Vertex(v.point) for v in self.vertices}
        emap = {}

        def get_edge(a: HVertex, b: HVertex):
            key = frozenset((a.id, b.id))
            if key not in emap:
                ln = Line.through(a.point, b.point)
                length = float(np.linalg.norm(b.point - a.point))
                emap[key] = (Edge(ln, 0.0, length, vmap[a.id], vmap[b.id]),
                             a.id)
            edge, start_id = emap[key]
            return edge, (start_id == a.id)

        faces = []
        for f in self.faces:
            pts = np.array([v.point for v in f.outer.vertices()])
            n = unit(newell_normal(pts))
            plane = Plane(pts[0], n)
            loops = []
            for lp in f.loops:
                oes = []
                for h in lp.halfedges():
                    e, fwd = get_edge(h.origin, h.twin.origin)
                    oes.append((e, fwd))
                loops.append(Loop(oes))
            faces.append(Face(plane, loops, same_sense=True))
        return Solid(Shell(faces))
