"""
nucleok.model.primitives
========================
Primitivas sólidas (Camada 5), construídas como B-Rep COMPLETO: geometria
analítica da Camada 2 + topologia costurada da Camada 3, com arestas e
vértices COMPARTILHADOS entre faces vizinhas (condição verificada por
``topo.validate``).

Topologias de referência (χ = V − E + F − R):
    caixa     V=8  E=12 F=6            χ=2  (esfera topológica)
    cilindro  V=2  E=3  F=3            χ=2  (costura lateral + 2 tampas)
    esfera    V=2  E=1  F=1            χ=2  (costura meridiana)
    toro      V=1  E=2  F=1            χ=0  (gênero 1)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..core.linalg import unit
from ..geom.curves import Circle, Line
from ..geom.surfaces import (CylindricalSurface, Plane, SphericalSurface,
                             ToroidalSurface)
from ..topo.entities import Edge, Face, Loop, Shell, Solid, Vertex

TWO_PI = 2.0 * np.pi


# ------------------------------------------------------------- utilidades
class _EdgeCache:
    """Compartilhamento de arestas retas entre faces (chave: par de
    vértices)."""

    def __init__(self):
        self._map: Dict[frozenset, Edge] = {}

    def line_between(self, a: Vertex, b: Vertex) -> Tuple[Edge, bool]:
        key = frozenset((a.id, b.id))
        if key not in self._map:
            length = float(np.linalg.norm(b.point - a.point))
            self._map[key] = Edge(Line.through(a.point, b.point),
                                  0.0, length, a, b)
        e = self._map[key]
        return e, (e.start is a)


def planar_face(vertices: Sequence[Vertex], normal,
                cache: _EdgeCache) -> Face:
    """Face plana pelo ciclo de vértices, com a normal MATERIAL dada.
    O ciclo deve ser anti-horário visto do lado da normal."""
    n = unit(np.asarray(normal, float))
    plane = Plane(vertices[0].point, n)
    oes = []
    for i in range(len(vertices)):
        e, fwd = cache.line_between(vertices[i],
                                    vertices[(i + 1) % len(vertices)])
        oes.append((e, fwd))
    return Face(plane, [Loop(oes)], same_sense=True)


# ------------------------------------------------------------------ caixa
def make_box(dx: float, dy: float, dz: float,
             origin=(0.0, 0.0, 0.0)) -> Solid:
    """Paralelepípedo ``[0,dx]×[0,dy]×[0,dz]`` transladado por
    ``origin``."""
    o = np.asarray(origin, float)
    P = [o + np.array(c, float) for c in [
        (0, 0, 0), (dx, 0, 0), (dx, dy, 0), (0, dy, 0),
        (0, 0, dz), (dx, 0, dz), (dx, dy, dz), (0, dy, dz)]]
    V = [Vertex(p) for p in P]
    cache = _EdgeCache()
    faces = [
        planar_face([V[0], V[3], V[2], V[1]], (0, 0, -1), cache),  # fundo
        planar_face([V[4], V[5], V[6], V[7]], (0, 0, +1), cache),  # topo
        planar_face([V[0], V[1], V[5], V[4]], (0, -1, 0), cache),
        planar_face([V[2], V[3], V[7], V[6]], (0, +1, 0), cache),
        planar_face([V[1], V[2], V[6], V[5]], (+1, 0, 0), cache),
        planar_face([V[3], V[0], V[4], V[7]], (-1, 0, 0), cache),
    ]
    return Solid(Shell(faces))


# --------------------------------------------------------------- cilindro
def make_cylinder(radius: float, height: float,
                  origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)) -> Solid:
    """Cilindro sólido de raio ``radius`` e altura ``height`` a partir de
    ``origin`` ao longo de ``axis``."""
    surf = CylindricalSurface(origin, axis, radius)
    o = surf.origin
    ax, X = surf.axis, surf.xdir

    pa = o + radius * X                     # (r, 0, 0) local
    pb = pa + height * ax
    va, vb = Vertex(pa), Vertex(pb)

    seam = Edge(Line(pa, ax), 0.0, height, va, vb)
    circ_b = Circle(o, ax, radius, xdir=X)
    circ_t = Circle(o + height * ax, ax, radius, xdir=X)
    eb = Edge(circ_b, 0.0, TWO_PI, va, va)
    et = Edge(circ_t, 0.0, TWO_PI, vb, vb)

    side = Face(surf,
                [Loop([(eb, True), (seam, True), (et, False),
                       (seam, False)])],
                same_sense=True,
                rect_domain=((0.0, TWO_PI), (0.0, height)))
    bottom = Face(Plane(o, -ax), [Loop([(eb, False)])], same_sense=True)
    top = Face(Plane(o + height * ax, ax), [Loop([(et, True)])],
               same_sense=True)
    return Solid(Shell([side, bottom, top]))


# ----------------------------------------------------------------- esfera
def make_sphere(radius: float, center=(0.0, 0.0, 0.0)) -> Solid:
    """Esfera sólida: uma face esférica com costura meridiana entre os
    polos (V=2, E=1, F=1, χ=2)."""
    surf = SphericalSurface(center, radius)
    c = surf.center
    south = Vertex(c - radius * surf.axis)
    north = Vertex(c + radius * surf.axis)
    # meridiano u=0: semicírculo no plano (xdir, axis)
    meridian = Circle(c, np.cross(surf.xdir, surf.axis) * -1.0, radius,
                      xdir=-surf.axis)
    seam = Edge(meridian, 0.0, np.pi, south, north)
    face = Face(surf, [Loop([(seam, True), (seam, False)])],
                same_sense=True,
                rect_domain=((0.0, TWO_PI), (-np.pi / 2, np.pi / 2)))
    return Solid(Shell([face]))


# ------------------------------------------------------------------- toro
def make_torus(major_radius: float, minor_radius: float,
               center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)) -> Solid:
    """Toro sólido: uma face com duas costuras (V=1, E=2, F=1, χ=0)."""
    surf = ToroidalSurface(center, axis, major_radius, minor_radius)
    c, ax, X = surf.center, surf.axis, surf.xdir
    p0 = c + (major_radius + minor_radius) * X       # (u=0, v=0)
    v0 = Vertex(p0)
    # costura u=0: círculo do tubo no plano (X, axis)
    tube = Circle(c + major_radius * X, np.cross(ax, X) * -1.0,
                  minor_radius, xdir=X)
    seam_u = Edge(tube, 0.0, TWO_PI, v0, v0)
    # costura v=0: círculo equatorial externo
    equator = Circle(c, ax, major_radius + minor_radius, xdir=X)
    seam_v = Edge(equator, 0.0, TWO_PI, v0, v0)
    face = Face(surf,
                [Loop([(seam_v, True), (seam_u, True),
                       (seam_v, False), (seam_u, False)])],
                same_sense=True,
                rect_domain=((0.0, TWO_PI), (0.0, TWO_PI)))
    return Solid(Shell([face]))
