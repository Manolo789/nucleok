"""
nucleok.model.sweep
===================
Sólidos por varredura (Camada 5): extrusão de perfis poligonais (com
furos) e revolução de perfis fechados em torno de um eixo — construindo
topologia B-Rep costurada e validável, não apenas malha.

    extrude(outer, holes, height)   prisma reto ao longo de +Z
    revolve(profile_rz)             sólido de revolução (perfil no
                                    meio-plano r–z, revolução completa)
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..core.linalg import polygon_area_2d
from ..geom.curves import Circle, Line
from ..geom.surfaces import Plane, RevolutionSurface
from ..topo.entities import Edge, Face, Loop, Shell, Solid, Vertex
from .primitives import TWO_PI, _EdgeCache

Poly2D = Sequence[Sequence[float]]


def _ccw(pts: np.ndarray) -> np.ndarray:
    return pts if polygon_area_2d(pts) > 0 else pts[::-1]


def _cw(pts: np.ndarray) -> np.ndarray:
    return pts if polygon_area_2d(pts) < 0 else pts[::-1]


# ---------------------------------------------------------------- extrusão
def extrude(outer: Poly2D, height: float,
            holes: Sequence[Poly2D] = (), z0: float = 0.0) -> Solid:
    """
    Prisma reto: extruda o polígono ``outer`` (com ``holes``) de ``z0``
    até ``z0 + height`` ao longo de +Z.

    Orientações normalizadas internamente (externo CCW, furos CW) para
    que TODAS as normais materiais apontem para fora do material — regra
    "material à esquerda do percurso".
    """
    loops2d: List[np.ndarray] = [_ccw(np.asarray(outer, float))]
    loops2d += [_cw(np.asarray(h, float)) for h in holes]

    cache = _EdgeCache()
    z1 = z0 + float(height)

    bot_rings: List[Loop] = []
    top_rings: List[Loop] = []
    side_faces: List[Face] = []

    for pts2 in loops2d:
        n = len(pts2)
        vb = [Vertex(np.array([x, y, z0])) for x, y in pts2]
        vt = [Vertex(np.array([x, y, z1])) for x, y in pts2]

        bot_oe, top_oe = [], []
        for i in range(n):
            j = (i + 1) % n
            eb, fb = cache.line_between(vb[i], vb[j])
            et, ft = cache.line_between(vt[i], vt[j])
            bot_oe.append((eb, fb))
            top_oe.append((et, ft))

            # face lateral do segmento i→j; normal = (dy, -dx, 0) —
            # "material à esquerda" vale p/ externo CCW e furo CW
            d = pts2[j] - pts2[i]
            normal = np.array([d[1], -d[0], 0.0])
            ev_i, fi = cache.line_between(vb[i], vt[i])
            ev_j, fj = cache.line_between(vb[j], vt[j])
            loop = Loop([(eb, fb), (ev_j, fj), (et, not ft),
                         (ev_i, not fi)])
            side_faces.append(
                Face(Plane(vb[i].point, normal), [loop], same_sense=True))

        # tampa inferior (normal -Z): percurso invertido
        bot_rings.append(Loop([(e, not f) for e, f in reversed(bot_oe)]))
        top_rings.append(Loop(top_oe))

    bottom = Face(Plane(np.array([0.0, 0.0, z0]), (0, 0, -1)), bot_rings,
                  same_sense=True)
    top = Face(Plane(np.array([0.0, 0.0, z1]), (0, 0, 1)), top_rings,
               same_sense=True)
    return Solid(Shell(side_faces + [bottom, top]))


# --------------------------------------------------------------- revolução
def revolve(profile_rz: Poly2D, origin=(0.0, 0.0, 0.0),
            axis=(0.0, 0.0, 1.0)) -> Solid:
    """
    Sólido de revolução COMPLETA (2π): o perfil fechado ``profile_rz``
    vive no meio-plano (r, z) com ``r > 0`` (perfil que não toca o eixo
    ⇒ topologia toroidal, χ=0; validado).

    Cada segmento do perfil vira uma face de
    :class:`~nucleok.geom.surfaces.RevolutionSurface`; vértices do perfil
    viram círculos compartilhados; a costura em u=0 é o próprio segmento.
    Perfil normalizado para CCW no plano (r, z) ⇒ normais para fora.
    """
    prof = _ccw(np.asarray(profile_rz, float))
    if np.any(prof[:, 0] <= 0):
        raise ValueError("revolve: perfil deve ter r > 0 (não pode tocar "
                         "o eixo nesta versão — cones com ápice chegam "
                         "com as booleanas)")
    o = np.asarray(origin, float)
    rs = RevolutionSurface(Line((1, 0, 0), (0, 0, 1)), o, axis)  # p/ frame
    X, ax = rs.xdir, rs.axis

    n = len(prof)
    p3d = [o + r * X + z * ax for r, z in prof]          # perfil em u=0
    verts = [Vertex(p) for p in p3d]
    circles = [Edge(Circle(o + z * ax, ax, r, xdir=X), 0.0, TWO_PI,
                    verts[i], verts[i])
               for i, (r, z) in enumerate(prof)]

    faces: List[Face] = []
    for i in range(n):
        j = (i + 1) % n
        gen = Line.through(p3d[i], p3d[j])
        length = float(np.linalg.norm(p3d[j] - p3d[i]))
        seam = Edge(gen, 0.0, length, verts[i], verts[j])
        surf = RevolutionSurface(gen, o, ax)
        loop = Loop([(circles[i], True), (seam, True),
                     (circles[j], False), (seam, False)])
        faces.append(Face(surf, [loop], same_sense=True,
                          rect_domain=((0.0, TWO_PI), (0.0, length))))
    return Solid(Shell(faces))
