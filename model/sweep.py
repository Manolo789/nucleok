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
            axis=(0.0, 0.0, 1.0), angle: float = TWO_PI) -> Solid:
    """
    Sólido de revolução do perfil fechado ``profile_rz`` (meio-plano
    r–z, ``r ≥ 0``) em torno do eixo, por ``angle`` radianos
    (``2π`` = volta completa).

    - Vértices com ``r = 0`` (perfil TOCANDO o eixo) são suportados:
      viram um único vértice-ápice sem círculo (cones, discos até o
      centro); segmentos inteiramente sobre o eixo são omitidos.
    - ``angle < 2π``: revolução PARCIAL — o sólido ganha duas tampas
      planas (o próprio perfil em ``u = 0`` e ``u = angle``).

    Cada segmento fora do eixo vira uma face de
    :class:`~nucleok.geom.surfaces.RevolutionSurface` com
    ``rect_domain = ((0, angle), (0, comprimento))``.
    """
    prof = _ccw(np.asarray(profile_rz, float))
    if np.any(prof[:, 0] < -1e-12):
        raise ValueError("revolve: perfil deve ter r >= 0")
    prof[:, 0] = np.clip(prof[:, 0], 0.0, None)
    angle = float(angle)
    full = abs(angle - TWO_PI) < 1e-12
    if not (0.0 < angle <= TWO_PI + 1e-12):
        raise ValueError("revolve: 0 < angle <= 2π")

    o = np.asarray(origin, float)
    frame = RevolutionSurface(Line((1, 0, 0), (0, 0, 1)), o, axis)
    X, Y, ax = frame.xdir, frame.ydir, frame.axis
    ca, sa = np.cos(angle), np.sin(angle)
    Xa = ca * X + sa * Y                       # xdir girado até `angle`

    n = len(prof)
    on_axis = prof[:, 0] < 1e-12

    def pt3(i, at_end: bool):
        r, z = prof[i]
        d = Xa if (at_end and not full) else X
        return o + r * d + z * ax

    # vértices: em u=0 e (se parcial) em u=angle; r=0 -> vértice único
    v0 = [Vertex(pt3(i, False)) for i in range(n)]
    if full:
        v1 = v0
    else:
        v1 = [v0[i] if on_axis[i] else Vertex(pt3(i, True))
              for i in range(n)]

    # arcos (círculos parciais/completos) por vértice fora do eixo
    arcs = [None] * n
    for i in range(n):
        if on_axis[i]:
            continue
        r, z = prof[i]
        circ = Circle(o + z * ax, ax, r, xdir=X)
        arcs[i] = Edge(circ, 0.0, angle, v0[i], v1[i])

    faces: List[Face] = []
    seams0: List = [None] * n                   # aresta do perfil em u=0
    seams1: List = [None] * n                   # ... e em u=angle
    for i in range(n):
        j = (i + 1) % n
        if on_axis[i] and on_axis[j]:
            # segmento sobre o eixo: não gera face de revolução, mas na
            # revolução PARCIAL é aresta real das DUAS tampas (uma vez
            # em cada orientação)
            if not full:
                gen = Line.through(v0[i].point, v0[j].point)
                length = float(np.linalg.norm(v0[j].point - v0[i].point))
                axis_edge = Edge(gen, 0.0, length, v0[i], v0[j])
                seams0[i] = axis_edge
                seams1[i] = axis_edge
            continue
        gen = Line.through(v0[i].point, v0[j].point)
        length = float(np.linalg.norm(v0[j].point - v0[i].point))
        seam0 = Edge(gen, 0.0, length, v0[i], v0[j])
        seams0[i] = seam0
        surf = RevolutionSurface(gen, o, ax)

        if full:
            oes = [(arcs[i], True)] if arcs[i] else []
            oes += [(seam0, True)]
            oes += [(arcs[j], False)] if arcs[j] else []
            oes += [(seam0, False)]
        else:
            gen1 = Line.through(v1[i].point, v1[j].point)
            seam1 = Edge(gen1, 0.0, length, v1[i], v1[j])
            seams1[i] = seam1
            oes = [(seam0, True)]
            oes += [(arcs[j], True)] if arcs[j] else []
            oes += [(seam1, False)]
            oes += [(arcs[i], False)] if arcs[i] else []
        faces.append(Face(surf, [Loop(oes)], same_sense=True,
                          rect_domain=((0.0, angle), (0.0, length))))

    if not full:
        # tampas: perfil em u=0 (normal -Y) e u=angle (normal girada)
        oes0 = [(seams0[i], False) for i in range(n - 1, -1, -1)
                if seams0[i] is not None]
        oes1 = [(seams1[i], True) for i in range(n)
                if seams1[i] is not None]
        n0 = -Y
        n1 = -sa * X + ca * Y
        faces.append(Face(Plane(v0[0].point, n0), [Loop(oes0)],
                          same_sense=True))
        faces.append(Face(Plane(v1[0].point, n1), [Loop(oes1)],
                          same_sense=True))

    from .ops import ensure_outward
    return ensure_outward(Solid(Shell(faces)))


# ------------------------------------------------------------------ loft
def loft(profiles: Sequence, cap: bool = True) -> Solid:
    """
    Sólido de LOFT linear entre seções: ``profiles`` é uma sequência de
    anéis (N, 3) — todos com o MESMO número de vértices e correspondência
    posicional (vértice j do anel i liga ao vértice j do anel i+1).

    Laterais: quadriláteros planos viram uma face; não planos viram duas
    faces triangulares (B-Rep facetado exato). Tampas (``cap=True``):
    faces planas dos anéis extremos (anéis devem ser planos).
    """
    rings = [np.asarray(p, float) for p in profiles]
    n = len(rings[0])
    if any(len(r) != n for r in rings):
        raise ValueError("loft: todos os perfis com o mesmo nº de pontos")
    if len(rings) < 2:
        raise ValueError("loft: pelo menos 2 perfis")

    from ..core.linalg import newell_normal
    from .primitives import planar_face

    verts = [[Vertex(p) for p in ring] for ring in rings]
    cache = _EdgeCache()
    faces: List[Face] = []

    def quad_or_tris(a, b, c, d):
        """Face(s) do quadrilátero a-b-c-d (CCW visto de fora)."""
        pa, pb, pc, pd = a.point, b.point, c.point, d.point
        vol6 = float(np.dot(np.cross(pb - pa, pc - pa), pd - pa))
        diag = max(np.linalg.norm(pc - pa), np.linalg.norm(pd - pb))
        if abs(vol6) < 1e-10 * max(diag, 1.0) ** 3:
            nrm = newell_normal([pa, pb, pc, pd])
            faces.append(planar_face([a, b, c, d], nrm, cache))
        else:
            faces.append(planar_face(
                [a, b, c], newell_normal([pa, pb, pc]), cache))
            faces.append(planar_face(
                [a, c, d], newell_normal([pa, pc, pd]), cache))

    for i in range(len(rings) - 1):
        r0, r1 = verts[i], verts[i + 1]
        for j in range(n):
            k = (j + 1) % n
            quad_or_tris(r0[j], r0[k], r1[k], r1[j])

    if cap:
        n0 = newell_normal([v.point for v in verts[0]])
        n1 = newell_normal([v.point for v in verts[-1]])
        faces.append(planar_face(list(reversed(verts[0])), -n0, cache))
        faces.append(planar_face(verts[-1], n1, cache))

    from .ops import ensure_outward
    return ensure_outward(Solid(Shell(faces)))


# ------------------------------------------------------- sweep genérico
def _rmf_frames(path: np.ndarray):
    """Frames de rotação mínima (método da dupla reflexão) ao longo da
    polilinha; tangentes suavizadas nos vértices internos (miter)."""
    from ..core.linalg import make_frame, unit as _u
    m = len(path)
    tans = []
    for i in range(m):
        if i == 0:
            t = path[1] - path[0]
        elif i == m - 1:
            t = path[-1] - path[-2]
        else:
            t = _u(path[i] - path[i - 1]) + _u(path[i + 1] - path[i])
        tans.append(_u(t))
    x0, y0, _ = make_frame(tans[0])
    frames = [(x0, y0, tans[0])]
    for i in range(1, m):
        x_prev, _, t_prev = frames[-1]
        # reflexão 1: pelo plano bissetor do deslocamento
        v1 = path[i] - path[i - 1]
        c1 = float(v1 @ v1)
        if c1 < 1e-30:
            frames.append(frames[-1])
            continue
        xl = x_prev - (2.0 / c1) * float(v1 @ x_prev) * v1
        tl = t_prev - (2.0 / c1) * float(v1 @ t_prev) * v1
        # reflexão 2: alinha a tangente refletida com a tangente local
        v2 = tans[i] - tl
        c2 = float(v2 @ v2)
        if c2 > 1e-30:
            xl = xl - (2.0 / c2) * float(v2 @ xl) * v2
        xl = _u(xl - float(xl @ tans[i]) * tans[i])
        frames.append((xl, np.cross(tans[i], xl), tans[i]))
    return frames


def sweep_path(profile_xy: Poly2D, path, cap: bool = True) -> Solid:
    """
    Varredura do perfil 2D fechado ao longo de uma POLILINHA 3D pela
    construção clássica de tubo em MEIA-ESQUADRIA: o anel inicial
    (perfil no plano ⟂ à primeira tangente) é PROPAGADO por projeções
    sucessivas — em cada vértice interno, projetado ao longo da tangente
    do segmento sobre o plano bissetor (a seção oblíqua compartilhada
    pelos dois prismas vizinhos), e no fim sobre o plano ⟂ à última
    tangente. Cada trecho é um prisma exato: para perfis centrados,
    ``volume = A · Σ comprimentos``. B-Rep facetado com tampas planas.
    """
    prof = _ccw(np.asarray(profile_xy, float))
    path = np.asarray(path, float)
    if len(path) < 2:
        raise ValueError("sweep_path: caminho com pelo menos 2 pontos")
    from ..core.linalg import make_frame, unit as _u

    m = len(path)
    tans = [_u(path[i + 1] - path[i]) for i in range(m - 1)]
    x0, y0, _ = make_frame(tans[0])
    ring = [path[0] + px * x0 + py * y0 for px, py in prof]
    rings = [list(ring)]

    def project(ring, along, plane_o, plane_n):
        denom = float(along @ plane_n)
        if abs(denom) < 1e-12:
            raise ValueError("sweep_path: curva fechada demais para a "
                             "esquadria (segmentos quase reversos)")
        return [p + (float((plane_o - p) @ plane_n) / denom) * along
                for p in ring]

    for i in range(1, m - 1):
        nrm = _u(tans[i - 1] + tans[i])
        ring = project(ring, tans[i - 1], path[i], nrm)
        rings.append(list(ring))
    ring = project(ring, tans[-1], path[-1], tans[-1])
    rings.append(list(ring))
    return loft(rings, cap=cap)
