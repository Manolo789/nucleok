"""
nucleok.model.ops
=================
Operações estruturais sobre sólidos (Camada 5):

- :func:`transform_solid` — transformação PROFUNDA: clona geometria e
  topologia (mapas de entidades preservam compartilhamento de arestas e
  vértices), suportando translação, rotação, escala UNIFORME e espelho
  (espelho inverte ``same_sense`` para manter as normais materiais
  corretas).
- :func:`ensure_outward` — garante orientação material para fora
  (volume assinado positivo), invertendo loops + ``same_sense`` se
  preciso.
- :func:`solid_from_tessellation` — reconstrói um B-Rep de faces PLANAS
  a partir de uma malha fechada: agrupa triângulos coplanares conexos em
  faces, extrai loops de fronteira (com furos), cicatriza T-vértices e
  compartilha arestas/vértices — a ponte malha→B-Rep usada pelas
  booleanas.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..core.linalg import polygon_area_2d, unit
from ..core.transform import Transform
from ..geom.curves import Circle, Ellipse, Line, Polyline
from ..geom.nurbs import NURBSCurve
from ..geom.surfaces import (ConicalSurface, CylindricalSurface, Plane,
                             RevolutionSurface, SphericalSurface,
                             ToroidalSurface)
from ..algo.tessellate import Tessellation
from ..topo.entities import Edge, Face, Loop, Shell, Solid, Vertex


# ------------------------------------------------------ análise do Trsf
def _similarity(trsf: Transform) -> Tuple[float, int]:
    """Fator de escala uniforme e sinal do determinante; lança se a
    transformação não for de similaridade (escala não uniforme)."""
    R = trsf.m[:3, :3]
    s = np.linalg.norm(R, axis=0)
    if not np.allclose(s, s[0], rtol=1e-9, atol=1e-12):
        raise NotImplementedError(
            "transform_solid: apenas similaridades (translação, rotação, "
            "escala uniforme, espelho); escala não uniforme deformaria "
            "círculos em elipses etc.")
    det = float(np.linalg.det(R))
    return float(s[0]), (1 if det > 0 else -1)


def _tv(trsf, v):
    return trsf.apply_vector(np.asarray(v, float))


# ----------------------------------------------------- geometria clonada
def transform_curve(curve, trsf: Transform, s: float):
    """Clona a curva transformada. Devolve ``(curva, escala_param)`` —
    ``t' = escala_param · t`` (Line/Polyline usam comprimento de arco)."""
    if isinstance(curve, Line):
        return Line(trsf.apply_point(curve.origin),
                    _tv(trsf, curve.direction)), s
    if isinstance(curve, Circle):
        c = Circle(trsf.apply_point(curve.center),
                   unit(_tv(trsf, curve.normal)), curve.radius * s,
                   xdir=unit(_tv(trsf, curve.xdir)))
        c.ydir = _tv(trsf, curve.ydir) / s        # mapeamento EXATO
        c.normal = unit(_tv(trsf, curve.normal))  # (inclusive espelho)
        return c, 1.0
    if isinstance(curve, Ellipse):
        e = Ellipse(trsf.apply_point(curve.center),
                    unit(_tv(trsf, curve.normal)), curve.a * s,
                    curve.b * s, xdir=unit(_tv(trsf, curve.xdir)))
        e.ydir = _tv(trsf, curve.ydir) / s
        return e, 1.0
    if isinstance(curve, NURBSCurve):
        return NURBSCurve(curve.degree, curve.knots.copy(),
                          trsf.apply_point(curve.P),
                          curve.W.copy()), 1.0
    if isinstance(curve, Polyline):
        return Polyline(trsf.apply_point(curve.points)), s
    raise NotImplementedError(
        f"transform_curve: {type(curve).__name__}")


def transform_surface(surface, trsf: Transform, s: float):
    """Clona a superfície transformada. Devolve ``(superfície,
    escala_param_v)`` para o mapeamento do ``rect_domain``."""
    if isinstance(surface, Plane):
        p = Plane(trsf.apply_point(surface.origin),
                  unit(_tv(trsf, surface.znormal)),
                  xdir=unit(_tv(trsf, surface.xdir)))
        p.ydir = _tv(trsf, surface.ydir) / s
        p.znormal = unit(_tv(trsf, surface.znormal))
        return p, s
    if isinstance(surface, CylindricalSurface):
        c = CylindricalSurface(trsf.apply_point(surface.origin),
                               unit(_tv(trsf, surface.axis)),
                               surface.radius * s,
                               xdir=unit(_tv(trsf, surface.xdir)))
        c.ydir = _tv(trsf, surface.ydir) / s
        return c, s
    if isinstance(surface, ConicalSurface):
        c = ConicalSurface(trsf.apply_point(surface.origin),
                           unit(_tv(trsf, surface.axis)),
                           surface.radius * s, surface.half_angle,
                           xdir=unit(_tv(trsf, surface.xdir)))
        c.ydir = _tv(trsf, surface.ydir) / s
        return c, s
    if isinstance(surface, SphericalSurface):
        c = SphericalSurface(trsf.apply_point(surface.center),
                             surface.radius * s,
                             axis=unit(_tv(trsf, surface.axis)),
                             xdir=unit(_tv(trsf, surface.xdir)))
        c.ydir = _tv(trsf, surface.ydir) / s
        return c, 1.0
    if isinstance(surface, ToroidalSurface):
        c = ToroidalSurface(trsf.apply_point(surface.center),
                            unit(_tv(trsf, surface.axis)),
                            surface.R * s, surface.r * s,
                            xdir=unit(_tv(trsf, surface.xdir)))
        c.ydir = _tv(trsf, surface.ydir) / s
        return c, 1.0
    if isinstance(surface, RevolutionSurface):
        gen, pscale = transform_curve(surface.generatrix, trsf, s)
        c = RevolutionSurface(gen, trsf.apply_point(surface.origin),
                              unit(_tv(trsf, surface.axis)))
        c.ydir = _tv(trsf, surface.ydir) / s
        c.xdir = unit(_tv(trsf, surface.xdir))
        return c, pscale
    raise NotImplementedError(
        f"transform_surface: {type(surface).__name__}")


# ---------------------------------------------------- topologia clonada
def transform_solid(solid: Solid, trsf: Transform) -> Solid:
    """Transformação profunda: novo sólido com geometria e topologia
    clonadas (arestas/vértices compartilhados continuam compartilhados)."""
    s, det = _similarity(trsf)
    vmap: Dict[int, Vertex] = {}
    emap: Dict[int, Tuple[Edge, float]] = {}

    def get_vertex(v: Vertex) -> Vertex:
        if v.id not in vmap:
            vmap[v.id] = Vertex(trsf.apply_point(v.point))
        return vmap[v.id]

    def get_edge(e: Edge) -> Edge:
        if e.id not in emap:
            curve, pscale = transform_curve(e.curve, trsf, s)
            emap[e.id] = (Edge(curve, e.t0 * pscale, e.t1 * pscale,
                               get_vertex(e.start), get_vertex(e.end)),
                          pscale)
        return emap[e.id][0]

    def clone_shell(shell: Shell) -> Shell:
        faces = []
        for f in shell.faces:
            surf, vscale = transform_surface(f.surface, trsf, s)
            loops = [Loop([(get_edge(e), fwd) for e, fwd in lp.edges])
                     for lp in f.loops]
            rd = f.rect_domain
            if rd is not None:
                (u0, u1), (v0, v1) = rd
                rd = ((u0, u1), (v0 * vscale, v1 * vscale))
            same = f.same_sense if det > 0 else (not f.same_sense)
            faces.append(Face(surf, loops, same_sense=same,
                              rect_domain=rd))
        return Shell(faces, closed=shell.closed)

    return Solid(clone_shell(solid.outer),
                 [clone_shell(sh) for sh in solid.voids])


def ensure_outward(solid: Solid) -> Solid:
    """Se o volume assinado for negativo, inverte a orientação material
    de todas as faces (in place; devolve o próprio sólido)."""
    from ..algo.properties import signed_volume
    if signed_volume(solid, 0.05) < 0:
        for f in solid.faces:
            f.same_sense = not f.same_sense
            for lp in f.loops:
                lp.edges = [(e, not fwd) for e, fwd in reversed(lp.edges)]
    return solid


# ----------------------------------------------- malha fechada -> B-Rep
def _cluster_weld(V: np.ndarray, T: np.ndarray, tol: float):
    """Solda por AGLOMERAÇÃO espacial: funde vértices a distância < tol
    (hash em células de lado tol, vizinhança 3³) — captura pares que a
    solda por grade separa por caírem em células de arredondamento
    diferentes. Remove triângulos degenerados após a fusão."""
    parent = list(range(len(V)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cells = {}
    keys = np.floor(V / tol).astype(np.int64)
    for i, k in enumerate(map(tuple, keys)):
        cells.setdefault(k, []).append(i)
    tol2 = tol * tol
    for i, k in enumerate(map(tuple, keys)):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in cells.get((k[0] + dx, k[1] + dy,
                                        k[2] + dz), ()):
                        if j > i and float(
                                ((V[i] - V[j]) ** 2).sum()) < tol2:
                            ri, rj = find(i), find(j)
                            if ri != rj:
                                parent[rj] = ri

    root = np.array([find(i) for i in range(len(V))])
    uniq, inv = np.unique(root, return_inverse=True)
    # posição do representante: média do aglomerado
    newV = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(newV, inv, V)
    np.add.at(cnt, inv, 1.0)
    newV /= cnt[:, None]
    newT = inv[T]
    ok = ((newT[:, 0] != newT[:, 1]) & (newT[:, 1] != newT[:, 2])
          & (newT[:, 2] != newT[:, 0]))
    return newV, newT[ok]


def solid_from_tessellation(tess: Tessellation,
                            angle_tol: float = 1e-6,
                            weld_decimals: int = 7) -> Solid:
    """
    Reconstrói um B-Rep de faces PLANAS a partir de uma malha triangular
    fechada:

    1. solda vértices e agrupa triângulos COPLANARES CONEXOS (union-find
       sobre arestas compartilhadas) em regiões = faces;
    2. extrai as fronteiras dirigidas de cada região (arestas usadas uma
       única vez), encadeando-as em loops — o de maior área é o externo,
       os demais são furos;
    3. cicatriza T-VÉRTICES (vértice de uma face caído no meio da aresta
       de outra — inevitável em saídas de BSP), para que cada aresta
       seja compartilhada exatamente por duas faces;
    4. materializa Vertex/Edge/Loop/Face com compartilhamento por
       identidade e valida.
    """
    t = tess.weld(weld_decimals)
    V, T = t.vertices, t.triangles
    if len(T) == 0:
        raise ValueError("tesselação vazia")
    scale = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    dist_tol = 1e-6 * scale
    V, T = _cluster_weld(V, T, dist_tol)
    if len(T) == 0:
        raise ValueError("tesselação degenerou na solda")
    t = Tessellation(V, T)
    N = t.face_normals()
    offs = np.einsum("ij,ij->i", N, V[T[:, 0]])

    # ---- 1) union-find por adjacência coplanar
    parent = list(range(len(T)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    edge2tri: Dict[Tuple[int, int], List[int]] = {}
    for k, (a, b, c) in enumerate(T):
        for i, j in ((a, b), (b, c), (c, a)):
            edge2tri.setdefault((min(i, j), max(i, j)), []).append(k)
    for tris in edge2tri.values():
        for i in range(1, len(tris)):
            k0, k1 = tris[0], tris[i]
            if (np.dot(N[k0], N[k1]) > 1.0 - angle_tol
                    and abs(offs[k0] - offs[k1]) < dist_tol):
                union(k0, k1)

    groups: Dict[int, List[int]] = {}
    for k in range(len(T)):
        groups.setdefault(find(k), []).append(k)

    # ---- 2) fronteiras dirigidas por grupo
    group_bnd: Dict[int, List[Tuple[int, int]]] = {}
    for g, tris in groups.items():
        dir_edges = set()
        for k in tris:
            a, b, c = T[k]
            for i, j in ((a, b), (b, c), (c, a)):
                if (j, i) in dir_edges:
                    dir_edges.remove((j, i))
                else:
                    dir_edges.add((i, j))
        group_bnd[g] = list(dir_edges)

    # ---- 3) cicatrização de T-vértices: qualquer vértice de fronteira
    # colinear e interior a uma aresta de fronteira divide essa aresta
    bnd_vids = sorted({i for bes in group_bnd.values()
                       for e in bes for i in e})
    bnd_pts = V[bnd_vids]

    def split_chain(i, j):
        a, b = V[i], V[j]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < dist_tol * dist_tol:
            return [(i, j)]
        ts = []
        rel = bnd_pts - a
        proj = rel @ ab / L2
        perp2 = np.einsum("ij,ij->i", rel, rel) - proj * proj * L2
        for idx in np.nonzero((perp2 < dist_tol * dist_tol)
                              & (proj > 1e-9) & (proj < 1 - 1e-9))[0]:
            vid = bnd_vids[idx]
            if vid != i and vid != j:
                ts.append((float(proj[idx]), vid))
        chain = [i] + [vid for _, vid in sorted(ts)] + [j]
        return [(chain[k], chain[k + 1]) for k in range(len(chain) - 1)]

    for g in group_bnd:
        healed = []
        for i, j in group_bnd[g]:
            healed += split_chain(i, j)
        group_bnd[g] = healed

    # ---- 4) materialização
    vmap: Dict[int, Vertex] = {}
    emap: Dict[Tuple[int, int], Edge] = {}

    def get_vertex(i):
        if i not in vmap:
            vmap[i] = Vertex(V[i])
        return vmap[i]

    def get_edge(i, j):
        key = (min(i, j), max(i, j))
        if key not in emap:
            va, vb = get_vertex(key[0]), get_vertex(key[1])
            ln = Line.through(va.point, vb.point)
            emap[key] = Edge(ln, 0.0,
                             float(np.linalg.norm(vb.point - va.point)),
                             va, vb)
        return emap[key], (key[0] == i)

    faces: List[Face] = []
    for g, tris in groups.items():
        n = N[tris[0]]
        # normal média ponderada por área (estabilidade)
        cr = np.cross(V[T[tris, 1]] - V[T[tris, 0]],
                      V[T[tris, 2]] - V[T[tris, 0]])
        n = unit(cr.sum(axis=0))
        plane = Plane(V[T[tris[0], 0]], n)

        nxt: Dict[int, List[int]] = {}
        for i, j in group_bnd[g]:
            nxt.setdefault(i, []).append(j)
        loops_idx: List[List[int]] = []
        used = set()
        for i, j in group_bnd[g]:
            if (i, j) in used:
                continue
            loop = [i]
            cur, prev = j, i
            used.add((i, j))
            guard = 0
            while cur != loop[0] and guard < 100000:
                guard += 1
                loop.append(cur)
                outs = [k for k in nxt.get(cur, ())
                        if (cur, k) not in used]
                if not outs:
                    break
                # em vértice de valência >1, evita voltar por onde veio
                nxt_v = outs[0]
                if len(outs) > 1:
                    nxt_v = max(
                        outs,
                        key=lambda k: -1.0 if k == prev else 1.0)
                used.add((cur, nxt_v))
                prev, cur = cur, nxt_v
            loops_idx.append(loop)

        uv_loops = [np.array([plane.parameters_of(V[i]) for i in lp])
                    for lp in loops_idx]
        areas = [abs(polygon_area_2d(l)) for l in uv_loops]
        order = list(np.argsort(areas)[::-1])
        topo_loops = []
        for oi in order:
            lp = loops_idx[oi]
            oes = []
            for k in range(len(lp)):
                e, fwd = get_edge(lp[k], lp[(k + 1) % len(lp)])
                oes.append((e, fwd))
            topo_loops.append(Loop(oes))
        faces.append(Face(plane, topo_loops, same_sense=True))

    return ensure_outward(Solid(Shell(faces)))
