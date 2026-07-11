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
def _tessellate_grid_face(face: Face, deflection: float,
                          divisions=None) -> Tessellation:
    (u0, u1), (v0, v1) = face.rect_domain
    nu, nv = (divisions if divisions is not None
              else face.surface.divisions(face.rect_domain, deflection))
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


def _unwrap_periodic(uv: np.ndarray, periodic_u: bool,
                     periodic_v: bool) -> np.ndarray:
    """Desdobra saltos de 2π em loops que cruzam a costura de
    superfícies periódicas (continuidade ponto a ponto)."""
    out = uv.copy()
    for col, per in ((0, periodic_u), (1, periodic_v)):
        if not per:
            continue
        for i in range(1, len(out)):
            d = out[i, col] - out[i - 1, col]
            if d > np.pi:
                out[i:, col] -= 2 * np.pi
                # reprocessa a partir daqui
            elif d < -np.pi:
                out[i:, col] += 2 * np.pi
    return out


def _refine_on_surface(surface, uv: np.ndarray, tris: np.ndarray,
                       deflection: float, max_passes: int = 7):
    """
    Refina a triangulação paramétrica até o desvio de corda ficar abaixo
    da deflexão: bisseção do ponto médio em ARESTAS compartilhadas (cache
    de midpoints garante malha sem rachaduras entre triângulos vizinhos).
    """
    uv_list = [p for p in uv]
    tris = [tuple(t) for t in tris]

    def p3(idx):
        u, v = uv_list[idx]
        return surface.evaluate(u, v)

    for _ in range(max_passes):
        # 1) marca arestas cujo ponto médio 3D foge da superfície
        marked = set()
        for a, b, c in tris:
            for i, j in ((a, b), (b, c), (c, a)):
                key = (min(i, j), max(i, j))
                if key in marked:
                    continue
                um = 0.5 * (uv_list[i] + uv_list[j])
                chord_mid = 0.5 * (p3(i) + p3(j))
                if np.linalg.norm(surface.evaluate(um[0], um[1])
                                  - chord_mid) > deflection:
                    marked.add(key)
        if not marked:
            break
        # 2) divide (templates 1/2/3 arestas) com midpoints compartilhados
        mid_cache = {}

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in mid_cache:
                uv_list.append(0.5 * (uv_list[i] + uv_list[j]))
                mid_cache[key] = len(uv_list) - 1
            return mid_cache[key]

        new_tris = []
        for a, b, c in tris:
            ka = (min(a, b), max(a, b)) in marked
            kb = (min(b, c), max(b, c)) in marked
            kc = (min(c, a), max(c, a)) in marked
            n_split = ka + kb + kc
            if n_split == 0:
                new_tris.append((a, b, c))
            elif n_split == 3:
                mab, mbc, mca = midpoint(a, b), midpoint(b, c), \
                    midpoint(c, a)
                new_tris += [(a, mab, mca), (mab, b, mbc),
                             (mca, mbc, c), (mab, mbc, mca)]
            else:
                # rotaciona para a 1ª aresta marcada ficar em (a, b)
                for _ in range(3):
                    if ka:
                        break
                    a, b, c = b, c, a
                    ka, kb, kc = kb, kc, ka
                m = midpoint(a, b)
                if n_split == 1:
                    new_tris += [(a, m, c), (m, b, c)]
                else:                        # 2 arestas: (a,b) e mais uma
                    if kb:
                        m2 = midpoint(b, c)
                        new_tris += [(a, m, c), (m, b, m2), (m, m2, c)]
                    else:
                        m2 = midpoint(c, a)
                        new_tris += [(m2, a, m), (m2, m, c), (m, b, c)]
        tris = new_tris
    return np.asarray(uv_list), np.asarray(tris, np.int64)


def _tessellate_trimmed_face(face: Face, deflection: float) -> Tessellation:
    """Face RECORTADA sobre superfície curva: loops levados ao (u, v) por
    inversão paramétrica (com desdobra de costura), ear clipping no
    domínio e refino por bisseção até a deflexão pedida."""
    surf = face.surface
    (nu0, nu1), (nv0, nv1) = surf.natural_domain
    per_u = np.isfinite(nu0) and np.isfinite(nu1) \
        and abs((nu1 - nu0) - 2 * np.pi) < 1e-9
    per_v = np.isfinite(nv0) and np.isfinite(nv1) \
        and abs((nv1 - nv0) - 2 * np.pi) < 1e-9

    loops_uv = []
    for lp in face.loops:
        uv = []
        for e, fwd in lp.edges:
            pts = e.sample(deflection * 0.5, min_pts=4)
            if not fwd:
                pts = pts[::-1]
            for p in pts[:-1]:
                uv.append(surf.parameters_of(p))
        loops_uv.append(_unwrap_periodic(np.asarray(uv), per_u, per_v))

    areas = [abs(polygon_area_2d(l)) for l in loops_uv]
    k = int(np.argmax(areas))
    outer = loops_uv[k]
    holes = [l for i, l in enumerate(loops_uv) if i != k]

    uv, tris = triangulate_polygon(outer, holes)
    uv, tris = _refine_on_surface(surf, uv, tris, deflection)
    pts = surf.evaluate(uv[:, 0], uv[:, 1])
    if not face.same_sense:               # ∂u×∂v = +normal em todas as
        tris = tris[:, ::-1]              # superfícies do kernel
    return Tessellation(pts, tris)


def tessellate_trimmed(surface, uv_loops, deflection: float = 0.05
                       ) -> Tessellation:
    """Utilidade de teste/uso direto: tessela um recorte dado já no
    espaço paramétrico (primeiro loop = externo)."""
    outer = np.asarray(uv_loops[0], float)
    holes = [np.asarray(h, float) for h in uv_loops[1:]]
    uv, tris = triangulate_polygon(outer, holes)
    uv, tris = _refine_on_surface(surface, uv, tris, deflection)
    pts = surface.evaluate(uv[:, 0], uv[:, 1])
    return Tessellation(pts, tris)


def tessellate_face(face: Face, deflection: float = 0.1,
                    divisions=None) -> Tessellation:
    if face.rect_domain is not None:
        return _tessellate_grid_face(face, deflection, divisions)
    if isinstance(face.surface, Plane):
        return _tessellate_planar_face(face, deflection)
    return _tessellate_trimmed_face(face, deflection)


def _harmonized_divisions(solid: Solid, deflection: float):
    """Divisões (nu, nv) por face, com nu UNIFICADO entre faces em grade
    que giram em torno do mesmo eixo com o mesmo domínio angular —
    paredes vizinhas compartilham os círculos de fronteira e precisam do
    mesmo número de setores para a malha do sólido ser estanque."""
    divs = {}
    groups = {}
    for f in solid.faces:
        if f.rect_domain is None:
            continue
        nu, nv = f.surface.divisions(f.rect_domain, deflection)
        divs[f.id] = [nu, nv]
        ax = getattr(f.surface, "axis", None)
        if ax is None:
            continue
        (u0, u1), _ = f.rect_domain
        key = (tuple(np.round(ax, 9)), round(u0, 9), round(u1, 9))
        groups.setdefault(key, []).append(f.id)
    for ids in groups.values():
        nu_max = max(divs[i][0] for i in ids)
        for i in ids:
            divs[i][0] = nu_max
    return {i: tuple(d) for i, d in divs.items()}


def tessellate(solid: Solid, deflection: float = 0.1) -> Tessellation:
    """Tessela o sólido inteiro (faces fundidas e soldadas), com
    subdivisões harmonizadas entre faces vizinhas para produzir uma
    malha estanque (requisito das booleanas e da classificação)."""
    divs = _harmonized_divisions(solid, deflection)
    parts = [tessellate_face(f, deflection, divs.get(f.id))
             for f in solid.faces]
    return Tessellation.merge(parts).weld()
