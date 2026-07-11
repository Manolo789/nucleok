"""
nucleok.algo.triangulate
========================
Triangulação de polígonos 2D com furos (Camada 4) por *ear clipping*,
usando os predicados EXATOS da Camada 1 nas decisões de orientação —
imune a colinearidades de ponto flutuante.

Furos são fundidos ao contorno externo por pontes (método do vértice de
máximo-x, Eberly): cada furo vira parte de um único polígono simples, que
então é recortado por orelhas. Complexidade O(n²) — adequada a fronteiras
de faces CAD (dezenas a poucas centenas de vértices).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..core.linalg import polygon_area_2d
from ..core.predicates import orient2d, point_in_triangle_2d


def _ensure_ccw(pts: np.ndarray) -> np.ndarray:
    return pts if polygon_area_2d(pts) > 0 else pts[::-1]


def _ensure_cw(pts: np.ndarray) -> np.ndarray:
    return pts if polygon_area_2d(pts) < 0 else pts[::-1]


def _bridge_hole(outer: List[np.ndarray], hole: List[np.ndarray]) -> List[np.ndarray]:
    """Funde ``hole`` (CW) em ``outer`` (CCW) por uma ponte dupla
    (método do vértice de máximo-x, Eberly)."""
    hi = int(np.argmax([p[0] for p in hole]))
    M = hole[hi]
    n = len(outer)

    # 1) raio +x a partir de M: interseção mais próxima com o contorno
    best_x, best_edge = np.inf, -1
    for i in range(n):
        a, b = outer[i], outer[(i + 1) % n]
        # meia-aberta: evita contar o mesmo vértice em duas arestas
        if not ((a[1] <= M[1] < b[1]) or (b[1] <= M[1] < a[1])):
            continue
        x = a[0] + (M[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        if x >= M[0] - 1e-12 and x < best_x:
            best_x, best_edge = x, i

    if best_edge < 0:
        P_idx = int(np.argmax([p[0] for p in outer]))
    else:
        a = outer[best_edge]
        b = outer[(best_edge + 1) % n]
        I = np.array([best_x, M[1]])
        # 2) candidato: extremo de maior x da aresta atingida
        P_idx = best_edge if a[0] > b[0] else (best_edge + 1) % n
        P = outer[P_idx]

        def key(q):
            return (np.arctan2(abs(q[1] - M[1]), q[0] - M[0]),
                    float(np.hypot(q[0] - M[0], q[1] - M[1])))

        # 3) vértices REFLEXOS dentro do triângulo (M, I, P) roubam a
        #    visibilidade — fica o de menor ângulo com +x (desempate:
        #    mais próximo de M)
        best_key = key(P)
        for j in range(n):
            q = outer[j]
            if q[0] == P[0] and q[1] == P[1]:
                continue
            prev_, next_ = outer[j - 1], outer[(j + 1) % n]
            if orient2d(prev_, q, next_) >= 0:      # não reflexo
                continue
            if point_in_triangle_2d(q, M, I, P):
                k = key(q)
                if k < best_key:
                    best_key, P_idx = k, j

    # 4) costura com vértices duplicados nas duas pontas da ponte
    merged = outer[:P_idx + 1]
    merged += [hole[(hi + k) % len(hole)] for k in range(len(hole))]
    merged += [hole[hi].copy(), outer[P_idx].copy()]
    merged += outer[P_idx + 1:]
    return merged


def triangulate_polygon(outer: Sequence[Sequence[float]],
                        holes: Sequence[Sequence[Sequence[float]]] = ()
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangula o polígono (contorno + furos). Devolve ``(pontos (N,2),
    triângulos (T,3))`` com triângulos ANTI-HORÁRIOS.
    A orientação dos contornos de entrada é normalizada internamente.
    """
    poly = [p for p in np.asarray(outer, float)]
    poly_arr = _ensure_ccw(np.asarray(poly))
    poly = [p for p in poly_arr]

    # furos: do mais à direita para o mais à esquerda (Eberly)
    hs = [_ensure_cw(np.asarray(h, float)) for h in holes]
    hs.sort(key=lambda h: -float(np.max(h[:, 0])))
    for h in hs:
        poly = _bridge_hole(poly, [p for p in h])

    pts = np.asarray(poly)
    n = len(pts)
    idx = list(range(n))
    tris: List[Tuple[int, int, int]] = []

    guard = 0
    while len(idx) > 3 and guard < 4 * n * n:
        guard += 1
        n_cur = len(idx)
        ear_found = False
        for k in range(n_cur):
            i0, i1, i2 = (idx[k - 1], idx[k], idx[(k + 1) % n_cur])
            a, b, c = pts[i0], pts[i1], pts[i2]
            if orient2d(a, b, c) <= 0:             # reflexo ou degenerado
                continue
            # nenhum outro vértice dentro da orelha (predicado exato);
            # duplicados das pontes (mesmas coordenadas dos cantos) não
            # bloqueiam
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                q = pts[j]
                if ((q[0] == a[0] and q[1] == a[1])
                        or (q[0] == b[0] and q[1] == b[1])
                        or (q[0] == c[0] and q[1] == c[1])):
                    continue
                if point_in_triangle_2d(q, a, b, c):
                    ok = False
                    break
            if ok:
                tris.append((i0, i1, i2))
                idx.pop(k)
                ear_found = True
                break
        if not ear_found:                           # duplicatas das pontes
            # remove um vértice degenerado (colinear) e tenta de novo
            for k in range(n_cur):
                i0, i1, i2 = (idx[k - 1], idx[k], idx[(k + 1) % n_cur])
                if orient2d(pts[i0], pts[i1], pts[i2]) == 0:
                    idx.pop(k)
                    break
            else:
                break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return pts, np.asarray(tris, dtype=np.int64)
