"""
nucleok.core.predicates
=======================
Predicados geométricos ROBUSTOS (Camada 1).

Decisões topológicas (de que lado? é colinear? é coplanar?) não podem
depender de ruído de ponto flutuante. Estratégia clássica de filtro
aritmético (Shewchuk, simplificado):

1. calcula o determinante em float64;
2. se o valor excede um limite de erro conservador, o SINAL é confiável;
3. caso contrário, recalcula EXATAMENTE com ``fractions.Fraction``
   (aritmética racional de precisão arbitrária do próprio Python).

O caminho exato é lento, mas só dispara em configurações quase
degeneradas — exatamente onde a robustez importa.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import numpy as np

# fator do filtro: erro relativo máximo da expressão em float64
_EPS = np.finfo(np.float64).eps


def _sign(x) -> int:
    return (x > 0) - (x < 0)


# ------------------------------------------------------------------ orient2d
def orient2d(a: Sequence[float], b: Sequence[float],
             c: Sequence[float]) -> int:
    """
    Orientação do trio 2D: +1 anti-horário, -1 horário, 0 colinear.
    Sinal de ``det[[bx-ax, by-ay], [cx-ax, cy-ay]]`` — EXATO.
    """
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])
    detl = (bx - ax) * (cy - ay)
    detr = (by - ay) * (cx - ax)
    det = detl - detr
    bound = 4.0 * _EPS * (abs(detl) + abs(detr))
    if abs(det) > bound:
        return _sign(det)
    # fallback exato
    Ax, Ay = Fraction(ax), Fraction(ay)
    Bx, By = Fraction(bx), Fraction(by)
    Cx, Cy = Fraction(cx), Fraction(cy)
    return _sign((Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax))


# ------------------------------------------------------------------ orient3d
def orient3d(a: Sequence[float], b: Sequence[float],
             c: Sequence[float], d: Sequence[float]) -> int:
    """
    Orientação do quádruplo 3D: sinal do volume assinado do tetraedro
    ``(a, b, c, d)``. +1 se ``d`` está do lado positivo do plano ``(a,b,c)``
    (regra da mão direita), -1 do outro lado, 0 coplanar — EXATO.
    """
    B = np.asarray(b, float) - np.asarray(a, float)
    C = np.asarray(c, float) - np.asarray(a, float)
    D = np.asarray(d, float) - np.asarray(a, float)
    m = np.stack([B, C, D])
    det = float(np.linalg.det(m))
    # limite conservador: soma dos módulos dos 6 termos da expansão
    t = np.abs(m)
    permanent = (t[0, 0] * (t[1, 1] * t[2, 2] + t[1, 2] * t[2, 1])
                 + t[0, 1] * (t[1, 0] * t[2, 2] + t[1, 2] * t[2, 0])
                 + t[0, 2] * (t[1, 0] * t[2, 1] + t[1, 1] * t[2, 0]))
    if abs(det) > 16.0 * _EPS * permanent:
        return _sign(det)
    # fallback exato
    F = Fraction
    ax, ay, az = (F(float(x)) for x in a)
    bx, by, bz = (F(float(x)) for x in b)
    cx, cy, cz = (F(float(x)) for x in c)
    dx, dy, dz = (F(float(x)) for x in d)
    m00, m01, m02 = bx - ax, by - ay, bz - az
    m10, m11, m12 = cx - ax, cy - ay, cz - az
    m20, m21, m22 = dx - ax, dy - ay, dz - az
    det_e = (m00 * (m11 * m22 - m12 * m21)
             - m01 * (m10 * m22 - m12 * m20)
             + m02 * (m10 * m21 - m11 * m20))
    return _sign(det_e)


# --------------------------------------------------------------- derivados
def collinear(a, b, c) -> bool:
    """Três pontos 3D são colineares? (exato via 3 projeções 2D)."""
    pts = [np.asarray(p, float) for p in (a, b, c)]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if orient2d([pts[0][i], pts[0][j]],
                    [pts[1][i], pts[1][j]],
                    [pts[2][i], pts[2][j]]) != 0:
            return False
    return True


def coplanar(a, b, c, d) -> bool:
    """Quatro pontos 3D são coplanares? (exato)."""
    return orient3d(a, b, c, d) == 0


def point_in_triangle_2d(p, a, b, c) -> bool:
    """Ponto 2D dentro (ou na borda) do triângulo — exato."""
    d1 = orient2d(a, b, p)
    d2 = orient2d(b, c, p)
    d3 = orient2d(c, a, p)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def segments_intersect_2d(p1, p2, q1, q2) -> bool:
    """Segmentos 2D ``p1p2`` e ``q1q2`` se intersectam? (exato,
    inclui casos colineares sobrepostos)."""
    d1 = orient2d(q1, q2, p1)
    d2 = orient2d(q1, q2, p2)
    d3 = orient2d(p1, p2, q1)
    d4 = orient2d(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0) or (d1 == 0) or (d2 == 0)) and \
       ((d3 > 0) != (d4 > 0) or (d3 == 0) or (d4 == 0)):
        if d1 == 0 and d2 == 0:                      # colineares
            def on(a, b, c):
                return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
                        and min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))
            return on(p1, p2, q1) or on(p1, p2, q2) \
                or on(q1, q2, p1) or on(q1, q2, p2)
        return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0) \
            or d1 == 0 or d2 == 0 or d3 == 0 or d4 == 0
    return False
