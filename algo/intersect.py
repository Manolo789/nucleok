"""
nucleok.algo.intersect
======================
Interseções fundamentais (Camada 4). São os blocos dos algoritmos maiores
(booleanas, seções): analíticas quando existem em forma fechada,
subdivisão + Newton quando não.

    line_line(l1, l2)            par de parâmetros mais próximos (3D)
    line_plane(line, plane)      parâmetro t da interseção
    plane_plane(p1, p2)          reta de interseção
    ray_triangle(...)            Möller–Trumbore
    line_sphere / line_cylinder  quadráticas clássicas
    curve_plane(curve, plane)    zeros de f(t) = distância assinada
                                 (amostragem + bisseção + Newton)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..geom.curves import Curve, Line
from ..geom.surfaces import Plane


def line_line(l1: Line, l2: Line, tol: float = 1e-12
              ) -> Tuple[float, float, float]:
    """Parâmetros ``(t1, t2)`` dos pontos mais próximos e a distância.
    Retas paralelas: ``t1 = 0`` e o pé correspondente em ``l2``."""
    d1, d2 = l1.direction, l2.direction
    r = l1.origin - l2.origin
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, r))
    e = float(np.dot(d2, r))
    den = a * c - b * b
    if abs(den) < tol:                             # paralelas
        t1 = 0.0
        t2 = e / c
    else:
        t1 = (b * e - c * d) / den
        t2 = (a * e - b * d) / den
    p1 = l1.evaluate(t1)
    p2 = l2.evaluate(t2)
    return t1, t2, float(np.linalg.norm(p1 - p2))


def line_plane(line: Line, plane: Plane,
               tol: float = 1e-12) -> Optional[float]:
    """Parâmetro da interseção reta×plano (``None`` se paralelas)."""
    denom = float(np.dot(line.direction, plane.znormal))
    if abs(denom) < tol:
        return None
    return float(np.dot(plane.origin - line.origin, plane.znormal) / denom)


def plane_plane(p1: Plane, p2: Plane, tol: float = 1e-12) -> Optional[Line]:
    """Reta de interseção de dois planos (``None`` se paralelos)."""
    d = np.cross(p1.znormal, p2.znormal)
    n = float(np.linalg.norm(d))
    if n < tol:
        return None
    # ponto na reta: resolve o sistema 2x2 no plano gerado pelas normais
    n1, n2 = p1.znormal, p2.znormal
    c1 = float(np.dot(n1, p1.origin))
    c2 = float(np.dot(n2, p2.origin))
    n1n2 = float(np.dot(n1, n2))
    det = 1.0 - n1n2 * n1n2
    k1 = (c1 - c2 * n1n2) / det
    k2 = (c2 - c1 * n1n2) / det
    return Line(k1 * n1 + k2 * n2, d / n)


def ray_triangle(origin, direction, a, b, c,
                 eps: float = 1e-12) -> Optional[float]:
    """Möller–Trumbore: ``t ≥ 0`` da interseção raio×triângulo, ou
    ``None``."""
    o = np.asarray(origin, float)
    d = np.asarray(direction, float)
    e1 = np.asarray(b, float) - np.asarray(a, float)
    e2 = np.asarray(c, float) - np.asarray(a, float)
    p = np.cross(d, e2)
    det = float(np.dot(e1, p))
    if abs(det) < eps:
        return None
    inv = 1.0 / det
    tv = o - np.asarray(a, float)
    u = float(np.dot(tv, p)) * inv
    if u < -eps or u > 1 + eps:
        return None
    q = np.cross(tv, e1)
    v = float(np.dot(d, q)) * inv
    if v < -eps or u + v > 1 + eps:
        return None
    t = float(np.dot(e2, q)) * inv
    return t if t >= 0.0 else None


def line_sphere(line: Line, center, radius: float) -> List[float]:
    """Parâmetros das interseções reta×esfera (0, 1 ou 2)."""
    oc = line.origin - np.asarray(center, float)
    b = 2.0 * float(np.dot(line.direction, oc))
    c = float(np.dot(oc, oc)) - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0:
        return []
    s = np.sqrt(disc)
    return sorted({(-b - s) / 2.0, (-b + s) / 2.0})


def line_cylinder(line: Line, origin, axis, radius: float) -> List[float]:
    """Interseções reta×cilindro infinito (projeção no plano ⟂ eixo)."""
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    d = line.direction - float(np.dot(line.direction, ax)) * ax
    o = (line.origin - np.asarray(origin, float))
    o = o - float(np.dot(o, ax)) * ax
    a = float(np.dot(d, d))
    if a < 1e-14:                                  # paralela ao eixo
        return []
    b = 2.0 * float(np.dot(o, d))
    c = float(np.dot(o, o)) - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    s = np.sqrt(disc)
    return sorted({(-b - s) / (2 * a), (-b + s) / (2 * a)})


def curve_plane(curve: Curve, plane: Plane, t0: float | None = None,
                t1: float | None = None, samples: int = 128,
                tol: float = 1e-10) -> List[float]:
    """
    Zeros de ``f(t) = distância assinada de C(t) ao plano`` — método
    genérico para qualquer curva paramétrica: amostragem densa para
    isolar mudanças de sinal + bisseção com refino de Newton.
    """
    a, b = curve.domain
    t0 = a if t0 is None else t0
    t1 = b if t1 is None else t1
    ts = np.linspace(t0, t1, samples)
    f = (curve.evaluate(ts) - plane.origin) @ plane.znormal

    roots: List[float] = []
    for i in range(samples - 1):
        fa, fb = float(f[i]), float(f[i + 1])
        if fa == 0.0:
            roots.append(float(ts[i]))
            continue
        if fa * fb > 0.0:
            continue
        lo, hi = float(ts[i]), float(ts[i + 1])
        flo = fa
        for _ in range(80):                       # bisseção robusta
            mid = 0.5 * (lo + hi)
            fm = float((curve.evaluate(mid) - plane.origin)
                       @ plane.znormal)
            if abs(fm) < tol or hi - lo < tol:
                break
            if flo * fm <= 0.0:
                hi = mid
            else:
                lo, flo = mid, fm
        # polimento de Newton (1–2 passos)
        t = 0.5 * (lo + hi)
        for _ in range(2):
            ft = float((curve.evaluate(t) - plane.origin) @ plane.znormal)
            dft = float(curve.derivative(t) @ plane.znormal)
            if abs(dft) < 1e-14:
                break
            t -= ft / dft
        roots.append(float(np.clip(t, t0, t1)))
    # dedup
    out: List[float] = []
    for r in sorted(roots):
        if not out or abs(r - out[-1]) > 10 * tol:
            out.append(r)
    return out
