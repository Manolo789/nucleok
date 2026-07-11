"""
nucleok.geom.nurbs
==================
Curvas e superfícies NURBS (Camada 2), implementadas do zero segundo os
algoritmos clássicos de Piegl & Tiller ("The NURBS Book"):

    A2.1  find_span            localização do intervalo de nós
    A2.2  basis_funs           funções de base B-spline
    A2.3  ders_basis_funs      derivadas das funções de base
    A3.1/A4.1  avaliação de curva (não-racional / racional)
    A3.5/A4.3  avaliação de superfície tensor-produto
    A5.1  inserção de nó (refino sem mudar a forma)

Racionais (pesos) suportados — o teste canônico é o círculo exato como
NURBS quadrática racional de 9 pontos de controle.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .curves import Curve


# ------------------------------------------------------------- base B-spline
def find_span(n: int, degree: int, t: float, knots: np.ndarray) -> int:
    """Índice ``i`` tal que ``knots[i] <= t < knots[i+1]`` (A2.1)."""
    if t >= knots[n + 1]:
        return n
    if t <= knots[degree]:
        return degree
    lo, hi = degree, n + 1
    mid = (lo + hi) // 2
    while t < knots[mid] or t >= knots[mid + 1]:
        if t < knots[mid]:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) // 2
    return mid


def basis_funs(span: int, t: float, degree: int,
               knots: np.ndarray) -> np.ndarray:
    """As ``degree+1`` funções de base não nulas em ``t`` (A2.2)."""
    N = np.zeros(degree + 1)
    left = np.zeros(degree + 1)
    right = np.zeros(degree + 1)
    N[0] = 1.0
    for j in range(1, degree + 1):
        left[j] = t - knots[span + 1 - j]
        right[j] = knots[span + j] - t
        saved = 0.0
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    return N


def ders_basis_funs(span: int, t: float, degree: int, n_ders: int,
                    knots: np.ndarray) -> np.ndarray:
    """Derivadas 0..n_ders das funções de base (A2.3). Retorna
    ``(n_ders+1, degree+1)``."""
    ndu = np.zeros((degree + 1, degree + 1))
    a = np.zeros((2, degree + 1))
    ders = np.zeros((n_ders + 1, degree + 1))
    left = np.zeros(degree + 1)
    right = np.zeros(degree + 1)
    ndu[0, 0] = 1.0
    for j in range(1, degree + 1):
        left[j] = t - knots[span + 1 - j]
        right[j] = knots[span + j] - t
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved
    ders[0] = ndu[:, degree]
    for r in range(degree + 1):
        s1, s2 = 0, 1
        a[0, 0] = 1.0
        for k in range(1, n_ders + 1):
            d = 0.0
            rk, pk = r - k, degree - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else degree - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            s1, s2 = s2, s1
    r = degree
    for k in range(1, n_ders + 1):
        ders[k] *= r
        r *= degree - k
    return ders


# ------------------------------------------------------------------- curva
class NURBSCurve(Curve):
    """Curva NURBS: grau, nós, pontos de controle ``(n+1, 3)`` e pesos."""

    def __init__(self, degree: int, knots, control_points, weights=None):
        self.degree = int(degree)
        self.knots = np.asarray(knots, float)
        self.P = np.asarray(control_points, float)
        self.W = (np.ones(len(self.P)) if weights is None
                  else np.asarray(weights, float))
        n = len(self.P) - 1
        if len(self.knots) != n + self.degree + 2:
            raise ValueError(
                f"nós inconsistentes: esperado {n + self.degree + 2}, "
                f"recebido {len(self.knots)}")

    # -------------------------------------------------------------- API
    @property
    def domain(self) -> Tuple[float, float]:
        return (float(self.knots[self.degree]),
                float(self.knots[-self.degree - 1]))

    @property
    def is_rational(self) -> bool:
        return not np.allclose(self.W, self.W[0])

    @property
    def is_closed(self) -> bool:
        a, b = self.domain
        return bool(np.linalg.norm(self.evaluate(a)
                                   - self.evaluate(b)) < 1e-9)

    def _point_h(self, t: float) -> np.ndarray:
        """Ponto em coordenadas homogêneas (Pw = [wx, wy, wz, w])."""
        n = len(self.P) - 1
        span = find_span(n, self.degree, t, self.knots)
        N = basis_funs(span, t, self.degree, self.knots)
        idx = slice(span - self.degree, span + 1)
        Pw = np.hstack([self.P[idx] * self.W[idx, None],
                        self.W[idx, None]])
        return N @ Pw

    def evaluate(self, t):
        t = np.asarray(t, float)
        if t.ndim == 0:
            h = self._point_h(float(t))
            return h[:3] / h[3]
        out = np.empty((len(t), 3))
        for i, ti in enumerate(t):
            h = self._point_h(float(ti))
            out[i] = h[:3] / h[3]
        return out

    def derivative(self, t):
        """Primeira derivada (regra do quociente sobre o espaço
        homogêneo — A4.2 com k=1)."""
        t = np.atleast_1d(np.asarray(t, float))
        n = len(self.P) - 1
        out = np.empty((len(t), 3))
        for i, ti in enumerate(t):
            span = find_span(n, self.degree, float(ti), self.knots)
            ders = ders_basis_funs(span, float(ti), self.degree, 1,
                                   self.knots)
            idx = slice(span - self.degree, span + 1)
            Pw = np.hstack([self.P[idx] * self.W[idx, None],
                            self.W[idx, None]])
            A = ders[0] @ Pw          # (x, y, z, w)
            Ad = ders[1] @ Pw
            out[i] = (Ad[:3] * A[3] - A[:3] * Ad[3]) / (A[3] ** 2)
        return out[0] if np.ndim(np.asarray(t)) == 1 and len(t) == 1 else out

    # ------------------------------------------------------ inserção de nó
    def insert_knot(self, t: float, times: int = 1) -> "NURBSCurve":
        """Refino por inserção de nó (A5.1) — a curva NÃO muda de forma
        (propriedade validada em teste)."""
        cur = self
        for _ in range(times):
            cur = cur._insert_once(t)
        return cur

    def _insert_once(self, t: float) -> "NURBSCurve":
        p = self.degree
        n = len(self.P) - 1
        k = find_span(n, p, t, self.knots)
        Pw = np.hstack([self.P * self.W[:, None], self.W[:, None]])
        Q = np.empty((len(Pw) + 1, 4))
        Q[:k - p + 1] = Pw[:k - p + 1]
        Q[k + 1:] = Pw[k:]
        for i in range(k - p + 1, k + 1):
            denom = self.knots[i + p] - self.knots[i]
            alpha = 0.0 if denom == 0.0 else (t - self.knots[i]) / denom
            Q[i] = alpha * Pw[i] + (1.0 - alpha) * Pw[i - 1]
        new_knots = np.insert(self.knots, k + 1, t)
        W = Q[:, 3]
        return NURBSCurve(p, new_knots, Q[:, :3] / W[:, None], W)

    # ------------------------------------------------------------ fábricas
    @classmethod
    def full_circle(cls, center, normal, radius: float) -> "NURBSCurve":
        """Círculo EXATO como NURBS quadrática racional (9 pontos de
        controle, pesos 1 e √2/2 alternados) — o teste canônico de
        implementações NURBS."""
        from ..core.linalg import make_frame
        x, y, _ = make_frame(normal)
        c = np.asarray(center, float)
        r = float(radius)
        w = np.sqrt(2.0) / 2.0
        pts = np.array([
            c + r * x, c + r * (x + y), c + r * y, c + r * (-x + y),
            c - r * x, c - r * (x + y), c - r * y, c + r * (x - y),
            c + r * x,
        ])
        weights = np.array([1, w, 1, w, 1, w, 1, w, 1], float)
        knots = np.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5,
                          0.75, 0.75, 1, 1, 1], float)
        return cls(2, knots, pts, weights)

    @classmethod
    def from_polyline(cls, points) -> "NURBSCurve":
        """Polilinha como B-spline de grau 1 (útil para I/O)."""
        pts = np.asarray(points, float)
        n = len(pts)
        knots = np.concatenate([[0.0], np.linspace(0, 1, n), [1.0]])
        return cls(1, knots, pts)


# --------------------------------------------------------------- superfície
class NURBSSurface:
    """Superfície NURBS tensor-produto (avaliação A3.5/A4.3)."""

    def __init__(self, degree_u: int, degree_v: int, knots_u, knots_v,
                 control_points, weights=None):
        self.pu, self.pv = int(degree_u), int(degree_v)
        self.U = np.asarray(knots_u, float)
        self.V = np.asarray(knots_v, float)
        self.P = np.asarray(control_points, float)     # (nu, nv, 3)
        self.W = (np.ones(self.P.shape[:2]) if weights is None
                  else np.asarray(weights, float))

    @property
    def domain(self):
        return ((float(self.U[self.pu]), float(self.U[-self.pu - 1])),
                (float(self.V[self.pv]), float(self.V[-self.pv - 1])))

    def evaluate(self, u: float, v: float) -> np.ndarray:
        nu = self.P.shape[0] - 1
        nv = self.P.shape[1] - 1
        su = find_span(nu, self.pu, u, self.U)
        sv = find_span(nv, self.pv, v, self.V)
        Nu = basis_funs(su, u, self.pu, self.U)
        Nv = basis_funs(sv, v, self.pv, self.V)
        iu = slice(su - self.pu, su + 1)
        iv = slice(sv - self.pv, sv + 1)
        Wij = self.W[iu, iv]
        Pw = np.concatenate([self.P[iu, iv] * Wij[..., None],
                             Wij[..., None]], axis=-1)
        h = np.einsum("i,ijk,j->k", Nu, Pw, Nv)
        return h[:3] / h[3]

    def normal(self, u: float, v: float, h: float = 1e-6) -> np.ndarray:
        du = (self.evaluate(u + h, v) - self.evaluate(u - h, v)) / (2 * h)
        dv = (self.evaluate(u, v + h) - self.evaluate(u, v - h)) / (2 * h)
        n = np.cross(du, dv)
        return n / np.linalg.norm(n)
