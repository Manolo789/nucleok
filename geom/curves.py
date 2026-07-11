"""
nucleok.geom.curves
===================
Curvas paramétricas analíticas (Camada 2): interface comum ``Curve`` e as
cônicas fundamentais — Line, Circle, Ellipse. NURBS em ``nucleok.geom.nurbs``.

Contrato da interface:
    evaluate(t)      ponto 3D (aceita escalar ou array de parâmetros)
    derivative(t)    dC/dt
    domain           (t0, t1) natural da curva
    is_closed        periodicidade
    length(t0, t1)   comprimento de arco (quadratura de Gauss-Legendre)

A GEOMETRIA aqui é ilimitada/completa (uma reta infinita, um círculo
inteiro); o RECORTE em arestas é papel da topologia (Camada 3) — essa
separação é o princípio central do B-Rep.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core.linalg import make_frame, unit


class Curve:
    """Base de curva paramétrica 3D."""

    def evaluate(self, t) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, t) -> np.ndarray:
        """Derivada numérica central por padrão; subclasses sobrescrevem."""
        h = 1e-7
        return (self.evaluate(np.asarray(t) + h)
                - self.evaluate(np.asarray(t) - h)) / (2 * h)

    @property
    def domain(self) -> Tuple[float, float]:
        return (0.0, 1.0)

    @property
    def is_closed(self) -> bool:
        return False

    def tangent(self, t) -> np.ndarray:
        d = self.derivative(t)
        n = np.linalg.norm(d, axis=-1, keepdims=(np.ndim(d) > 1))
        return d / n

    def length(self, t0: float | None = None, t1: float | None = None,
               n_gauss: int = 32) -> float:
        """Comprimento de arco por Gauss–Legendre em ``n_gauss`` nós."""
        a, b = self.domain
        t0 = a if t0 is None else t0
        t1 = b if t1 is None else t1
        x, w = np.polynomial.legendre.leggauss(n_gauss)
        t = 0.5 * (t1 - t0) * x + 0.5 * (t1 + t0)
        speed = np.linalg.norm(self.derivative(t), axis=-1)
        return float(0.5 * (t1 - t0) * np.sum(w * speed))

    def sample(self, t0: float, t1: float, n: int) -> np.ndarray:
        return self.evaluate(np.linspace(t0, t1, n))


class Line(Curve):
    """Reta: ``C(t) = origin + t * direction`` (direção UNITÁRIA, logo o
    parâmetro é comprimento de arco)."""

    def __init__(self, origin, direction):
        self.origin = np.asarray(origin, float)
        self.direction = unit(np.asarray(direction, float))

    @classmethod
    def through(cls, p0, p1) -> "Line":
        return cls(p0, np.asarray(p1, float) - np.asarray(p0, float))

    def evaluate(self, t):
        t = np.asarray(t, float)
        return self.origin + np.multiply.outer(t, self.direction)

    def derivative(self, t):
        t = np.asarray(t, float)
        if t.ndim == 0:
            return self.direction.copy()
        return np.broadcast_to(self.direction, t.shape + (3,)).copy()

    @property
    def domain(self):
        return (-np.inf, np.inf)

    def project(self, p) -> float:
        """Parâmetro do pé da perpendicular de ``p`` sobre a reta."""
        return float(np.dot(np.asarray(p, float) - self.origin,
                            self.direction))


class Circle(Curve):
    """Círculo completo de raio ``radius`` no plano do frame
    ``(center, xdir, ydir)``; parâmetro = ângulo em ``[0, 2π]``."""

    def __init__(self, center, normal, radius: float, xdir=None):
        self.center = np.asarray(center, float)
        self.radius = float(radius)
        if xdir is None:
            self.xdir, self.ydir, self.normal = make_frame(normal)
        else:
            self.normal = unit(np.asarray(normal, float))
            self.xdir = unit(np.asarray(xdir, float))
            self.ydir = np.cross(self.normal, self.xdir)

    def evaluate(self, t):
        t = np.asarray(t, float)
        c, s = np.cos(t), np.sin(t)
        return (self.center + self.radius
                * (np.multiply.outer(c, self.xdir)
                   + np.multiply.outer(s, self.ydir)))

    def derivative(self, t):
        t = np.asarray(t, float)
        c, s = np.cos(t), np.sin(t)
        return self.radius * (np.multiply.outer(-s, self.xdir)
                              + np.multiply.outer(c, self.ydir))

    @property
    def domain(self):
        return (0.0, 2.0 * np.pi)

    @property
    def is_closed(self):
        return True

    def parameter_of(self, p) -> float:
        """Ângulo do ponto (assumido sobre o círculo) em ``[0, 2π)``."""
        d = np.asarray(p, float) - self.center
        ang = float(np.arctan2(np.dot(d, self.ydir), np.dot(d, self.xdir)))
        return ang % (2.0 * np.pi)


class Ellipse(Curve):
    """Elipse com semi-eixos ``a`` (ao longo de xdir) e ``b``."""

    def __init__(self, center, normal, a: float, b: float, xdir=None):
        self.center = np.asarray(center, float)
        self.a, self.b = float(a), float(b)
        if xdir is None:
            self.xdir, self.ydir, self.normal = make_frame(normal)
        else:
            self.normal = unit(np.asarray(normal, float))
            self.xdir = unit(np.asarray(xdir, float))
            self.ydir = np.cross(self.normal, self.xdir)

    def evaluate(self, t):
        t = np.asarray(t, float)
        return (self.center
                + np.multiply.outer(self.a * np.cos(t), self.xdir)
                + np.multiply.outer(self.b * np.sin(t), self.ydir))

    def derivative(self, t):
        t = np.asarray(t, float)
        return (np.multiply.outer(-self.a * np.sin(t), self.xdir)
                + np.multiply.outer(self.b * np.cos(t), self.ydir))

    @property
    def domain(self):
        return (0.0, 2.0 * np.pi)

    @property
    def is_closed(self):
        return True


class Polyline(Curve):
    """Polilinha (útil como curva auxiliar e para perfis); parâmetro é o
    comprimento de arco acumulado."""

    def __init__(self, points):
        self.points = np.asarray(points, float)
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self._cum = np.concatenate([[0.0], np.cumsum(seg)])

    def evaluate(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        t = np.clip(t, 0.0, self._cum[-1])
        i = np.clip(np.searchsorted(self._cum, t, side="right") - 1,
                    0, len(self.points) - 2)
        seg_len = self._cum[i + 1] - self._cum[i]
        seg_len = np.where(seg_len == 0.0, 1.0, seg_len)
        u = ((t - self._cum[i]) / seg_len)[:, None]
        out = (1 - u) * self.points[i] + u * self.points[i + 1]
        return out[0] if np.ndim(t) == 1 and len(t) == 1 else out

    @property
    def domain(self):
        return (0.0, float(self._cum[-1]))
