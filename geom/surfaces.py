"""
nucleok.geom.surfaces
=====================
Superfícies paramétricas analíticas (Camada 2): Plane, Cylindrical,
Conical, Spherical, Toroidal e Revolution — as mesmas classes canônicas
de qualquer kernel B-Rep.

Contrato:
    evaluate(u, v)          ponto 3D
    normal(u, v)            normal unitária (∂u × ∂v normalizado)
    parameters_of(p)        inversão: (u, v) do ponto sobre a superfície
    natural_domain          domínio paramétrico "cheio" da superfície
    divisions(dom, defl)    nº de subdivisões (u, v) p/ tesselar com a
                            deflexão de corda pedida

Todas as superfícies carregam um FRAME (origem + base ortonormal destro),
o que torna transformação e serialização (STEP AXIS2_PLACEMENT_3D)
diretas.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core.linalg import make_frame, unit


def _chord_divs(radius: float, sweep: float, deflection: float,
                lo: int = 8, hi: int = 512) -> int:
    """Nº de segmentos para um arco de ``sweep`` rad em raio ``radius``
    com desvio de corda ≤ ``deflection``."""
    if radius <= 0.0:
        return lo
    x = max(-1.0, min(1.0, 1.0 - deflection / radius))
    step = 2.0 * np.arccos(x)
    if step <= 0.0:
        return hi
    return int(np.clip(np.ceil(sweep / step), lo, hi))


class Surface:
    """Base de superfície paramétrica."""

    def evaluate(self, u, v) -> np.ndarray:
        raise NotImplementedError

    def normal(self, u: float, v: float, h: float = 1e-6) -> np.ndarray:
        du = (self.evaluate(u + h, v) - self.evaluate(u - h, v))
        dv = (self.evaluate(u, v + h) - self.evaluate(u, v - h))
        n = np.cross(du, dv)
        ln = np.linalg.norm(n)
        return n / ln if ln > 0 else n

    def parameters_of(self, p) -> Tuple[float, float]:
        raise NotImplementedError

    @property
    def natural_domain(self):
        return ((0.0, 1.0), (0.0, 1.0))

    def divisions(self, dom, deflection: float) -> Tuple[int, int]:
        return (24, 24)


class Plane(Surface):
    """Plano com frame explícito: ``S(u,v) = origin + u·xdir + v·ydir``."""

    def __init__(self, origin, normal, xdir=None):
        self.origin = np.asarray(origin, float)
        if xdir is None:
            self.xdir, self.ydir, self.znormal = make_frame(normal)
        else:
            self.znormal = unit(np.asarray(normal, float))
            self.xdir = unit(np.asarray(xdir, float))
            self.ydir = np.cross(self.znormal, self.xdir)

    def evaluate(self, u, v):
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        return (self.origin + np.multiply.outer(u, self.xdir)
                + np.multiply.outer(v, self.ydir))

    def normal(self, u=0.0, v=0.0, h=1e-6):
        return self.znormal.copy()

    def parameters_of(self, p):
        d = np.asarray(p, float) - self.origin
        return float(np.dot(d, self.xdir)), float(np.dot(d, self.ydir))

    def signed_distance(self, p) -> float:
        return float(np.dot(np.asarray(p, float) - self.origin,
                            self.znormal))

    def divisions(self, dom, deflection):
        return (1, 1)


class CylindricalSurface(Surface):
    """Cilindro infinito: ``S(u,v) = O + r(cos u·X + sin u·Y) + v·Z``."""

    def __init__(self, origin, axis, radius: float, xdir=None):
        self.origin = np.asarray(origin, float)
        self.radius = float(radius)
        if xdir is None:
            self.xdir, self.ydir, self.axis = make_frame(axis)
        else:
            self.axis = unit(np.asarray(axis, float))
            self.xdir = unit(np.asarray(xdir, float))
            self.ydir = np.cross(self.axis, self.xdir)

    def evaluate(self, u, v):
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        return (self.origin
                + self.radius * (np.multiply.outer(np.cos(u), self.xdir)
                                 + np.multiply.outer(np.sin(u), self.ydir))
                + np.multiply.outer(v, self.axis))

    def parameters_of(self, p):
        d = np.asarray(p, float) - self.origin
        v = float(np.dot(d, self.axis))
        r = d - v * self.axis
        u = float(np.arctan2(np.dot(r, self.ydir),
                             np.dot(r, self.xdir))) % (2 * np.pi)
        return u, v

    @property
    def natural_domain(self):
        return ((0.0, 2 * np.pi), (-np.inf, np.inf))

    def divisions(self, dom, deflection):
        (u0, u1), (v0, v1) = dom
        return (_chord_divs(self.radius, u1 - u0, deflection), 1)


class ConicalSurface(Surface):
    """Cone: raio ``r(v) = radius + v·tan(half_angle)`` ao longo do eixo."""

    def __init__(self, origin, axis, radius: float, half_angle: float,
                 xdir=None):
        self.origin = np.asarray(origin, float)
        self.radius = float(radius)
        self.half_angle = float(half_angle)
        if xdir is None:
            self.xdir, self.ydir, self.axis = make_frame(axis)
        else:
            self.axis = unit(np.asarray(axis, float))
            self.xdir = unit(np.asarray(xdir, float))
            self.ydir = np.cross(self.axis, self.xdir)

    def _r(self, v):
        return self.radius + np.tan(self.half_angle) * np.asarray(v, float)

    def evaluate(self, u, v):
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        r = self._r(v)
        return (self.origin
                + np.multiply.outer(r * np.cos(u), self.xdir)
                + np.multiply.outer(r * np.sin(u), self.ydir)
                + np.multiply.outer(v, self.axis))

    def parameters_of(self, p):
        d = np.asarray(p, float) - self.origin
        v = float(np.dot(d, self.axis))
        r = d - v * self.axis
        u = float(np.arctan2(np.dot(r, self.ydir),
                             np.dot(r, self.xdir))) % (2 * np.pi)
        return u, v

    @property
    def natural_domain(self):
        return ((0.0, 2 * np.pi), (-np.inf, np.inf))

    def divisions(self, dom, deflection):
        (u0, u1), (v0, v1) = dom
        rmax = max(abs(self._r(v0)), abs(self._r(v1)))
        return (_chord_divs(rmax, u1 - u0, deflection), 2)


class SphericalSurface(Surface):
    """Esfera: ``u`` azimute [0,2π], ``v`` latitude [-π/2, π/2]."""

    def __init__(self, center, radius: float, axis=(0, 0, 1), xdir=None):
        self.center = np.asarray(center, float)
        self.radius = float(radius)
        if xdir is None:
            self.xdir, self.ydir, self.axis = make_frame(axis)
        else:
            self.axis = unit(np.asarray(axis, float))
            self.xdir = unit(np.asarray(xdir, float))
            self.ydir = np.cross(self.axis, self.xdir)

    def evaluate(self, u, v):
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        cv = np.cos(v)
        return (self.center + self.radius
                * (np.multiply.outer(cv * np.cos(u), self.xdir)
                   + np.multiply.outer(cv * np.sin(u), self.ydir)
                   + np.multiply.outer(np.sin(v), self.axis)))

    def parameters_of(self, p):
        d = (np.asarray(p, float) - self.center) / self.radius
        z = float(np.clip(np.dot(d, self.axis), -1.0, 1.0))
        v = float(np.arcsin(z))
        u = float(np.arctan2(np.dot(d, self.ydir),
                             np.dot(d, self.xdir))) % (2 * np.pi)
        return u, v

    @property
    def natural_domain(self):
        return ((0.0, 2 * np.pi), (-np.pi / 2, np.pi / 2))

    def divisions(self, dom, deflection):
        (u0, u1), (v0, v1) = dom
        return (_chord_divs(self.radius, u1 - u0, deflection),
                _chord_divs(self.radius, v1 - v0, deflection))


class ToroidalSurface(Surface):
    """Toro: raio maior ``R`` (do eixo ao centro do tubo) e menor ``r``."""

    def __init__(self, center, axis, major_radius: float,
                 minor_radius: float, xdir=None):
        self.center = np.asarray(center, float)
        self.R = float(major_radius)
        self.r = float(minor_radius)
        if xdir is None:
            self.xdir, self.ydir, self.axis = make_frame(axis)
        else:
            self.axis = unit(np.asarray(axis, float))
            self.xdir = unit(np.asarray(xdir, float))
            self.ydir = np.cross(self.axis, self.xdir)

    def evaluate(self, u, v):
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        rad = self.R + self.r * np.cos(v)
        return (self.center
                + np.multiply.outer(rad * np.cos(u), self.xdir)
                + np.multiply.outer(rad * np.sin(u), self.ydir)
                + np.multiply.outer(self.r * np.sin(v), self.axis))

    def parameters_of(self, p):
        d = np.asarray(p, float) - self.center
        z = float(np.dot(d, self.axis))
        u = float(np.arctan2(np.dot(d, self.ydir),
                             np.dot(d, self.xdir))) % (2 * np.pi)
        rad = float(np.linalg.norm(d - z * self.axis))
        v = float(np.arctan2(z, rad - self.R)) % (2 * np.pi)
        return u, v

    @property
    def natural_domain(self):
        return ((0.0, 2 * np.pi), (0.0, 2 * np.pi))

    def divisions(self, dom, deflection):
        (u0, u1), (v0, v1) = dom
        return (_chord_divs(self.R + self.r, u1 - u0, deflection),
                _chord_divs(self.r, v1 - v0, deflection))


class RevolutionSurface(Surface):
    """
    Superfície de revolução genérica: revoluciona uma curva GERATRIZ
    (Camada 2 — qualquer :class:`~nucleok.geom.curves.Curve`) em torno do
    eixo dado. ``u`` = ângulo de revolução; ``v`` = parâmetro da geratriz.
    É a superfície que unifica os sólidos de revolução da Camada 5.
    """

    def __init__(self, generatrix, origin=(0, 0, 0), axis=(0, 0, 1)):
        self.generatrix = generatrix
        self.origin = np.asarray(origin, float)
        self.xdir, self.ydir, self.axis = make_frame(axis)

    def evaluate(self, u, v):
        p = self.generatrix.evaluate(v)               # (…, 3)
        d = p - self.origin
        z = np.dot(d, self.axis)
        rx = np.dot(d, self.xdir)
        ry = np.dot(d, self.ydir)
        r = np.hypot(rx, ry)
        phi0 = np.arctan2(ry, rx)
        ang = phi0 + np.asarray(u, float)
        return (self.origin
                + np.asarray(r * np.cos(ang))[..., None] * self.xdir
                + np.asarray(r * np.sin(ang))[..., None] * self.ydir
                + np.asarray(z)[..., None] * self.axis)

    def parameters_of(self, p):
        d = np.asarray(p, float) - self.origin
        u = float(np.arctan2(np.dot(d, self.ydir),
                             np.dot(d, self.xdir))) % (2 * np.pi)
        # v: resolve no meio-plano (r, z) — projeção sobre a geratriz
        dz = float(np.dot(d, self.axis))
        rr = float(np.linalg.norm(d - dz * self.axis))

        def rz(t):
            q = self.generatrix.evaluate(t) - self.origin
            qz = float(np.dot(q, self.axis))
            return np.array([float(np.linalg.norm(q - qz * self.axis)),
                             qz])

        t0, t1 = self.generatrix.domain
        if not (np.isfinite(t0) and np.isfinite(t1)):
            # geratriz ilimitada (ex.: Line): projeção em forma fechada
            a = rz(0.0)
            b = rz(1.0)
            e = b - a
            ee = float(e @ e)
            if ee < 1e-30:
                return u, 0.0
            t = float((np.array([rr, dz]) - a) @ e / ee)
            return u, t
        ts = np.linspace(t0, t1, 256)
        pts = self.generatrix.evaluate(ts)
        g = pts - self.origin
        gz = g @ self.axis
        gr = np.linalg.norm(g - np.multiply.outer(gz, self.axis), axis=1)
        i = int(np.argmin((gz - dz) ** 2 + (gr - rr) ** 2))
        return u, float(ts[i])

    @property
    def natural_domain(self):
        return ((0.0, 2 * np.pi), self.generatrix.domain)

    def divisions(self, dom, deflection):
        (u0, u1), (v0, v1) = dom
        ts = np.linspace(v0, v1, 64)
        pts = self.generatrix.evaluate(ts)
        d = pts - self.origin
        z = d @ self.axis
        rmax = float(np.max(np.linalg.norm(
            d - np.multiply.outer(z, self.axis), axis=1)))
        nv = max(2, len(_adaptive_curve_params(self.generatrix, v0, v1,
                                               deflection)) - 1)
        return (_chord_divs(rmax, u1 - u0, deflection), nv)


def _adaptive_curve_params(curve, t0: float, t1: float,
                           deflection: float, max_depth: int = 10):
    """Parâmetros de amostragem por bisseção até o desvio de corda ficar
    abaixo da deflexão (usado por tesselação e por superfícies de
    revolução)."""
    params = [t0, t1]

    def refine(a, b, depth):
        pa = curve.evaluate(a)
        pb = curve.evaluate(b)
        m = 0.5 * (a + b)
        pm = curve.evaluate(m)
        chord_mid = 0.5 * (pa + pb)
        if (np.linalg.norm(pm - chord_mid) > deflection
                and depth < max_depth):
            refine(a, m, depth + 1)
            params.append(m)
            refine(m, b, depth + 1)

    refine(t0, t1, 0)
    return np.array(sorted(params))
