"""
nucleok.core.transform
======================
Transformações afins 3D (Camada 1): matriz homogênea 4x4 com fábricas para
translação, rotação em torno de eixo arbitrário, escala uniforme e espelho.

Aplica-se distintamente a PONTOS (afetados pela translação), VETORES
(não afetados) e NORMAIS (inversa-transposta — correto sob escala
não-uniforme/espelho).
"""

from __future__ import annotations

import numpy as np

from .linalg import unit


class Transform:
    """Transformação afim rígida ou de similaridade em 3D."""

    __slots__ = ("m",)

    def __init__(self, matrix: np.ndarray | None = None):
        self.m = (np.eye(4) if matrix is None
                  else np.asarray(matrix, dtype=np.float64).reshape(4, 4))

    # ------------------------------------------------------------ fábricas
    @classmethod
    def identity(cls) -> "Transform":
        return cls()

    @classmethod
    def translation(cls, offset) -> "Transform":
        t = cls()
        t.m[:3, 3] = np.asarray(offset, float)
        return t

    @classmethod
    def rotation(cls, axis, angle_rad: float,
                 center=(0.0, 0.0, 0.0)) -> "Transform":
        """Rotação de ``angle_rad`` em torno do eixo por ``center``
        (fórmula de Rodrigues)."""
        k = unit(np.asarray(axis, float))
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]])
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.eye(3) + s * K + (1.0 - c) * (K @ K)
        t = cls()
        t.m[:3, :3] = R
        ctr = np.asarray(center, float)
        t.m[:3, 3] = ctr - R @ ctr
        return t

    @classmethod
    def scaling(cls, factor: float, center=(0.0, 0.0, 0.0)) -> "Transform":
        t = cls()
        t.m[:3, :3] *= float(factor)
        ctr = np.asarray(center, float)
        t.m[:3, 3] = ctr - float(factor) * ctr
        return t

    @classmethod
    def mirror(cls, normal, point=(0.0, 0.0, 0.0)) -> "Transform":
        """Reflexão pelo plano com a normal e ponto dados
        (Householder afim)."""
        n = unit(np.asarray(normal, float))
        H = np.eye(3) - 2.0 * np.outer(n, n)
        t = cls()
        t.m[:3, :3] = H
        p = np.asarray(point, float)
        t.m[:3, 3] = p - H @ p
        return t

    @classmethod
    def from_frame(cls, origin, xdir, ydir, zdir) -> "Transform":
        """Transformação que leva o frame canônico (O, X, Y, Z) ao frame
        dado — útil para posicionar geometria construída em coordenadas
        locais."""
        t = cls()
        t.m[:3, 0] = np.asarray(xdir, float)
        t.m[:3, 1] = np.asarray(ydir, float)
        t.m[:3, 2] = np.asarray(zdir, float)
        t.m[:3, 3] = np.asarray(origin, float)
        return t

    # ---------------------------------------------------------- operações
    def __matmul__(self, other: "Transform") -> "Transform":
        return Transform(self.m @ other.m)

    def inverse(self) -> "Transform":
        return Transform(np.linalg.inv(self.m))

    @property
    def is_rigid(self) -> bool:
        """Preserva distâncias e orientação? (R ortogonal, det=+1)"""
        R = self.m[:3, :3]
        return (np.allclose(R @ R.T, np.eye(3), atol=1e-12)
                and np.linalg.det(R) > 0)

    # ----------------------------------------------------------- aplicação
    def apply_point(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        if p.ndim == 1:
            return self.m[:3, :3] @ p + self.m[:3, 3]
        return p @ self.m[:3, :3].T + self.m[:3, 3]

    def apply_vector(self, vv: np.ndarray) -> np.ndarray:
        vv = np.asarray(vv, float)
        if vv.ndim == 1:
            return self.m[:3, :3] @ vv
        return vv @ self.m[:3, :3].T

    def apply_normal(self, n: np.ndarray) -> np.ndarray:
        """Normais transformam pela inversa-transposta."""
        it = np.linalg.inv(self.m[:3, :3]).T
        n = np.asarray(n, float)
        out = (it @ n) if n.ndim == 1 else n @ it.T
        ln = np.linalg.norm(out, axis=-1, keepdims=(n.ndim > 1))
        return out / ln

    def __repr__(self) -> str:
        return f"Transform(\n{self.m}\n)"
