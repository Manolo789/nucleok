"""
nucleok.core.tolerance
======================
Tolerâncias numéricas centralizadas (Camada 1 do núcleo-K).

Todo teste de igualdade geométrica do kernel passa por aqui — nunca por
``==`` ou por epsilons esparramados pelo código. Modelo inspirado no de
kernels comerciais: tolerância LINEAR (distâncias, em unidades do modelo),
ANGULAR (radianos) e PARAMÉTRICA (espaço de parâmetros de curvas e
superfícies).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerance:
    """Conjunto de tolerâncias do kernel."""
    linear: float = 1e-7        # distâncias (unidade do modelo, ex.: mm)
    angular: float = 1e-9       # ângulos (rad)
    parametric: float = 1e-9    # espaço paramétrico

    def same_point(self, d2: float) -> bool:
        """Dois pontos coincidem? Recebe a distância AO QUADRADO."""
        return d2 <= self.linear * self.linear

    def is_zero(self, x: float) -> bool:
        return abs(x) <= self.linear

    def same_angle(self, a: float, b: float) -> bool:
        return abs(a - b) <= self.angular


#: Tolerância padrão global do kernel (modelos em mm).
DEFAULT = Tolerance()
