"""
nucleok.topo.validate
=====================
Validação topológica de sólidos B-Rep (Camada 4).

Verificações:
- **Fechamento 2-manifold**: em uma casca fechada, cada aresta é usada
  EXATAMENTE duas vezes pelo conjunto de loops, com orientações opostas
  (a condição de "costura" que garante superfície sem borda e com
  orientação consistente).
- **Conectividade dos loops**: cada aresta termina onde a próxima começa.
- **Euler–Poincaré**: ``V − E + F = 2(S − H) + R`` — reportamos
  ``χ = V − E + F`` e o gênero implícito ``H = S − χ/2 + R/2`` (para
  cascas sem anéis, χ=2 ⇒ esfera topológica; χ=0 ⇒ toro).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .entities import Solid


@dataclass
class ValidationReport:
    ok: bool
    V: int
    E: int
    F: int
    rings: int                       # loops internos (furos em faces)
    euler_characteristic: int
    genus: float                     # H estimado p/ S=1
    issues: List[str] = field(default_factory=list)

    def __str__(self):
        status = "VÁLIDO" if self.ok else "INVÁLIDO"
        lines = [f"[{status}] V={self.V} E={self.E} F={self.F} "
                 f"R={self.rings}  χ={self.euler_characteristic} "
                 f"(gênero≈{self.genus:g})"]
        lines += [f"  ! {m}" for m in self.issues]
        return "\n".join(lines)


def validate(solid: Solid) -> ValidationReport:
    issues: List[str] = []

    # uso de arestas: id -> [orientações]
    usage: Dict[int, List[bool]] = {}
    rings = 0
    for face in solid.faces:
        rings += max(0, len(face.loops) - 1)
        for i, loop in enumerate(face.loops):
            if not loop.edges:
                issues.append(f"{face}: loop vazio")
                continue
            if not loop.is_connected():
                issues.append(f"{face}: loop {i} desconexo "
                              "(aresta não termina onde a próxima começa)")
            for e, fwd in loop.edges:
                usage.setdefault(e.id, []).append(fwd)

    for eid, uses in usage.items():
        if len(uses) != 2:
            issues.append(f"aresta #{eid} usada {len(uses)}x "
                          "(casca fechada exige exatamente 2)")
        elif uses[0] == uses[1]:
            issues.append(f"aresta #{eid} usada 2x na MESMA orientação "
                          "(faces vizinhas com normais inconsistentes)")

    V = len(solid.unique_vertices())
    E = len(solid.unique_edges())
    F = len(solid.faces)
    chi = V - E + F - rings
    genus = (2 * len(solid.shells) - chi) / 2.0 - len(solid.voids)

    return ValidationReport(ok=not issues, V=V, E=E, F=F, rings=rings,
                            euler_characteristic=chi, genus=genus,
                            issues=issues)
