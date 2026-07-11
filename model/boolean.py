"""
nucleok.model.boolean
=====================
Operações booleanas B-Rep (Camada 5) — API estabilizada; implementação é
o PRÓXIMO MARCO do núcleo-K (ver README, seção "Roadmap").

Booleanas exatas em B-Rep são o algoritmo mais delicado de um kernel.
O plano de implementação, apoiado no que JÁ existe nas camadas 1–4:

1. **Interseção face×face** — por par de superfícies: analítica quando
   há forma fechada (plano×plano → ``algo.intersect.plane_plane``;
   plano×quádrica → cônicas) e marching por continuação (Newton nos dois
   campos) no caso geral. Produz as CURVAS DE INTERSEÇÃO.
2. **Imprint** — as curvas são inseridas nas duas faces como novas
   arestas, subdividindo loops (operadores de Euler ``mev``/``mef`` da
   Camada 4 são exatamente as ferramentas para isso).
3. **Classificação** — cada fragmento de face é classificado DENTRO/FORA
   do outro sólido (``algo.classify``, com ponto de teste no interior
   do fragmento).
4. **Seleção e costura** — união: fora∪fora; interseção: dentro∩dentro;
   diferença: A-fora + B-dentro com orientação invertida. Costura das
   cascas e validação (``topo.validate``).

Os predicados exatos da Camada 1 garantem decisões topológicas
consistentes nos casos quase degenerados (faces coplanares, arestas
tangentes) — a causa nº 1 de falhas em booleanas ingênuas.
"""

from __future__ import annotations

from ..topo.entities import Solid


class BooleanNotReady(NotImplementedError):
    """Booleanas B-Rep chegam no próximo marco (plano no docstring do
    módulo). Para booleanas imediatas em nível de malha, use o SDF do
    parametricus com ``MeshSDF`` sobre a tesselação do núcleo-K."""


def fuse(a: Solid, b: Solid) -> Solid:
    raise BooleanNotReady(fuse.__doc__ or BooleanNotReady.__doc__)


def common(a: Solid, b: Solid) -> Solid:
    raise BooleanNotReady(BooleanNotReady.__doc__)


def cut(a: Solid, b: Solid) -> Solid:
    raise BooleanNotReady(BooleanNotReady.__doc__)
