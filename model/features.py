"""
nucleok.model.features
======================
Features locais de aresta (Camada 5): :func:`chamfer_edge` e
:func:`fillet_edge` para arestas RETAS entre duas faces PLANAS
(o caso clássico de quebrar cantos de peças prismáticas).

Estratégia: construir um sólido CORTADOR posicionado no frame da aresta
(via :class:`~nucleok.core.transform.Transform.from_frame` +
:func:`~nucleok.model.ops.transform_solid`) e subtraí-lo com
:func:`~nucleok.model.boolean.cut`:

- **chanfro**: prisma triangular cuja base é o plano do chanfro (a
  hipotenusa liga os pontos a distância ``d`` sobre cada face);
- **filete**: prisma quadrilátero (a cunha completa até o centro do
  arco) MENOS o cilindro de raio ``r`` tangente às duas faces — o que
  sobra do corte é exatamente a superfície cilíndrica do filete
  (facetada, como toda booleana v0.2).

Funciona para ângulos diedros convexos genéricos (não só 90°): o centro
do filete é resolvido para ficar a distância ``r`` de AMBOS os planos.
Filetes/chanfros analíticos em arestas curvas (rolling ball) ficam para
as booleanas analíticas.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core.linalg import unit
from ..core.transform import Transform
from ..geom.surfaces import Plane
from ..algo.classify import Location, classify_point
from ..topo.entities import Edge, Face, Solid
from .boolean import cut
from .ops import transform_solid
from .primitives import make_cylinder
from .sweep import extrude


def _adjacent_planar_faces(solid: Solid, edge: Edge) -> Tuple[Face, Face]:
    """As duas faces do sólido que usam a aresta (ambas planas)."""
    hits = []
    for f in solid.faces:
        for lp in f.loops:
            if any(e is edge for e, _ in lp.edges):
                hits.append(f)
                break
    if len(hits) != 2:
        raise ValueError(
            f"aresta compartilhada por {len(hits)} faces (esperado 2)")
    for f in hits:
        if not isinstance(f.surface, Plane):
            raise NotImplementedError(
                "chamfer/fillet v0.2: apenas arestas entre faces PLANAS "
                "(arestas em superfícies curvas exigem booleanas "
                "analíticas)")
    return hits[0], hits[1]


def _edge_frame(solid: Solid, edge: Edge):
    """Frame e direções da cunha: ponto médio E, direção da aresta d̂,
    normais materiais n1/n2 e direções w1/w2 (dentro de cada face,
    perpendiculares à aresta, apontando PARA DENTRO da face)."""
    f1, f2 = _adjacent_planar_faces(solid, edge)
    p0, p1 = edge.start.point, edge.end.point
    d = unit(p1 - p0)
    length = float(np.linalg.norm(p1 - p0))
    E = 0.5 * (p0 + p1)

    def material_normal(f: Face):
        n = f.surface.normal(0.0, 0.0)
        return n if f.same_sense else -n

    n1, n2 = material_normal(f1), material_normal(f2)
    delta = 1e-3 * max(length, 1.0)

    def inward(nf):
        w = unit(np.cross(nf, d))
        probe = E + delta * w - 0.25 * delta * nf
        if classify_point(solid, probe, deflection=delta) \
                is not Location.INSIDE:
            w = -w
        return w

    w1, w2 = inward(n1), inward(n2)
    return E, d, length, n1, n2, w1, w2


def _place_prism(profile_xy, E, d, w_ref, length, margin) -> Solid:
    """Extruda o perfil 2D no frame local (X=w_ref, Z=d̂) e posiciona."""
    z = unit(np.asarray(d, float))
    x = unit(np.asarray(w_ref, float))
    y = np.cross(z, x)
    pr = extrude(profile_xy, length + 2 * margin)
    T = Transform.from_frame(E - (0.5 * length + margin) * z, x, y, z)
    return transform_solid(pr, T)


def chamfer_edge(solid: Solid, edge: Edge, distance: float,
                 deflection: float = 0.01) -> Solid:
    """Chanfro simétrico de largura ``distance`` na aresta reta
    ``edge`` (entre faces planas). Devolve novo sólido (facetado)."""
    E, d, length, n1, n2, w1, w2 = _edge_frame(solid, edge)
    # frame local: X = w1, Y = d×w1 -> coords 2D de w2 e n_out
    z, x = unit(d), unit(w1)
    y = np.cross(z, x)

    def to2d(v):
        return np.array([float(np.dot(v, x)), float(np.dot(v, y))])

    P1 = distance * to2d(w1)
    P2 = distance * to2d(w2)
    n_out = unit(n1 + n2)
    apex = (distance * 2.0) * to2d(n_out)      # fora do material
    prism = _place_prism([tuple(apex), tuple(P1), tuple(P2)],
                         E, d, w1, length, margin=0.05 * length + 1e-6)
    return cut(solid, prism, deflection=deflection)


def fillet_edge(solid: Solid, edge: Edge, radius: float,
                deflection: float = 0.01) -> Solid:
    """Filete (arredondamento) de raio ``radius`` na aresta reta
    ``edge`` (entre faces planas, ângulo diedro convexo). Devolve novo
    sólido; a superfície do filete é o cilindro tangente às duas faces
    (facetado pela deflexão)."""
    E, d, length, n1, n2, w1, w2 = _edge_frame(solid, edge)
    # centro do arco: E + α(w1+w2) com distância r a AMBOS os planos
    denom = abs(float(np.dot(w2, n1)))
    if denom < 1e-9:
        raise ValueError("faces paralelas: filete indefinido")
    alpha = radius / denom
    C3 = E + alpha * (w1 + w2)

    z, x = unit(d), unit(w1)
    y = np.cross(z, x)

    def to2d(v):
        return np.array([float(np.dot(v, x)), float(np.dot(v, y))])

    P1 = radius * to2d(w1)                     # pé da tangência na face 1
    P2 = radius * to2d(w2)
    C = to2d(alpha * (w1 + w2))
    n_out = unit(n1 + n2)
    apex = (radius * 2.0) * to2d(n_out)
    margin = 0.05 * length + 1e-6

    wedge = _place_prism([tuple(apex), tuple(P1), tuple(C), tuple(P2)],
                         E, d, w1, length, margin)
    cyl = make_cylinder(radius, length + 4 * margin)
    Tc = Transform.from_frame(C3 - (0.5 * length + 2 * margin) * z,
                              x, y, z)
    cyl = transform_solid(cyl, Tc)
    # cortador como MALHA crua: evita reconstruir/re-tesselar um B-Rep
    # intermediário (fonte de quase-duplicatas na cadeia de dois cortes)
    cutter = cut(wedge, cyl, deflection=deflection, rebuild=False)
    return cut(solid, cutter, deflection=deflection)
