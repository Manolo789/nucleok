"""
nucleok.io.step_writer
======================
Escritor STEP (ISO 10303-21, esquema AP214) — Camada 6, SEM OCCT.

Serializa o B-Rep do núcleo-K como ``MANIFOLD_SOLID_BREP``: a topologia
(ADVANCED_FACE → FACE_BOUND → EDGE_LOOP → ORIENTED_EDGE → EDGE_CURVE →
VERTEX_POINT) mapeia 1:1 nas entidades da Camada 3, e a geometria
(PLANE, CYLINDRICAL/SPHERICAL/CONICAL/TOROIDAL_SURFACE,
SURFACE_OF_REVOLUTION, LINE, CIRCLE, B_SPLINE_CURVE_WITH_KNOTS) nas
classes da Camada 2 — evidência prática da separação geometria/topologia.

Arestas e vértices COMPARTILHADOS entre faces são emitidos uma única vez
(mesma identidade `#id`), preservando a costura topológica no arquivo.
"""

from __future__ import annotations

import datetime
from typing import Dict, List

import numpy as np

from ..geom.curves import Circle, Line
from ..geom.nurbs import NURBSCurve
from ..geom.surfaces import (ConicalSurface, CylindricalSurface, Plane,
                             RevolutionSurface, SphericalSurface,
                             ToroidalSurface)
from ..topo.entities import Edge, Face, Solid, Vertex


def _fmt(x: float) -> str:
    s = f"{float(x):.12g}"
    return s if ("." in s or "E" in s or "e" in s) else s + "."


class _Writer:
    def __init__(self):
        self.lines: List[str] = []
        self.n = 0
        self._vmap: Dict[int, int] = {}      # Vertex.id  -> #id
        self._emap: Dict[int, int] = {}      # Edge.id    -> #id

    def add(self, body: str) -> int:
        self.n += 1
        self.lines.append(f"#{self.n}={body};")
        return self.n

    # ------------------------------------------------------- geometria
    def point(self, p) -> int:
        return self.add(f"CARTESIAN_POINT('',({_fmt(p[0])},{_fmt(p[1])},"
                        f"{_fmt(p[2])}))")

    def direction(self, d) -> int:
        return self.add(f"DIRECTION('',({_fmt(d[0])},{_fmt(d[1])},"
                        f"{_fmt(d[2])}))")

    def axis2(self, origin, zdir, xdir) -> int:
        return self.add(f"AXIS2_PLACEMENT_3D('',#{self.point(origin)},"
                        f"#{self.direction(zdir)},#{self.direction(xdir)})")

    def curve(self, c) -> int:
        if isinstance(c, Line):
            vec = self.add(f"VECTOR('',#{self.direction(c.direction)},1.)")
            return self.add(f"LINE('',#{self.point(c.origin)},#{vec})")
        if isinstance(c, Circle):
            a2 = self.axis2(c.center, c.normal, c.xdir)
            return self.add(f"CIRCLE('',#{a2},{_fmt(c.radius)})")
        if isinstance(c, NURBSCurve) and not c.is_rational:
            pts = ",".join(f"#{self.point(p)}" for p in c.P)
            uk, counts = np.unique(c.knots, return_counts=True)
            mult = ",".join(str(int(m)) for m in counts)
            knots = ",".join(_fmt(k) for k in uk)
            return self.add(
                f"B_SPLINE_CURVE_WITH_KNOTS('',{c.degree},({pts}),"
                f".UNSPECIFIED.,.F.,.F.,({mult}),({knots}),.UNSPECIFIED.)")
        # curva genérica (inclui NURBS racional): amostra como B-spline
        # grau 1 — aproximação declarada, sem quebrar o arquivo
        t0, t1 = c.domain
        pts3 = c.evaluate(np.linspace(t0, t1, 64))
        return self.curve(NURBSCurve.from_polyline(pts3))

    def surface(self, s) -> int:
        if isinstance(s, Plane):
            return self.add(
                f"PLANE('',#{self.axis2(s.origin, s.znormal, s.xdir)})")
        if isinstance(s, CylindricalSurface):
            return self.add(f"CYLINDRICAL_SURFACE('',"
                            f"#{self.axis2(s.origin, s.axis, s.xdir)},"
                            f"{_fmt(s.radius)})")
        if isinstance(s, ConicalSurface):
            return self.add(f"CONICAL_SURFACE('',"
                            f"#{self.axis2(s.origin, s.axis, s.xdir)},"
                            f"{_fmt(s.radius)},{_fmt(s.half_angle)})")
        if isinstance(s, SphericalSurface):
            return self.add(f"SPHERICAL_SURFACE('',"
                            f"#{self.axis2(s.center, s.axis, s.xdir)},"
                            f"{_fmt(s.radius)})")
        if isinstance(s, ToroidalSurface):
            return self.add(f"TOROIDAL_SURFACE('',"
                            f"#{self.axis2(s.center, s.axis, s.xdir)},"
                            f"{_fmt(s.R)},{_fmt(s.r)})")
        if isinstance(s, RevolutionSurface):
            cid = self.curve(s.generatrix)
            a1 = self.add(f"AXIS1_PLACEMENT('',#{self.point(s.origin)},"
                          f"#{self.direction(s.axis)})")
            return self.add(f"SURFACE_OF_REVOLUTION('',#{cid},#{a1})")
        raise NotImplementedError(
            f"superfície {type(s).__name__} ainda sem mapeamento STEP")

    # ------------------------------------------------------- topologia
    def vertex(self, v: Vertex) -> int:
        if v.id not in self._vmap:
            self._vmap[v.id] = self.add(
                f"VERTEX_POINT('',#{self.point(v.point)})")
        return self._vmap[v.id]

    def edge(self, e: Edge) -> int:
        if e.id not in self._emap:
            self._emap[e.id] = self.add(
                f"EDGE_CURVE('',#{self.vertex(e.start)},"
                f"#{self.vertex(e.end)},#{self.curve(e.curve)},.T.)")
        return self._emap[e.id]

    def face(self, f: Face) -> int:
        bound_ids = []
        for i, loop in enumerate(f.loops):
            oe_ids = []
            for e, fwd in loop.edges:
                flag = ".T." if fwd else ".F."
                oe_ids.append(self.add(
                    f"ORIENTED_EDGE('',*,*,#{self.edge(e)},{flag})"))
            lid = self.add("EDGE_LOOP('',("
                           + ",".join(f"#{i_}" for i_ in oe_ids) + "))")
            kind = "FACE_OUTER_BOUND" if i == 0 else "FACE_BOUND"
            bound_ids.append(self.add(f"{kind}('',#{lid},.T.)"))
        sense = ".T." if f.same_sense else ".F."
        bounds = ",".join(f"#{b}" for b in bound_ids)
        return self.add(f"ADVANCED_FACE('',({bounds}),"
                        f"#{self.surface(f.surface)},{sense})")


def write_step(solid: Solid, path: str, name: str = "nucleok_part") -> None:
    """Escreve o sólido como STEP AP214 (MANIFOLD_SOLID_BREP)."""
    w = _Writer()

    # contexto de aplicação/produto (esqueleto AP214 mínimo e válido)
    app = w.add("APPLICATION_CONTEXT('automotive design')")
    w.add(f"APPLICATION_PROTOCOL_DEFINITION('draft international standard',"
          f"'automotive_design',1998,#{app})")
    pctx = w.add(f"PRODUCT_CONTEXT('',#{app},'mechanical')")
    product = w.add(f"PRODUCT('{name}','{name}','',(#{pctx}))")
    pdf = w.add(f"PRODUCT_DEFINITION_FORMATION('','',#{product})")
    pdctx = w.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',"
                  f"#{app},'design')")
    pdef = w.add(f"PRODUCT_DEFINITION('design','',#{pdf},#{pdctx})")
    pds = w.add(f"PRODUCT_DEFINITION_SHAPE('','',#{pdef})")

    # unidades (mm, rad, sr) + incerteza
    lu = w.add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
    au = w.add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    su = w.add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    unc = w.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),"
                f"#{lu},'distance_accuracy_value','')")
    ctx = w.add(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)"
                f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc}))"
                f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{lu},#{au},#{su}))"
                f"REPRESENTATION_CONTEXT('',''))")

    # casca fechada + sólido
    face_ids = [w.face(f) for f in solid.faces]
    shell = w.add("CLOSED_SHELL('',("
                  + ",".join(f"#{i}" for i in face_ids) + "))")
    brep = w.add(f"MANIFOLD_SOLID_BREP('{name}',#{shell})")
    origin = w.axis2((0, 0, 0), (0, 0, 1), (1, 0, 0))
    rep = w.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('',"
                f"(#{origin},#{brep}),#{ctx})")
    w.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    header = (
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION(('nucleo-K B-Rep'),'2;1');\n"
        f"FILE_NAME('{name}','{now}',('nucleo-K'),('nucleo-K'),"
        "'nucleo-K 0.1','nucleo-K','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\n"
        "ENDSEC;\n"
        "DATA;\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        fh.write("\n".join(w.lines))
        fh.write("\nENDSEC;\nEND-ISO-10303-21;\n")
