"""
nucleok.io.step_reader
======================
Leitor STEP (ISO 10303-21) — Camada 6, SEM OCCT.

Cobre o subconjunto B-Rep manifold que o escritor emite (e que a maioria
dos CADs exporta para peças analíticas): MANIFOLD_SOLID_BREP /
CLOSED_SHELL / ADVANCED_FACE com PLANE, CYLINDRICAL/CONICAL/SPHERICAL/
TOROIDAL_SURFACE e SURFACE_OF_REVOLUTION; arestas LINE, CIRCLE e
B_SPLINE_CURVE_WITH_KNOTS. Round-trip com o escritor é validado na suíte
(topologia idêntica + volume preservado).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np

from ..geom.curves import Circle, Line
from ..geom.nurbs import NURBSCurve
from ..geom.surfaces import (ConicalSurface, CylindricalSurface, Plane,
                             RevolutionSurface, SphericalSurface,
                             ToroidalSurface)
from ..topo.entities import Edge, Face, Loop, Shell, Solid, Vertex

_ENT_RE = re.compile(r"#(\d+)\s*=\s*(.+);\s*$")
_HEAD_RE = re.compile(r"^([A-Z0-9_]+)\s*\((.*)\)$", re.S)


def _split_args(s: str) -> List[str]:
    """Divide argumentos no nível superior (respeita parênteses e
    aspas)."""
    out, depth, cur, in_str = [], 0, [], False
    for ch in s:
        if in_str:
            cur.append(ch)
            if ch == "'":
                in_str = False
            continue
        if ch == "'":
            in_str = True
            cur.append(ch)
        elif ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return out


class StepFile:
    """Arquivo STEP tokenizado: ``entities[id] = (TIPO, [args])``."""

    def __init__(self, path: str):
        with open(path, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        data = text.split("DATA;", 1)[1].split("ENDSEC;", 1)[0]
        # junta linhas de continuação de cada instância
        self.entities: Dict[int, Tuple[str, List[str]]] = {}
        for raw in data.replace("\n", " ").split(";"):
            raw = raw.strip()
            if not raw.startswith("#"):
                continue
            eid, body = raw[1:].split("=", 1)
            body = body.strip()
            m = _HEAD_RE.match(body)
            if not m:                    # instância complexa "(A()B())"
                self.entities[int(eid)] = ("COMPLEX", [body])
                continue
            self.entities[int(eid)] = (m.group(1),
                                       _split_args(m.group(2)))

    # ----------------------------------------------------------- helpers
    def ref(self, arg: str) -> int:
        return int(arg.strip().lstrip("#"))

    def refs(self, arg: str) -> List[int]:
        inner = arg.strip()[1:-1]
        return [self.ref(a) for a in _split_args(inner) if a.strip()]

    def floats(self, arg: str) -> List[float]:
        inner = arg.strip()[1:-1]
        return [float(a) for a in _split_args(inner)]

    def by_type(self, name: str) -> List[int]:
        return [i for i, (t, _) in self.entities.items() if t == name]

    # ---------------------------------------------------------- geometria
    def _point(self, eid: int) -> np.ndarray:
        _, args = self.entities[eid]
        return np.asarray(self.floats(args[1]))

    def _dir(self, eid: int) -> np.ndarray:
        _, args = self.entities[eid]
        return np.asarray(self.floats(args[1]))

    def _axis2(self, eid: int):
        _, args = self.entities[eid]
        o = self._point(self.ref(args[1]))
        z = self._dir(self.ref(args[2]))
        x = (self._dir(self.ref(args[3])) if len(args) > 3
             and args[3].strip() != "$" else None)
        return o, z, x

    def _curve(self, eid: int):
        typ, args = self.entities[eid]
        if typ == "LINE":
            o = self._point(self.ref(args[1]))
            _, vargs = self.entities[self.ref(args[2])]      # VECTOR
            d = self._dir(self.ref(vargs[1])) * float(vargs[2])
            return Line(o, d)
        if typ == "CIRCLE":
            o, z, x = self._axis2(self.ref(args[1]))
            return Circle(o, z, float(args[2]), xdir=x)
        if typ == "B_SPLINE_CURVE_WITH_KNOTS":
            degree = int(args[1])
            pts = np.array([self._point(i) for i in self.refs(args[2])])
            mult = [int(m) for m in _split_args(args[6].strip()[1:-1])]
            kn = self.floats(args[7])
            knots = np.repeat(kn, mult)
            return NURBSCurve(degree, knots, pts)
        raise NotImplementedError(f"curva STEP não suportada: {typ}")

    def _surface(self, eid: int):
        typ, args = self.entities[eid]
        if typ == "PLANE":
            o, z, x = self._axis2(self.ref(args[1]))
            return Plane(o, z, xdir=x)
        if typ == "CYLINDRICAL_SURFACE":
            o, z, x = self._axis2(self.ref(args[1]))
            return CylindricalSurface(o, z, float(args[2]), xdir=x)
        if typ == "CONICAL_SURFACE":
            o, z, x = self._axis2(self.ref(args[1]))
            return ConicalSurface(o, z, float(args[2]), float(args[3]),
                                  xdir=x)
        if typ == "SPHERICAL_SURFACE":
            o, z, x = self._axis2(self.ref(args[1]))
            return SphericalSurface(o, float(args[2]), axis=z, xdir=x)
        if typ == "TOROIDAL_SURFACE":
            o, z, x = self._axis2(self.ref(args[1]))
            return ToroidalSurface(o, z, float(args[2]), float(args[3]),
                                   xdir=x)
        if typ == "SURFACE_OF_REVOLUTION":
            gen = self._curve(self.ref(args[1]))
            _, a1 = self.entities[self.ref(args[2])]  # AXIS1_PLACEMENT
            o = self._point(self.ref(a1[1]))
            z = self._dir(self.ref(a1[2]))
            return RevolutionSurface(gen, o, z)
        raise NotImplementedError(f"superfície STEP não suportada: {typ}")


def read_step(path: str) -> List[Solid]:
    """Lê todos os MANIFOLD_SOLID_BREP do arquivo como sólidos do
    núcleo-K."""
    sf = StepFile(path)
    vcache: Dict[int, Vertex] = {}
    ecache: Dict[int, Edge] = {}

    def get_vertex(eid: int) -> Vertex:
        if eid not in vcache:
            _, args = sf.entities[eid]                # VERTEX_POINT
            vcache[eid] = Vertex(sf._point(sf.ref(args[1])))
        return vcache[eid]

    def get_edge(eid: int) -> Edge:
        if eid not in ecache:
            _, args = sf.entities[eid]                # EDGE_CURVE
            v1 = get_vertex(sf.ref(args[1]))
            v2 = get_vertex(sf.ref(args[2]))
            curve = sf._curve(sf.ref(args[3]))
            # intervalo paramétrico deduzido dos vértices sobre a curva
            if isinstance(curve, Line):
                t0 = curve.project(v1.point)
                t1 = curve.project(v2.point)
            elif isinstance(curve, Circle):
                if v1 is v2:
                    t0, t1 = 0.0, 2.0 * np.pi
                    # reancora o xdir no vértice p/ params coerentes
                    curve = Circle(curve.center, curve.normal,
                                   curve.radius,
                                   xdir=(v1.point - curve.center)
                                   / curve.radius)
                else:
                    t0 = curve.parameter_of(v1.point)
                    t1 = curve.parameter_of(v2.point)
                    if t1 <= t0:
                        t1 += 2.0 * np.pi
            else:                                     # NURBS
                t0, t1 = curve.domain
            ecache[eid] = Edge(curve, t0, t1, v1, v2)
        return ecache[eid]

    def get_loop(eid: int) -> Loop:
        _, args = sf.entities[eid]                    # EDGE_LOOP
        oes = []
        for oe_id in sf.refs(args[1]):
            _, oargs = sf.entities[oe_id]             # ORIENTED_EDGE
            fwd = oargs[4].strip() == ".T."
            oes.append((get_edge(sf.ref(oargs[3])), fwd))
        return Loop(oes)

    solids: List[Solid] = []
    for msb in sf.by_type("MANIFOLD_SOLID_BREP"):
        _, margs = sf.entities[msb]
        _, sargs = sf.entities[sf.ref(margs[1])]      # CLOSED_SHELL
        faces = []
        for fid in sf.refs(sargs[1]):
            _, fargs = sf.entities[fid]               # ADVANCED_FACE
            loops = []
            outer_first = []
            for bid in sf.refs(fargs[1]):
                btyp, bargs = sf.entities[bid]
                lp = get_loop(sf.ref(bargs[1]))
                if btyp == "FACE_OUTER_BOUND":
                    outer_first.insert(0, lp)
                else:
                    loops.append(lp)
            surface = sf._surface(sf.ref(fargs[2]))
            same_sense = fargs[3].strip() == ".T."
            face = Face(surface, outer_first + loops, same_sense)
            _restore_rect_domain(face)
            faces.append(face)
        solids.append(Solid(Shell(faces)))
    return solids


def _restore_rect_domain(face: Face) -> None:
    """Reconstrói a dica de patch retangular para faces não-planas de
    patch completo (necessária para a tesselação por grade)."""
    if isinstance(face.surface, Plane):
        return
    # face de patch completo: domínio u = volta inteira; v = varredura
    # dos parâmetros dos vértices sobre a superfície
    vs = []
    for lp in face.loops:
        for e, _ in lp.edges:
            for vt in (e.start, e.end):
                vs.append(face.surface.parameters_of(vt.point)[1])
    if not vs:
        return
    v0, v1 = float(min(vs)), float(max(vs))
    (nu0, nu1), (nv0, nv1) = face.surface.natural_domain
    if isinstance(face.surface, SphericalSurface):
        v0, v1 = nv0, nv1
    if isinstance(face.surface, ToroidalSurface):
        v0, v1 = nv0, nv1
    if isinstance(face.surface, RevolutionSurface):
        # geratriz possivelmente ilimitada: v vem dos vértices (acima) e
        # o domínio natural em u é a volta completa
        pass
    if v1 - v0 < 1e-12:
        v0, v1 = nv0, nv1
    if not (np.isfinite(v0) and np.isfinite(v1)):
        return
    face.rect_domain = ((nu0, nu1), (v0, v1))
