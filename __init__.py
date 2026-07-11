"""
núcleo-K — kernel de geometria computacional B-Rep em Python puro
=================================================================

Kernel independente (única dependência: NumPy), construído em camadas:

    1. ``nucleok.core``   — vetores, transformações afins, tolerâncias,
                            predicados geométricos ROBUSTOS (filtro
                            float + fallback exato em racionais)
    2. ``nucleok.geom``   — curvas e superfícies analíticas + NURBS
                            (Piegl & Tiller), geometria ILIMITADA
    3. ``nucleok.topo``   — Vertex/Edge/Loop/Face/Shell/Solid, operadores
                            de Euler (half-edge), validação topológica
    4. ``nucleok.algo``   — interseções, classificação ponto×sólido,
                            triangulação com furos, tesselação,
                            propriedades de massa
    5. ``nucleok.model``  — primitivas, extrusão, revolução; booleanas
                            (API pronta, implementação = próximo marco)
    6. ``nucleok.io``     — STEP (escritor E leitor próprios), IGES
                            (wireframe), STL — SEM OCCT

Uso rápido::

    from nucleok import make_cylinder, write_step, validate, volume
    s = make_cylinder(radius=8, height=30)
    print(validate(s))                 # V=2 E=3 F=3 χ=2 [VÁLIDO]
    print(volume(s))                   # ~π·64·30
    write_step(s, "eixo.step")         # STEP AP214, kernel próprio

Independência: este pacote não importa nada do ``parametricus`` — pode
ser copiado/instalado sozinho em qualquer projeto.
"""

from .core.tolerance import Tolerance, DEFAULT as DEFAULT_TOLERANCE
from .core.transform import Transform
from .core.predicates import (orient2d, orient3d, collinear, coplanar,
                              point_in_triangle_2d)
from .geom.curves import Curve, Line, Circle, Ellipse, Polyline
from .geom.nurbs import NURBSCurve, NURBSSurface
from .geom.surfaces import (Surface, Plane, CylindricalSurface,
                            ConicalSurface, SphericalSurface,
                            ToroidalSurface, RevolutionSurface)
from .topo.entities import Vertex, Edge, Loop, Face, Shell, Solid
from .topo.euler import HEModel
from .topo.validate import validate, ValidationReport
from .algo.intersect import (line_line, line_plane, plane_plane,
                             ray_triangle, line_sphere, line_cylinder,
                             curve_plane)
from .algo.triangulate import triangulate_polygon
from .algo.tessellate import Tessellation, tessellate, tessellate_face
from .algo.classify import classify_point, is_inside, Location
from .algo.properties import (volume, signed_volume, surface_area,
                              centroid, edge_length_total)
from .model.primitives import (make_box, make_cylinder, make_sphere,
                               make_torus)
from .model.sweep import extrude, revolve
from .model import boolean
from .io.stl import write_stl, read_stl
from .io.step_writer import write_step
from .io.step_reader import read_step
from .io.iges import write_iges

__version__ = "0.1.0"

__all__ = [
    "Tolerance", "DEFAULT_TOLERANCE", "Transform",
    "orient2d", "orient3d", "collinear", "coplanar",
    "point_in_triangle_2d",
    "Curve", "Line", "Circle", "Ellipse", "Polyline",
    "NURBSCurve", "NURBSSurface",
    "Surface", "Plane", "CylindricalSurface", "ConicalSurface",
    "SphericalSurface", "ToroidalSurface", "RevolutionSurface",
    "Vertex", "Edge", "Loop", "Face", "Shell", "Solid",
    "HEModel", "validate", "ValidationReport",
    "line_line", "line_plane", "plane_plane", "ray_triangle",
    "line_sphere", "line_cylinder", "curve_plane",
    "triangulate_polygon", "Tessellation", "tessellate",
    "tessellate_face",
    "classify_point", "is_inside", "Location",
    "volume", "signed_volume", "surface_area", "centroid",
    "edge_length_total",
    "make_box", "make_cylinder", "make_sphere", "make_torus",
    "extrude", "revolve", "boolean",
    "write_stl", "read_stl", "write_step", "read_step", "write_iges",
]
