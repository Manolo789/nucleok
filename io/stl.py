"""
nucleok.io.stl
==============
STL binário e ASCII (Camada 6) sobre a tesselação do próprio kernel —
nenhuma dependência externa.
"""

from __future__ import annotations

import struct
from typing import Union

import numpy as np

from ..algo.tessellate import Tessellation, tessellate
from ..topo.entities import Solid


def write_stl(obj: Union[Solid, Tessellation], path: str,
              deflection: float = 0.05, binary: bool = True,
              name: str = "nucleok") -> None:
    t = obj if isinstance(obj, Tessellation) else tessellate(obj,
                                                             deflection)
    v = t.vertices.astype(np.float32)
    f = t.triangles
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    n = (n / ln).astype(np.float32)

    if binary:
        with open(path, "wb") as fh:
            fh.write(name.encode()[:80].ljust(80, b"\0"))
            fh.write(struct.pack("<I", len(f)))
            rec = np.zeros(len(f), dtype=[("n", "<3f4"), ("a", "<3f4"),
                                          ("b", "<3f4"), ("c", "<3f4"),
                                          ("attr", "<u2")])
            rec["n"], rec["a"], rec["b"], rec["c"] = n, p0, p1, p2
            fh.write(rec.tobytes())
    else:
        with open(path, "w") as fh:
            fh.write(f"solid {name}\n")
            for k in range(len(f)):
                fh.write(f"facet normal {n[k,0]:.6e} {n[k,1]:.6e} "
                         f"{n[k,2]:.6e}\n outer loop\n")
                for p in (p0[k], p1[k], p2[k]):
                    fh.write(f"  vertex {p[0]:.6e} {p[1]:.6e} "
                             f"{p[2]:.6e}\n")
                fh.write(" endloop\nendfacet\n")
            fh.write(f"endsolid {name}\n")


def read_stl(path: str) -> Tessellation:
    """Lê STL (binário ou ASCII) como :class:`Tessellation` soldada."""
    with open(path, "rb") as fh:
        head = fh.read(80)
        rest = fh.read()
    if len(rest) >= 4:
        (nf,) = struct.unpack("<I", rest[:4])
        if len(rest) == 4 + nf * 50:
            rec = np.frombuffer(rest, offset=4, count=nf, dtype=[
                ("n", "<3f4"), ("a", "<3f4"), ("b", "<3f4"),
                ("c", "<3f4"), ("attr", "<u2")])
            tri = np.stack([rec["a"], rec["b"], rec["c"]], axis=1)
            verts = tri.reshape(-1, 3).astype(np.float64)
            faces = np.arange(len(verts)).reshape(-1, 3)
            return Tessellation(verts, faces).weld(6)
    text = (head + rest).decode("ascii", errors="ignore")
    coords = [[float(x) for x in ln.split()[1:4]]
              for ln in text.splitlines() if ln.strip().startswith("vertex")]
    if not coords or len(coords) % 3:
        raise ValueError(f"STL inválido: {path}")
    verts = np.asarray(coords, float)
    faces = np.arange(len(verts)).reshape(-1, 3)
    return Tessellation(verts, faces).weld(6)
