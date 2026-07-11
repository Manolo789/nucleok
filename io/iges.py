"""
nucleok.io.iges
===============
Escritor IGES 5.3 (Camada 6) — primeiro degrau da rota IGES: WIREFRAME
das arestas do sólido (entidades 110 Line e 100 Circular Arc; demais
curvas como 106 Copious Data/polilinha).

O formato IGES de colunas fixas (80 colunas, seções S/G/D/P/T) é emitido
integralmente por código próprio. B-Rep completo em IGES (entidades
186/514/510/508/144/142) é marco futuro — ver README.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..geom.curves import Circle, Line
from ..topo.entities import Solid


def _fixed(text: str, section: str, num: int) -> str:
    return f"{text:<72s}{section}{num:>7d}"


def _params(values, ptr_d: int, seq_start: int) -> List[str]:
    """Formata a lista de parâmetros em registros de 64 colunas."""
    body = ",".join(values) + ";"
    chunks = [body[i:i + 64] for i in range(0, len(body), 64)]
    return [f"{c:<64s} {ptr_d:>7d}P{seq_start + k:>7d}"
            for k, c in enumerate(chunks)]


def _g(x: float) -> str:
    return f"{float(x):.9G}"


def write_iges(solid: Solid, path: str, name: str = "nucleok",
               deflection: float = 0.05) -> None:
    """Escreve as arestas do sólido como wireframe IGES."""
    dir_lines: List[str] = []
    par_lines: List[str] = []
    d_seq = 1
    p_seq = 1

    def add_entity(etype: int, values: List[str]):
        nonlocal d_seq, p_seq
        recs = _params(values, d_seq, p_seq)
        n = len(recs)
        # duas linhas de diretório por entidade
        dir_lines.append(_fixed(
            f"{etype:>8d}{p_seq:>8d}{0:>8d}{0:>8d}{0:>8d}{0:>8d}"
            f"{0:>8d}{0:>8d}{'00000000':>8s}", "D", d_seq))
        dir_lines.append(_fixed(
            f"{etype:>8d}{0:>8d}{0:>8d}{n:>8d}{0:>8d}{'':>8s}"
            f"{'':>8s}{'':>8s}{0:>8d}", "D", d_seq + 1))
        par_lines.extend(recs)
        d_seq += 2
        p_seq += n

    for e in solid.unique_edges():
        c = e.curve
        if isinstance(c, Line):
            p0 = c.evaluate(e.t0)
            p1 = c.evaluate(e.t1)
            add_entity(110, ["110"] + [_g(v) for v in (*p0, *p1)])
        elif isinstance(c, Circle):
            # entidade 100 vive no plano XY local (transformação omitida
            # p/ círculos em planos gerais → cai na polilinha)
            if np.allclose(c.normal, [0, 0, 1]):
                z = float(c.center[2])
                a = c.evaluate(e.t0)
                b = c.evaluate(e.t1)
                add_entity(100, ["100", _g(z), _g(c.center[0]),
                                 _g(c.center[1]), _g(a[0]), _g(a[1]),
                                 _g(b[0]), _g(b[1])])
                continue
            pts = e.sample(deflection)
            vals = ["106", "2", str(len(pts))]
            for p in pts:
                vals += [_g(p[0]), _g(p[1]), _g(p[2])]
            add_entity(106, vals)
        else:
            pts = e.sample(deflection)
            vals = ["106", "2", str(len(pts))]
            for p in pts:
                vals += [_g(p[0]), _g(p[1]), _g(p[2])]
            add_entity(106, vals)

    start = [_fixed(f"nucleo-K wireframe: {name}", "S", 1)]
    gvals = [f"1H,,1H;,{len(name)}H{name},{len(name)}H{name},"
             f"7Hnucleok,7Hnucleok,32,308,15,308,15,"
             f"{len(name)}H{name},1.,2,2HMM,1,0.01,"
             f"15H20260710.000000,1E-06,1000.,7Hnucleok,7Hnucleok,11,0,"
             f"15H20260710.000000;"]
    g_lines = []
    body = gvals[0]
    chunks = [body[i:i + 72] for i in range(0, len(body), 72)]
    for k, cband in enumerate(chunks):
        g_lines.append(_fixed(cband, "G", k + 1))

    term = (f"S{len(start):>7d}G{len(g_lines):>7d}D{len(dir_lines):>7d}"
            f"P{len(par_lines):>7d}")
    with open(path, "w") as fh:
        for ln in start + g_lines + dir_lines + par_lines:
            fh.write(ln + "\n")
        fh.write(_fixed(term, "T", 1) + "\n")


# ----------------------------------------------------------------- leitura
def read_iges(path: str):
    """
    Leitor IGES 5.3 do subconjunto WIREFRAME (o mesmo que
    :func:`write_iges` emite): entidades **110** (Line), **100**
    (Circular Arc) e **106** (Copious Data forma 2 = polilinha).

    Devolve uma lista de ``(curva, t0, t1)`` com curvas do núcleo-K
    (:class:`~nucleok.geom.curves.Line`,
    :class:`~nucleok.geom.curves.Circle`,
    :class:`~nucleok.geom.curves.Polyline`), prontas para virar
    :class:`~nucleok.topo.entities.Edge`.

    Entidades fora do subconjunto são ignoradas silenciosamente
    (contabilizadas em ``read_iges.skipped`` após a chamada). IGES B-Rep
    (186/514/510/508) permanece no roadmap.
    """
    import numpy as np

    from ..geom.curves import Circle, Line, Polyline

    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    d_lines = [ln for ln in lines if len(ln) > 72 and ln[72] == "D"]
    p_lines = [ln for ln in lines if len(ln) > 72 and ln[72] == "P"]

    # diretório: 2 linhas por entidade; campo 1 = tipo, campo 2 =
    # ponteiro para a 1ª linha de parâmetros (1-based na seção P)
    entities = []                     # (tipo, seq_D da 1ª linha)
    for k in range(0, len(d_lines) - 1, 2):
        first = d_lines[k]
        etype = int(first[0:8])
        d_seq = int(first[73:80])
        entities.append((etype, d_seq))

    # parâmetros: conteúdo em cols 1..64, ponteiro DE em cols 66..72
    by_de = {}
    for ln in p_lines:
        de = int(ln[65:72])
        by_de.setdefault(de, []).append(ln[0:64])

    def params(de_ptr):
        body = "".join(by_de.get(de_ptr, [])).strip()
        body = body.rstrip(";").rstrip()
        return [tok.strip() for tok in body.split(",") if tok.strip()]

    out = []
    skipped = 0
    for etype, d_seq in entities:
        vals = params(d_seq)
        if not vals or int(float(vals[0])) != etype:
            skipped += 1
            continue
        if etype == 110:                       # Line: x1 y1 z1 x2 y2 z2
            p0 = np.array([float(v) for v in vals[1:4]])
            p1 = np.array([float(v) for v in vals[4:7]])
            ln = Line.through(p0, p1)
            out.append((ln, 0.0, float(np.linalg.norm(p1 - p0))))
        elif etype == 100:            # Arc no plano z: zt cx cy x1 y1 x2 y2
            zt, cx, cy = (float(v) for v in vals[1:4])
            x1, y1, x2, y2 = (float(v) for v in vals[4:8])
            r = float(np.hypot(x1 - cx, y1 - cy))
            c = Circle((cx, cy, zt), (0, 0, 1), r)
            t0 = float(np.arctan2(y1 - cy, x1 - cx)) % (2 * np.pi)
            t1 = float(np.arctan2(y2 - cy, x2 - cx)) % (2 * np.pi)
            if t1 <= t0 + 1e-12:               # arco CCW; fecho completo
                t1 += 2 * np.pi
            out.append((c, t0, t1))
        elif etype == 106:                     # polilinha 3D (forma 2)
            n = int(float(vals[2]))
            pts = np.array([float(v) for v in vals[3:3 + 3 * n]]
                           ).reshape(n, 3)
            pl = Polyline(pts)
            out.append((pl, 0.0, pl.total_length))
        else:
            skipped += 1
    read_iges.skipped = skipped
    return out
