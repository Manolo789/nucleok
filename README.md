# núcleo-K

Kernel de geometria computacional **B-Rep** em Python puro. Única dependência: **NumPy**.

Nasceu dentro do [parametricus](../README.md), mas é um pacote
**independente**: não importa nada do resto do repositório e pode ser
copiado ou instalado sozinho em qualquer projeto.

```python
from nucleok import make_cylinder, extrude, revolve, validate, volume, write_step

eixo = make_cylinder(radius=8, height=30)
print(validate(eixo))       # [VÁLIDO] V=2 E=3 F=3 R=0  χ=2 (gênero≈0)
print(volume(eixo))         # ≈ π·64·30

flange = revolve([(20, 0), (40, 0), (40, 6), (20, 6)])   # perfil no meio-plano r–z
write_step(flange, "flange.step")                        # STEP AP214, kernel próprio
```

## Arquitetura em camadas

O núcleo-K segue a estratégia de construir um kernel **em camadas**, cada
uma usando apenas as anteriores:

| Camada | Pacote | Conteúdo | Status |
|---|---|---|---|
| 1. Fundação matemática | `nucleok.core` | vetores/frames (`linalg`), transformações afins 4×4 (`transform`), tolerâncias centralizadas (`tolerance`), predicados geométricos **robustos** — `orient2d`/`orient3d` com filtro em float64 e *fallback exato* em `fractions.Fraction` (`predicates`) | ✅ |
| 2. Geometria analítica | `nucleok.geom` | `Line`, `Circle`, `Ellipse`, `Polyline`; **NURBS** completas (Piegl & Tiller: `find_span` A2.1, bases A2.2, derivadas A2.3, inserção de nó A5.1, círculo racional exato de 9 pontos); superfícies `Plane`, `Cylindrical`, `Conical`, `Spherical`, `Toroidal`, `Revolution` com `parameters_of` (inversão) | ✅ |
| 3. Topologia B-Rep | `nucleok.topo` | `Vertex`/`Edge`/`Loop`/`Face`/`Shell`/`Solid` com **separação rigorosa geometria×topologia** (aresta = recorte `[t0,t1]` de curva ilimitada; face = recorte de superfície por loops + `same_sense`) | ✅ |
| 4. Algoritmos fundamentais | `nucleok.algo` + `topo.euler`/`topo.validate` | interseções (reta/plano/quádricas em forma fechada; curva×plano por bisseção+Newton), classificação ponto×sólido (paridade de raio), triangulação 2D com furos (*ear clipping* + pontes de Eberly, decisões por predicado exato), tesselação com deflexão de corda, **operadores de Euler** (`mvfs`/`mev`/`mef` em half-edge), validação Euler–Poincaré + fechamento 2-manifold, propriedades de massa (divergência) | ✅ |
| 5. Modelagem sólida | `nucleok.model` | primitivas (caixa, cilindro, esfera, toro), `extrude` com furos, `revolve` (volta completa E parcial, perfis tocando o eixo — cones com ápice), `loft`, `sweep_path` (esquadria exata nos cantos), **booleanas `fuse`/`common`/`cut`** (BSP facetada + reconstrução B-Rep válida com cicatrização de T-vértices), **`chamfer_edge`/`fillet_edge`** (arestas retas entre faces planas), `transform_solid` (transformação profunda: rígida, escala uniforme, espelho) | ✅ |
| 6. Interoperabilidade | `nucleok.io` | **STEP AP214: escritor E leitor próprios** (round-trip com topologia idêntica e volume preservado, inclusive de resultados booleanos), STL binário/ASCII (leitura e escrita), **IGES 5.3 wireframe: escritor E leitor** (110/100/106, round-trip validado) | ✅ |

## Números da suíte (tests/test_nucleok.py — 72 asserções)

- círculo NURBS racional: desvio de raio ≤ 1e-15; inserção de nó preserva a forma;
- predicados: caso 1e-30 resolvido pelo caminho exato;
- topologias canônicas: caixa V8/E12/F6 χ=2 · cilindro 2/3/3 χ=2 · esfera 2/1/1 χ=2 · toro 1/2/1 χ=0 · cone (perfil no eixo) 3/3/2 χ=2;
- tesselação de recortes em superfícies curvas: área do losango no cilindro < 0,5% do analítico; malhas de sólidos ESTANQUES (amostragem harmonizada);
- tetraedro por operadores de Euler: χ=2 invariante, volume exato 1/6;
- booleanas: caixas com volumes EXATOS (fuse 12, common 4, cut 4, χ=2); caixa−cilindro < 1%; lente esfera∩esfera < 4% (borda facetada);
- chanfro com volume EXATO a³−(d²/2)L; filete < 0,2% do analítico;
- loft (frustum V = h/3(A₁+A₂+√(A₁A₂)) exato) e sweep com esquadria (caminho em Z: volume exato A·ΣL);
- transformação profunda: espelho preserva volume; escala 2× dá exatamente 8× (Δ ~1e-11);
- BVH: classificação idêntica à força bruta, ~135× mais rápida em consultas;
- round-trip STEP: Δvolume = 0 em 6 sólidos + resultado booleano; round-trip IGES wireframe: contagens e comprimentos exatos.

## Roadmap (pós-0.2)

Os oito itens do plano 0.2 (recortes, transformação profunda, booleanas,
chanfros/filetes, loft/sweep, revolução parcial, BVH, leitor IGES) estão
implementados e testados. Refinamentos que permanecem:

1. **Booleanas/filetes ANALÍTICOS** — hoje o resultado é facetado
   (deflexão escolhida pelo chamador); a versão exata (curvas de
   interseção + imprint por operadores de Euler, plano em
   `model/boolean.py::_ANALYTIC_PLAN`) preservaria as superfícies
   curvas.
2. IGES B-Rep (entidades 186/514/510/508) — hoje o IGES é wireframe
   (escrita e leitura).
3. STEP: p-curves em faces trimadas genéricas e instâncias/assemblies.
4. Superfícies NURBS no pipeline topológico (faces sobre
   `NURBSSurface`).
5. Offsets e shelling.

## Uso independente

```
pip install numpy
cp -r nucleok/ seu_projeto/
```

## Contribuidores

<a href="https://github.com/Manolo789/nucleok/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Manolo789/nucleok" />
</a>
