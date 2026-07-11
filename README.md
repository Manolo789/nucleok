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
| 1. Fundação matemática | `nucleok.core` | vetores/frames (`linalg`), transformações afins 4×4 (`transform`), tolerâncias centralizadas (`tolerance`), predicados geométricos **robustos** — `orient2d`/`orient3d` com filtro em float64 e *fallback exato* em `fractions.Fraction` (`predicates`) | - [X] |
| 2. Geometria analítica | `nucleok.geom` | `Line`, `Circle`, `Ellipse`, `Polyline`; **NURBS** completas (Piegl & Tiller: `find_span` A2.1, bases A2.2, derivadas A2.3, inserção de nó A5.1, círculo racional exato de 9 pontos); superfícies `Plane`, `Cylindrical`, `Conical`, `Spherical`, `Toroidal`, `Revolution` com `parameters_of` (inversão) | - [X] |
| 3. Topologia B-Rep | `nucleok.topo` | `Vertex`/`Edge`/`Loop`/`Face`/`Shell`/`Solid` com **separação rigorosa geometria×topologia** (aresta = recorte `[t0,t1]` de curva ilimitada; face = recorte de superfície por loops + `same_sense`) | - [X] |
| 4. Algoritmos fundamentais | `nucleok.algo` + `topo.euler`/`topo.validate` | interseções (reta/plano/quádricas em forma fechada; curva×plano por bisseção+Newton), classificação ponto×sólido (paridade de raio), triangulação 2D com furos (*ear clipping* + pontes de Eberly, decisões por predicado exato), tesselação com deflexão de corda, **operadores de Euler** (`mvfs`/`mev`/`mef` em half-edge), validação Euler–Poincaré + fechamento 2-manifold, propriedades de massa (divergência) | -[X] |
| 5. Modelagem sólida | `nucleok.model` | primitivas (caixa, cilindro, esfera, toro — B-Rep costurado, χ verificado), `extrude` com furos, `revolve`; **booleanas: API pronta, implementação é o próximo marco** (plano em `model/boolean.py`); filetes/chanfros após booleanas | -[ ] |
| 6. Interoperabilidade | `nucleok.io` | **STEP AP214: escritor E leitor próprios** (`MANIFOLD_SOLID_BREP`, superfícies analíticas + `SURFACE_OF_REVOLUTION` + B-splines; round-trip com topologia idêntica e volume preservado), STL binário/ASCII (leitura e escrita), IGES 5.3 wireframe | -[ ] |

## Números da suíte (tests/test_nucleok.py)

- círculo NURBS racional: desvio de raio ≤ 1e-15; inserção de nó preserva a forma;
- predicados: caso 1e-30 resolvido pelo caminho exato;
- topologias canônicas: caixa V8/E12/F6 χ=2 · cilindro 2/3/3 χ=2 · esfera 2/1/1 χ=2 · toro 1/2/1 χ=0 · prisma-com-furo χ=0 (gênero 1);
- volumes vs. analítico < 1% (deflexão 0.005); revolução confere com Pappus;
- tetraedro por operadores de Euler: χ=2 invariante em todos os passos, volume exato 1/6;
- round-trip STEP (escreve→lê): Δ de volume = 0 para caixa, cilindro, esfera, toro, revolução e extrusão com furo.

## Roadmap

1. **Booleanas B-Rep** (`fuse`/`common`/`cut`) — interseção face×face
   (analítica + marching), *imprint* com operadores de Euler,
   classificação e costura; ver plano detalhado em `model/boolean.py`.
2. Filetes e chanfros (rolling-ball sobre arestas; depende das booleanas).
3. Loft e sweep genérico (perfis NURBS, trajetória com frames de
   Frenet/RMF).
4. Transformação profunda de sólidos (clonagem geometria+topologia).
5. STEP: recortes genéricos (faces trimadas em superfícies curvas com
   p-curves), instâncias/assemblies.
6. IGES B-Rep (entidades 186/514/510/508) e leitura.
7. Aceleração espacial (BVH) para tesselação/classificação em cenas
   grandes.

## Uso independente

```
pip install numpy
cp -r nucleok/ seu_projeto/
```

## Contribuidores

<a href="https://github.com/Manolo789/nucleok/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Manolo789/nucleok" />
</a>
