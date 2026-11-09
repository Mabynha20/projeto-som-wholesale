# TAREFAS.md — O que cada um precisa fazer

> Procure o seu nome, veja **o que você recebe**, **o que você entrega** e siga o passo a passo.
> O contexto completo e as decisões do projeto estão no [PROJETO.md](PROJETO.md).

## Visão geral

| Parte | Pessoa | Etapa | Notebook | Depende de | Status |
|---|---|---|---|---|---|
| 1 | **Amabilly** | Base, EDA e pré-processamento | `01_Amabilly_EDA_preprocessamento.ipynb` | — | ✅ Concluído |
| 2 | **Roger** | Treino do SOM e ajuste de parâmetros | `02_Roger_treino_som.ipynb` | Parte 1 | ⬜ Pode começar |
| 3 | **Erica** | Visualizações do mapa | `03_Erica_visualizacoes.ipynb` | Parte 2 | ⬜ |
| 4 | **Rafael** | Clusters no mapa e outliers | `04_Rafael_clusters_outliers.ipynb` | Parte 2 | ⬜ |
| 5 | **Herbe** | Comparação com K-Means e PCA | `05_Herbe_comparacao.ipynb` | Partes 2 e 4 | ⬜ |
| 6 | **Ruan** | Perfis, aplicações e consolidação | `06_Ruan_perfis.ipynb` | Partes 4 e 5 | ⬜ |

**Fluxo dos arquivos entre as partes:**

```
Amabilly ──► dados/X_scaled.csv, dados/dados_preprocessados.csv
   │
Roger ─────► dados/som.pkl (mapa treinado)
   │
   ├──► Erica  (figuras do mapa)
   └──► Rafael ──► dados/clusters.csv (cluster e outlier de cada cliente)
                     │
                     ├──► Herbe ──► tabela de comparação (ARI, silhouette)
                     └──► Ruan  ──► perfis, conclusão, relatório e slides
```

Cada pessoa escreve no relatório a parte que fez e apresenta essa parte (cerca de 2–3 minutos).

---

## Regras para todo mundo

1. **Antes de começar, sempre rode `git pull`.**
2. **Trabalhe só no seu notebook** (nome da tabela acima). Não edite o notebook dos outros.
3. **Use `random_seed=42` / `random_state=42`** em tudo (SOM, K-Means, PCA).
4. **Salve as figuras em `figuras/`** com o número da sua parte no começo (ex.: `03_umatrix.png`).
5. **Carregue os dados assim** (não refaça o pré-processamento):
   ```python
   import pandas as pd, pickle
   X_scaled = pd.read_csv("dados/X_scaled.csv").values
   df = pd.read_csv("dados/dados_preprocessados.csv")   # tem Canal, Regiao e os gastos originais
   GASTOS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
   ```
6. **Paleta dos gráficos** (para manter o visual igual): azul `#2a78d6`, laranja `#eb6834`, cinza `#52514e`. Horeca = azul, Varejo = laranja.
7. **Ao terminar:**
   - escreva embaixo de cada gráfico uma interpretação curta;
   - atualize o seu status aqui e no `PROJETO.md`, adicionando uma linha no histórico;
   - rode `git add`, `git commit` e `git push`.

---

## Parte 1: Amabilly · Base, EDA e pré-processamento ✅

**Entregou:**
- `dados/X_scaled.csv`: 440 × 6, com `log1p` e depois `StandardScaler`.
- `dados/dados_preprocessados.csv`: a base original com as colunas `Canal` e `Regiao` em texto e as colunas `log_*`.
- Figuras 01 a 07 em `figuras/`.

**Achados que ajudam as próximas partes:**
- **Dois blocos de produtos:**
  - `Grocery`/`Detergents_Paper`/`Milk`, com correlação de 0,92 entre Grocery e Detergents_Paper. É o bloco do Varejo.
  - `Fresh`/`Frozen`, um bloco mais fraco. É o bloco do Horeca.
- **`Channel` bem separado:** a mediana do Varejo gasta cerca de 14,6× mais em `Detergents_Paper`.
- **Outliers:** não foram removidos. Os candidatos são clientes com gasto quase zero em `Fresh`, `Delicassen` ou `Detergents_Paper`.

---

## Parte 2: Roger · Treino do SOM

**Recebe:** `dados/X_scaled.csv`
**Entrega:** o mapa treinado em `dados/som.pkl` e uma tabela de testes com QE/TE.

**Passo a passo:**
1. Instale a biblioteca: `pip install minisom`.
2. Use um mapa **10×10** (regra: neurônios ≈ 5·√N; com N = 440, cerca de 105).
3. Teste `sigma` em {1.0, 1.5, 2.0} e `learning_rate` em {0.3, 0.5}, o que dá 6 combinações.
4. Compare as combinações pelo **erro de quantização (QE)**, quanto menor melhor, e pelo **erro topográfico (TE)**, quanto menor melhor.
5. Escolha a melhor configuração e justifique em uma tabela.
6. Treine o modelo final e **salve em `dados/som.pkl`**.

```python
from minisom import MiniSom
import pickle

resultados = []
for sigma in [1.0, 1.5, 2.0]:
    for lr in [0.3, 0.5]:
        som = MiniSom(10, 10, X_scaled.shape[1], sigma=sigma, learning_rate=lr, random_seed=42)
        som.pca_weights_init(X_scaled)
        som.train_random(X_scaled, 5000)
        resultados.append({"sigma": sigma, "learning_rate": lr,
                           "QE": som.quantization_error(X_scaled),
                           "TE": som.topographic_error(X_scaled)})
pd.DataFrame(resultados).sort_values(["TE", "QE"])

# depois de escolher a melhor configuração:
som = MiniSom(10, 10, 6, sigma=..., learning_rate=..., random_seed=42)
som.pca_weights_init(X_scaled)
som.train_random(X_scaled, 5000)
with open("dados/som.pkl", "wb") as f:
    pickle.dump(som, f)
```

**Como os outros carregam o seu mapa:**
```python
with open("dados/som.pkl", "rb") as f:
    som = pickle.load(f)
```

**Para o relatório:** o que é o SOM (em uma frase), por que um mapa 10×10, a tabela de testes e os parâmetros escolhidos.
**Assim que o `som.pkl` subir, avise o grupo**, porque a Erica e o Rafael dependem dele.

---

## Parte 3: Erica · Visualizações do mapa

**Recebe:** `dados/som.pkl`, `dados/X_scaled.csv` e `dados/dados_preprocessados.csv`
**Entrega:** U-Matrix, component planes, hit map e o mapa com `Channel` sobreposto, cada um com interpretação.

**Passo a passo:**
1. **U-Matrix:** use `som.distance_map()`.
   - Regiões claras são **fronteiras** entre grupos.
   - Regiões escuras são **grupos**.
2. **Component planes:** faça um mapa por variável com `som.get_weights()[:, :, i]`, para `i` de 0 a 5, na ordem de `GASTOS`. Eles mostram em que região do mapa cada produto é alto.
3. **Hit map:** use `som.activation_response(X_scaled)`. Ele mostra quantos clientes caíram em cada neurônio.
4. **Sobreposição do `Channel`:** para cada cliente, pegue `som.winner(x)` e marque no mapa com a cor do canal (Horeca em azul, Varejo em laranja), por cima da U-Matrix.
5. Escreva uma interpretação curta de cada gráfico.
   - Exemplo: "a região superior concentra Varejo e coincide com valores altos de Grocery".

**Dica:** o mapa deve separar um lado "mercearia/limpeza" (Varejo) de um lado "frescos/congelados" (Horeca), como apareceu na EDA. Confirme ou desminta essa hipótese.

---

## Parte 4: Rafael · Clusters e outliers

**Recebe:** `dados/som.pkl` e `dados/X_scaled.csv`
**Entrega:** o k escolhido, o mapa colorido por cluster, a lista de outliers e o arquivo **`dados/clusters.csv`**.

**Passo a passo:**
1. Rode o K-Means sobre os **pesos dos neurônios** (e não sobre os clientes):
   ```python
   pesos = som.get_weights().reshape(-1, X_scaled.shape[1])   # 100 neurônios × 6
   ```
2. Escolha o k pelo **cotovelo** e pelo **silhouette**, testando k = 2 a 6 com `random_state=42`.
3. Atribua a cada cliente o cluster do seu BMU, isto é, do neurônio vencedor, obtido com `som.winner(x)`.
4. Identifique os **outliers** pelo erro de quantização individual:
   ```python
   erro = np.linalg.norm(X_scaled - som.quantization(X_scaled), axis=1)
   outliers = np.where(erro > np.percentile(erro, 95))[0]
   ```
   Confira também se há neurônios isolados na U-Matrix.
5. Pinte o mapa com as cores dos clusters.
6. **Salve `dados/clusters.csv`**, que o Herbe e o Ruan vão usar. Ele deve ter uma linha por cliente, na mesma ordem da base:

   | coluna | conteúdo |
   |---|---|
   | `bmu_x`, `bmu_y` | posição do neurônio vencedor |
   | `cluster` | cluster do SOM (0 a k−1) |
   | `erro_quant` | erro de quantização individual |
   | `outlier` | True/False (acima do percentil 95) |

**Assim que o `clusters.csv` subir, avise o grupo** e diga qual k foi escolhido.

---

## Parte 5: Herbe · Comparação com K-Means e PCA

**Recebe:** `dados/X_scaled.csv` e `dados/clusters.csv` (com o k do Rafael)
**Entrega:** os gráficos PCA 2D, o ARI, o silhouette e uma tabela de prós e contras.

**Passo a passo:**
1. Rode o **K-Means direto em `X_scaled`**, com o **mesmo k** do Rafael e `random_state=42`.
2. Rode o **PCA com 2 componentes** e faça dois gráficos de dispersão lado a lado:
   - um colorido pelos clusters do SOM (coluna `cluster` do `clusters.csv`);
   - outro colorido pelos clusters do K-Means.
3. Calcule as métricas:
   - `adjusted_rand_score(cluster_som, cluster_kmeans)`, que mede o quanto os dois agrupamentos concordam;
   - `silhouette_score(X_scaled, ...)` para cada agrupamento;
   - a variância explicada pelo PCA (`pca.explained_variance_ratio_`).
4. Discuta as diferenças:
   - o SOM preserva a topologia e mostra transições entre grupos;
   - o K-Means só particiona;
   - o PCA é linear.
5. Monte uma **tabela de prós e contras** (SOM × K-Means × PCA).

---

## Parte 6: Ruan · Perfis, aplicações e consolidação

**Recebe:** `dados/dados_preprocessados.csv`, `dados/clusters.csv` e os resultados de todos
**Entrega:** os perfis nomeados, as aplicações, a conclusão, o relatório final e os slides.

**Passo a passo:**
1. Junte os clusters à base e calcule o perfil de cada grupo, **em valores reais** (colunas originais, não as padronizadas):
   ```python
   base = df.join(pd.read_csv("dados/clusters.csv"))
   base.groupby("cluster")[GASTOS].median()      # mediana é mais robusta que média aqui
   pd.crosstab(base["cluster"], base["Canal"], normalize="index")
   ```
2. Faça um gráfico de **radar ou de barras** com o perfil de cada cluster.
3. **Dê nome aos perfis.** Hipóteses a confirmar com os dados:
   - "Restaurantes e hotéis": alto Fresh/Frozen
   - "Mercados e varejo": alto Grocery/Detergents_Paper
   - "Pequenos compradores": gasto baixo em tudo
   - "Clientes atípicos": os outliers
4. Discuta as **aplicações**: marketing direcionado, logística, estoque e detecção de clientes anômalos.
5. Escreva a **conclusão e as limitações**: base pequena, sensibilidade a hiperparâmetros e escolha do k.
6. **Consolide** o relatório e os slides com a parte de cada um.

**Enquanto espera as outras partes:** adiante a estrutura dos slides e a teoria do SOM.

---

## Cronograma

| Fase | Quem | Depende de |
|---|---|---|
| Dias 1–2 | Amabilly | — ✅ |
| Dias 2–3 | Roger | `X_scaled.csv` (já disponível) |
| Dias 3–5 | Erica e Rafael em paralelo; Herbe começa o K-Means e o PCA | `som.pkl` |
| Dias 4–5 | Herbe termina a comparação | `clusters.csv` |
| Dias 5–6 | Ruan | `clusters.csv` e a comparação do Herbe |
| Dia 7 | Todos | Revisão e ensaio da apresentação |
