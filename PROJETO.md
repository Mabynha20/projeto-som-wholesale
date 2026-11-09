# PROJETO.md — Projeto Final de Aprendizado Não Supervisionado (SOM)

> **Para que serve este arquivo:** registro central do projeto. Quem retomar o trabalho (inclusive o Claude em uma nova sessão) deve ler este arquivo primeiro e atualizá-lo sempre que algo mudar: decisões, status de cada parte e o histórico no final.

---

## 1. Contexto

- **Disciplina:** Aprendizado Não Supervisionado (projeto final em grupo).
- **Opções dadas pelo professor:**
  1. Autoencoder: compressão, reconstrução e detecção de anomalias.
  2. **SOM (Self-Organizing Map): segmentação e visualização** ← **ESCOLHIDO**
  3. GAN: geração sintética e avaliação de qualidade.
- **Por que o SOM:** é o mais fácil e rápido. Treina em segundos sem GPU, a biblioteca `minisom` já entrega U-Matrix, BMUs e métricas, e a comparação com K-Means e PCA é simples no `sklearn`. O Autoencoder seria a 2ª opção; a GAN foi descartada por ter treino instável e avaliação subjetiva.

## 2. Enunciado do professor (Projeto 2: SOM)

- Escolher uma base com múltiplas variáveis numéricas ou vetorizadas.
- Padronizar os dados e treinar um Self-Organizing Map.
- Interpretar regiões do mapa, distâncias, agrupamentos e outliers.
- Comparar os achados com K-Means, PCA ou outra técnica auxiliar.
- Nomear os perfis encontrados e discutir possíveis aplicações.

**Entrega exigida:** mapa treinado, U-Matrix, clusters/perfis e interpretação dos grupos.

## 3. Decisões técnicas

| Item | Decisão | Status |
|---|---|---|
| Base de dados | **Wholesale Customers** (UCI/Kaggle): 440 clientes, 6 variáveis de gasto anual (Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen), mais `Channel` e `Region` | Em uso na Parte 1; falta confirmação formal do grupo |
| Pré-processamento | `np.log1p` nas 6 variáveis de gasto, depois `StandardScaler` | ✅ Feito (Parte 1) |
| Outliers no pré-processamento | **Não remover.** Ficam todos os 440 clientes; a detecção é feita depois pelo SOM (Rafael) | ✅ Definido (Parte 1) |
| Biblioteca | `minisom` + `scikit-learn` + `pandas`/`matplotlib`/`seaborn` | Sugerido |
| Tamanho do mapa | 10×10 (regra: neurônios ≈ 5·√N; com N = 440, cerca de 105) | Sugerido |
| Hiperparâmetros iniciais | `sigma=1.5`, `learning_rate=0.5`, `pca_weights_init`, 5000 iterações | A ajustar (Roger) |
| Semente | `random_seed=42` em tudo (SOM e K-Means) | Definido |
| Ambiente | Notebooks por parte (o primeiro é `01_Amabilly_EDA_preprocessamento.ipynb`); roda local ou no Colab (baixa a base da UCI se o CSV não estiver em `dados/`) | Em uso |
| Repositório | **GitHub público:** https://github.com/Mabynha20/projeto-som-wholesale (branch `main`). Cada pessoa cria o próprio notebook numerado (`02_Roger_...`, `03_Erica_...` etc.), faz `git pull` antes de começar e atualiza o PROJETO.md ao terminar. Combinados completos no README | ✅ Criado |
| Paleta dos gráficos | Azul `#2a78d6` (série única e Horeca), laranja `#eb6834` (Varejo), cinza `#52514e` (referências). Manter nas outras partes | Definido (Parte 1) |
| Métricas | Erro de quantização, erro topográfico, silhouette, Adjusted Rand Index | Definido |
| Outliers (no SOM) | Erro de quantização individual > percentil 95 e neurônios isolados na U-Matrix | Definido |
| Clusters | K-Means sobre os **pesos dos neurônios**, com k escolhido pelo cotovelo e pelo silhouette (testar k = 2 a 6) | Definido, k a definir (Rafael) |

## 4. Integrantes e divisão de tarefas

| # | Pessoa | Etapa | Entrega | Status |
|---|---|---|---|---|
| 1 | **Amabilly** | Base, análise exploratória e pré-processamento | `X_scaled` pronto e EDA com gráficos | ✅ Concluído (falta revisão do grupo) |
| 2 | **Roger** | Treino do SOM e ajuste de parâmetros | Mapa treinado e tabela de testes (QE/TE) | ⬜ Não iniciado (já desbloqueado) |
| 3 | **Erica** | Visualizações do mapa | U-Matrix, component planes, hit map, sobreposição de `Channel` | ⬜ Não iniciado |
| 4 | **Rafael** | Clusters no mapa e outliers | k escolhido, mapa colorido por cluster, lista de outliers | ⬜ Não iniciado |
| 5 | **Herbe** | Comparação com K-Means e PCA | Gráficos PCA 2D, ARI, silhouette, tabela de prós e contras | ⬜ Não iniciado |
| 6 | **Ruan** | Perfis, aplicações e consolidação | Perfis nomeados, aplicações, conclusão, relatório e slides | ⬜ Não iniciado |

Cada pessoa escreve no relatório a parte que fez e apresenta essa parte (cerca de 2–3 minutos cada).

Legenda de status: ⬜ Não iniciado · 🟨 Em andamento · ✅ Concluído

## 5. Arquivos do projeto

```
projeto final/
├── PROJETO.md                              ← este arquivo
├── README.md / requirements.txt / .gitignore
├── 01_Amabilly_EDA_preprocessamento.ipynb  ← Parte 1 (executado, com saídas)
├── dados/
│   ├── Wholesale customers data.csv        ← base original (UCI)
│   ├── X_scaled.csv                        ← 440×6, log1p + StandardScaler → ENTRADA DO SOM
│   └── dados_preprocessados.csv            ← base original + Canal/Regiao em texto + colunas log_*
└── figuras/                                ← 01 a 07 (PNG) da EDA, para relatório e slides
```

Como as outras partes carregam os dados:
```python
import pandas as pd
X_scaled = pd.read_csv("dados/X_scaled.csv").values      # entrada do SOM / K-Means / PCA
df = pd.read_csv("dados/dados_preprocessados.csv")       # rótulos (Canal, Regiao) e valores originais
```
A ordem das linhas é a mesma em todos os arquivos (índice 0 a 439).

## 6. Passo a passo por etapa

### 6.1 Amabilly: dados e pré-processamento ✅
Feito em `01_Amabilly_EDA_preprocessamento.ipynb`. **Principais achados (para o relatório e os slides):**
- **Qualidade:** a base está limpa. Não há ausentes, duplicatas nem gastos ≤ 0.
- **Assimetria forte:** em todas as variáveis a média é maior que a mediana. A assimetria vai de 2,6 (`Fresh`) a 11,2 (`Delicassen`). Depois do `log1p`, fica entre −1,6 e −0,2.
- **Escalas diferentes:** `Fresh` tem média de cerca de 12 mil e `Delicassen`, de cerca de 1,5 mil. Por isso o `StandardScaler`.
- **Correlação, dois blocos:**
  - `Grocery` × `Detergents_Paper` = 0,92 (0,80 após o log) e `Milk` × `Grocery` = 0,73. É o bloco "mercearia/varejo".
  - `Fresh` × `Frozen` = 0,35. É o bloco "perecíveis/Horeca", mais fraco.
- **Channel:** 298 Horeca (68%) e 142 Varejo (32%), desbalanceado.
  - O Varejo gasta cerca de 14,6× mais em `Detergents_Paper`, 4,6× mais em `Grocery` e 3,6× mais em `Milk` (medianas).
  - O Horeca gasta cerca de 1,6× mais em `Fresh` e 1,9× mais em `Frozen`.
  - Deve aparecer bem separado no SOM.
- **Outliers:** pelo IQR, 42 clientes têm ao menos um valor fora do intervalo após o log. Eles foram mantidos. Os candidatos mais fortes são clientes com gasto quase zero em `Fresh`, `Delicassen` ou `Detergents_Paper` (valores de 3 a 10).
- **Hipótese para o Ruan:** o SOM deve separar um eixo "mercearia/limpeza" de um eixo "frescos/congelados".

### 6.2 Roger: treino do SOM
1. Instalar com `pip install minisom`.
2. Mapa 10×10; testar `sigma` em {1.0, 1.5, 2.0} e `learning_rate` em {0.3, 0.5}.
3. Comparar as combinações pelo erro de quantização (QE) e pelo erro topográfico (TE).
4. Escolher a melhor configuração e documentar a escolha numa tabela.

```python
from minisom import MiniSom
som = MiniSom(10, 10, X_scaled.shape[1], sigma=1.5, learning_rate=0.5, random_seed=42)
som.pca_weights_init(X_scaled)
som.train_random(X_scaled, 5000)
print(som.quantization_error(X_scaled), som.topographic_error(X_scaled))
```

### 6.3 Erica: visualizações
1. **U-Matrix:** `som.distance_map()`. Regiões claras indicam fronteiras entre grupos; regiões escuras indicam grupos.
2. **Component planes:** `som.get_weights()[:, :, i]`, um mapa por variável.
3. **Hit map:** `som.activation_response(X_scaled)`.
4. Sobrepor `Channel` (Horeca vs. Varejo) no mapa para validar a separação. A coluna `Canal` já está em texto em `dados_preprocessados.csv`.
5. Escrever uma interpretação curta de cada gráfico.

### 6.4 Rafael: clusters e outliers
1. Rodar K-Means sobre os pesos: `som.get_weights().reshape(-1, n_features)`.
2. Escolher k pelo cotovelo e pelo silhouette (k = 2 a 6).
3. Atribuir a cada cliente o cluster do seu BMU (`som.winner(x)`).
4. Identificar os outliers pelo erro de quantização individual:

```python
erro = np.linalg.norm(X_scaled - som.quantization(X_scaled), axis=1)
outliers = np.where(erro > np.percentile(erro, 95))[0]
```
5. Pintar o mapa com as cores dos clusters.

### 6.5 Herbe: comparação
1. Rodar K-Means direto em `X_scaled`, com o mesmo k.
2. Rodar PCA em 2 componentes e fazer dois gráficos de dispersão (cores do SOM e cores do K-Means).
3. Calcular `adjusted_rand_score` entre os dois agrupamentos e o silhouette de cada um.
4. Discutir: o SOM preserva a topologia e mostra transições; o K-Means só particiona; o PCA é linear.
5. Montar uma tabela de prós e contras.

### 6.6 Ruan: perfis e consolidação
1. Calcular o perfil médio de cada cluster (`df.groupby("cluster").mean()`) e fazer um gráfico de radar ou de barras.
2. Nomear os perfis. Hipóteses a confirmar com os dados:
   - "Restaurantes e hotéis" (alto Fresh/Frozen)
   - "Mercados e varejo" (alto Grocery/Detergents_Paper)
   - "Pequenos compradores" (gasto baixo em tudo)
   - "Clientes atípicos" (outliers)
3. Discutir aplicações: marketing direcionado, logística, estoque, detecção de clientes anômalos.
4. Escrever a conclusão e as limitações (tamanho da base, sensibilidade a hiperparâmetros, escolha de k).
5. Consolidar o notebook final e os slides.

## 7. Cronograma e dependências

| Fase | Quem | Depende de |
|---|---|---|
| Dias 1–2 | Amabilly | Ninguém (✅ concluído) |
| Dias 2–3 | Roger | Amabilly (`X_scaled`, já disponível) |
| Dias 3–5 | Erica, Rafael e Herbe em paralelo | Roger (mapa treinado) |
| Dias 5–6 | Ruan | Rafael (clusters) e Herbe (comparação) |
| Dia 7 | Todos | Revisão e ensaio da apresentação |

Enquanto espera as outras etapas, o Ruan pode adiantar a estrutura dos slides e a teoria do SOM.

## 8. Checklist da entrega final

- [x] Base escolhida, EDA e pré-processamento (`X_scaled`)
- [ ] Mapa SOM treinado (com hiperparâmetros documentados)
- [ ] U-Matrix com interpretação
- [ ] Component planes e hit map
- [ ] Clusters e perfis nomeados
- [ ] Outliers identificados e discutidos
- [ ] Comparação com K-Means e PCA (ARI e silhouette)
- [ ] Aplicações possíveis
- [ ] Conclusão e limitações
- [ ] Notebook final rodando do início ao fim
- [ ] Slides da apresentação

## 9. Próximos passos, pendências e dúvidas em aberto

- **Amabilly:** adicionar os 5 colegas como colaboradores no GitHub (Settings → Collaborators, pelo usuário do GitHub de cada um) e mandar o link do repositório no grupo. Escrever a parte dela no relatório a partir da seção 6.1. Montar os 2–3 slides com as figuras `02`, `04`, `05` e `06`.
- **Roger:** já pode começar o treino do SOM.
- Confirmar a base de dados (Wholesale Customers) com o grupo.
- Confirmar o prazo de entrega e a data da apresentação com o professor. **Data: ___ (preencher)**
- Definir o formato da entrega (notebook, PDF ou slides?).
- Coletar o usuário do GitHub de cada colega: Roger ___, Erica ___, Rafael ___, Herbe ___, Ruan ___.
- Quem precisar do SOM treinado: o Roger salva o modelo em `dados/som.pkl` (combinado no README).
- Decidir se tudo vai para um único notebook no final (o Ruan consolida) ou se ficam notebooks separados por parte.

## 10. Histórico de atualizações

| Data | O que mudou | Por quê |
|---|---|---|
| 2026-11-05 | Projeto 2 (SOM) escolhido; base, divisão de tarefas, passo a passo e cronograma definidos. Nenhum código feito ainda. | Planejamento inicial. |
| 2026-11-09 | **Parte 1 (Amabilly) concluída.** Base baixada da UCI para `dados/`. Criado e executado `01_Amabilly_EDA_preprocessamento.ipynb` (qualidade, descritivas, histogramas, boxplots, correlação, gastos por canal, log1p, padronização). Gerados `X_scaled.csv`, `dados_preprocessados.csv` e 7 figuras em `figuras/`. Achados registrados na seção 6.1. | Desbloquear o treino do SOM (Roger). |
| 2026-11-09 | Decisão: **outliers não são removidos** no pré-processamento. | A base é pequena (440), clientes grandes são um perfil real e a detecção de outliers é tarefa do SOM (Rafael). |
| 2026-11-09 | Criado o repositório público no GitHub (https://github.com/Mabynha20/projeto-som-wholesale), com README (como usar e combinados do grupo), `requirements.txt` e `.gitignore`. Também substitui o Colab como forma principal de compartilhar. | Dar acesso aos colegas para continuarem as partes deles, com histórico e sem sobrescrever o trabalho do outro. |
| 2026-11-09 | Definida a paleta padrão dos gráficos e a estrutura de pastas (`dados/`, `figuras/`). Notebook roda local ou no Colab. | Manter um visual consistente entre as partes e facilitar o compartilhamento. |
