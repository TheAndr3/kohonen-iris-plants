# ==============================================================================
# Trabalho Acadêmico: Redes de Kohonen (SOM) aplicadas ao Dataset Iris Plants
# ==============================================================================
# Objetivo: Aplicar Mapas Auto-Organizáveis (SOM) ao dataset Iris para
# descobrir regularidades e agrupar os dados de forma não supervisionada.
#
# Bibliotecas utilizadas: numpy, pandas, scikit-learn, matplotlib, minisom
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from minisom import MiniSom
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo (salva em arquivo sem necessidade de GUI)
import matplotlib.pyplot as plt

# Semente aleatória para reprodutibilidade dos resultados
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ==============================================================================
# PASSO 1: CARREGAMENTO E DIVISÃO DO DATASET
# ==============================================================================
print("=" * 70)
print("PASSO 1: Carregamento e Divisão do Dataset")
print("=" * 70)

# Definindo os nomes das colunas do dataset Iris
nomes_colunas = [
    'comprimento_sepala',  # sepal length (cm)
    'largura_sepala',      # sepal width (cm)
    'comprimento_petala',  # petal length (cm)
    'largura_petala',      # petal width (cm)
    'classe'               # espécie da planta
]

# Carregando o dataset a partir do arquivo iris.data (mesmo diretório do script)
df = pd.read_csv('iris.data', header=None, names=nomes_colunas)

# Removendo linhas vazias que possam existir no final do arquivo
df.dropna(inplace=True)

print(f"Total de amostras carregadas: {len(df)}")
print(f"Classes encontradas: {df['classe'].unique()}")
print(f"Distribuição das classes:\n{df['classe'].value_counts()}\n")

# Separando os atributos preditivos (X) das classes reais (y)
# ATENÇÃO: As classes NÃO serão passadas à rede durante o treinamento.
# Elas serão usadas apenas para validação posterior.
X = df[nomes_colunas[:-1]].values  # 4 atributos numéricos
y = df['classe'].values            # rótulos reais (apenas para validação)

# Normalização dos dados para o intervalo [0, 1]
# Isso é importante para o SOM, pois evita que atributos com escalas
# maiores dominem o cálculo de distância.
normalizador = MinMaxScaler()
X_normalizado = normalizador.fit_transform(X)

# Divisão dos dados em treinamento (70%) e teste (30%)
# stratify=y garante que a proporção das classes seja mantida em ambos os conjuntos
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_normalizado, y,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=y
)

print(f"Amostras de treinamento: {len(X_treino)}")
print(f"Amostras de teste: {len(X_teste)}")
print()

# ==============================================================================
# PASSO 2: TREINAMENTO DAS REDES KOHONEN (SOM)
# ==============================================================================
print("=" * 70)
print("PASSO 2: Treinamento das Redes Kohonen (SOM)")
print("=" * 70)

# Configurações das três topologias de grade 2D a serem testadas
tamanhos_grade = [5, 10, 15]

# Número de atributos de entrada (4 para o Iris)
num_atributos = X_treino.shape[1]

# Número de iterações de treinamento
num_iteracoes = 1000

# ---------------------------------------------------------------
# PARÂMETRO RESTRITO: Taxa de aprendizado fixa em 0.01
# ---------------------------------------------------------------
taxa_aprendizado = 0.01

# Dicionário para armazenar as redes treinadas
redes_som = {}

for tamanho in tamanhos_grade:
    print(f"\n--- Treinando SOM {tamanho}x{tamanho} ---")

    # ---------------------------------------------------------------
    # PARÂMETRO RESTRITO: Raio de vizinhança fixo (sem decaimento)
    # O parâmetro 'sigma' define o raio inicial da vizinhança.
    # A função 'decay_function' controla como sigma e learning_rate
    # decaem ao longo das iterações.
    # Aqui, usamos uma função lambda que RETORNA SEMPRE O VALOR INICIAL,
    # desabilitando completamente o decaimento do raio e da taxa de aprendizado.
    # ---------------------------------------------------------------
    raio_vizinhanca = tamanho / 2  # raio proporcional ao tamanho da grade

    som = MiniSom(
        x=tamanho,
        y=tamanho,
        input_len=num_atributos,
        sigma=raio_vizinhanca,
        learning_rate=taxa_aprendizado,
        neighborhood_function='gaussian',  # função de vizinhança gaussiana
        topology='rectangular',            # topologia retangular da grade
        random_seed=RANDOM_SEED,
        # Desabilitando o decaimento da taxa de aprendizado — permanece fixa em 0.01
        # O parâmetro 'decay_function' aceita callable: retornamos sempre o valor original
        decay_function=lambda lr, t, max_iter: lr
    )

    # ---------------------------------------------------------------
    # PARÂMETRO RESTRITO: Desabilitando o decaimento do raio (sigma)
    # A biblioteca minisom não aceita callable para sigma_decay_function,
    # então sobrescrevemos o atributo interno após a criação do objeto.
    # A lambda abaixo retorna sempre o sigma original, mantendo o raio fixo.
    # ---------------------------------------------------------------
    som._sigma_decay_function = lambda sigma, t, max_iter: sigma

    # Inicialização dos pesos com valores aleatórios
    som.random_weights_init(X_treino)

    # Treinamento da rede SOM
    # train_random: apresenta amostras aleatórias do conjunto de treinamento
    som.train_random(X_treino, num_iteracoes, verbose=True)

    # Armazenando a rede treinada
    redes_som[tamanho] = som
    print(f"SOM {tamanho}x{tamanho} treinado com sucesso!")

print()

# ==============================================================================
# PASSO 3: GERAÇÃO DA U-MATRIX (UNIFIED DISTANCE MATRIX)
# ==============================================================================
print("=" * 70)
print("PASSO 3: Geração da U-Matrix")
print("=" * 70)

# A U-Matrix mostra a distância média entre cada neurônio e seus vizinhos.
# Regiões com valores ALTOS (cores quentes) indicam fronteiras entre clusters.
# Regiões com valores BAIXOS (cores frias) indicam neurônios similares (mesmo cluster).

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('U-Matrix para as Três Topologias SOM', fontsize=16, fontweight='bold')

for idx, tamanho in enumerate(tamanhos_grade):
    som = redes_som[tamanho]

    # Extraindo a U-Matrix da rede treinada
    u_matrix = som.distance_map()

    # Plotando a U-Matrix como um mapa de calor
    ax = axes[idx]
    im = ax.pcolormesh(u_matrix.T, cmap='bone_r', edgecolors='gray', linewidth=0.5)
    ax.set_title(f'SOM {tamanho}x{tamanho}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Neurônios (eixo X)')
    ax.set_ylabel('Neurônios (eixo Y)')
    ax.set_aspect('equal')  # manter proporção quadrada

    # Barra de cores indicando a escala de distâncias
    plt.colorbar(im, ax=ax, label='Distância média')

plt.tight_layout()
plt.savefig('u_matrix_comparacao.png', dpi=150, bbox_inches='tight')
print("Gráfico da U-Matrix salvo em 'u_matrix_comparacao.png'")
# plt.show() — descomente esta linha se estiver executando em ambiente com GUI
print()

# ==============================================================================
# PASSO 4: APLICAÇÃO DO ALGORITMO K-MEANS NOS PESOS DA REDE
# ==============================================================================
print("=" * 70)
print("PASSO 4: Aplicação do K-means nos Pesos da Rede SOM")
print("=" * 70)

# REGRA OBRIGATÓRIA: k=3 (representando as 3 espécies de Iris)
# REGRA OBRIGATÓRIA: Distância Euclidiana (padrão do KMeans do scikit-learn)
k = 3

# Dicionário para armazenar os modelos K-means treinados
modelos_kmeans = {}

for tamanho in tamanhos_grade:
    som = redes_som[tamanho]

    # Extraindo os pesos (vetores de peso) de todos os neurônios da rede SOM
    # get_weights() retorna uma matriz de forma (tamanho, tamanho, num_atributos)
    pesos = som.get_weights()

    # Reorganizando para uma matriz 2D: (tamanho*tamanho, num_atributos)
    # Cada linha representa o vetor de pesos de um neurônio
    pesos_2d = pesos.reshape(-1, num_atributos)

    print(f"\n--- K-means para SOM {tamanho}x{tamanho} ---")
    print(f"Número de neurônios: {pesos_2d.shape[0]}")
    print(f"Dimensão dos pesos: {pesos_2d.shape[1]}")

    # Aplicando K-means nos pesos dos neurônios (NÃO nos dados originais)
    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_SEED,
        n_init=10,       # número de inicializações para robustez
        algorithm='lloyd' # algoritmo clássico de Lloyd (distância Euclidiana)
    )
    kmeans.fit(pesos_2d)

    modelos_kmeans[tamanho] = kmeans
    print(f"K-means aplicado com sucesso! Centroides encontrados: {k}")
    print(f"Inércia (soma das distâncias intra-cluster): {kmeans.inertia_:.4f}")

print()

# ==============================================================================
# PASSO 5: TESTE E VALIDAÇÃO DOS GRUPOS
# ==============================================================================
print("=" * 70)
print("PASSO 5: Teste e Validação dos Grupos")
print("=" * 70)

# Mapeamento numérico das classes para facilitar a comparação
mapa_classes = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
nomes_classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

for tamanho in tamanhos_grade:
    som = redes_som[tamanho]
    kmeans = modelos_kmeans[tamanho]

    print(f"\n{'='*60}")
    print(f"  RESULTADOS PARA TOPOLOGIA SOM {tamanho}x{tamanho}")
    print(f"{'='*60}")

    # Para cada amostra de teste, encontramos o neurônio vencedor (BMU)
    # e verificamos a qual cluster do K-means esse neurônio pertence.
    rotulos_preditos = []

    for amostra in X_teste:
        # Encontrando o BMU (Best Matching Unit) — o neurônio mais próximo
        bmu = som.winner(amostra)

        # Calculando o índice linear do neurônio na grade
        # (para buscar no resultado do K-means)
        indice_neuronio = bmu[0] * tamanho + bmu[1]

        # O cluster atribuído pelo K-means a esse neurônio
        cluster = kmeans.labels_[indice_neuronio]
        rotulos_preditos.append(cluster)

    rotulos_preditos = np.array(rotulos_preditos)

    # ---------------------------------------------------------------
    # Mapeamento dos clusters do K-means para as classes reais
    # ---------------------------------------------------------------
    # Como o K-means não sabe os nomes das classes, precisamos descobrir
    # qual cluster corresponde a qual classe. Para cada cluster, verificamos
    # qual classe real aparece com mais frequência.
    # ---------------------------------------------------------------
    mapeamento_cluster_classe = {}
    for cluster_id in range(k):
        # Índices das amostras de teste atribuídas a este cluster
        indices_cluster = np.where(rotulos_preditos == cluster_id)[0]

        if len(indices_cluster) > 0:
            # Contando quantas amostras de cada classe real caíram neste cluster
            classes_no_cluster = y_teste[indices_cluster]
            classe_mais_frequente = pd.Series(classes_no_cluster).mode()[0]
            mapeamento_cluster_classe[cluster_id] = classe_mais_frequente
        else:
            mapeamento_cluster_classe[cluster_id] = "Vazio"

    print(f"\nMapeamento dos clusters para as classes reais:")
    for cluster_id, classe in mapeamento_cluster_classe.items():
        print(f"  Cluster {cluster_id} -> {classe}")

    # ---------------------------------------------------------------
    # Exibindo a tabela de resultados detalhados
    # ---------------------------------------------------------------
    print(f"\n{'Classe Real':<20} | {'Cluster':<10} | {'Classe Predita':<20} | {'Correto?'}")
    print("-" * 75)

    acertos = 0
    total = len(X_teste)

    for i in range(total):
        classe_real = y_teste[i]
        cluster = rotulos_preditos[i]
        classe_predita = mapeamento_cluster_classe.get(cluster, "Desconhecido")
        correto = "SIM" if classe_real == classe_predita else "NAO"

        if classe_real == classe_predita:
            acertos += 1

        # Exibindo apenas os primeiros e últimos resultados para não poluir o terminal
        if i < 5 or i >= total - 5:
            print(f"{classe_real:<20} | {cluster:<10} | {classe_predita:<20} | {correto}")
        elif i == 5:
            print(f"{'...':<20} | {'...':<10} | {'...':<20} | ...")

    taxa_acerto = (acertos / total) * 100
    print(f"\n>>> RESULTADO: {acertos}/{total} amostras classificadas corretamente")
    print(f">>> TAXA DE ACERTO: {taxa_acerto:.1f}%")

    # ---------------------------------------------------------------
    # Detalhamento por classe
    # ---------------------------------------------------------------
    print(f"\nDetalhamento por classe:")
    for classe in nomes_classes:
        indices_classe = np.where(y_teste == classe)[0]
        total_classe = len(indices_classe)
        if total_classe > 0:
            acertos_classe = sum(
                1 for i in indices_classe
                if mapeamento_cluster_classe.get(rotulos_preditos[i], "") == classe
            )
            print(f"  {classe:<20}: {acertos_classe}/{total_classe} "
                  f"({(acertos_classe/total_classe)*100:.1f}% de acerto)")

print("\n" + "=" * 70)
print("Execução concluída com sucesso!")
print("=" * 70)
