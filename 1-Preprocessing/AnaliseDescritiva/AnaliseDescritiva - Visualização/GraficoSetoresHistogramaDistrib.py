import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lendo o arquivo csv
df = pd.read_csv('DataMiningSamples/0-Datasets/heartClear.data')

# Criando uma nova coluna 'idade_categoria'
binsIdade = [29, 40, 50, 60, 77] # limites das categorias
labelsIdade = ['29-39', '40-49', '50-59', '60-77'] # rótulos das categorias
df['conjunto_idade'] = pd.cut(df['idade'], bins=binsIdade, labels=labelsIdade)

# Loop pelas colunas
for coluna in df.columns:
    # Histograma
    plt.figure()
    if coluna == 'idade':
        sns.histplot(data=df, x='conjunto_idade')
        plt.title(f'Histograma da coluna {coluna}')
    plt.savefig(f'DataMiningSamples/5-AnaliseDescritiva/Graficos2/Histograma/historgrama_{coluna}.png')
    plt.close()

    # Gráfico de setores
    plt.figure()
    if coluna == 'idade':
        df['conjunto_idade'].value_counts().plot(kind='pie')
        plt.title(f'Gráfico de setores da coluna {coluna}') 
    plt.savefig(f'DataMiningSamples/5-AnaliseDescritiva/Graficos2/Setores/grafico_setores_{coluna}.png')
    plt.close()

    # Dispersão  
    plt.figure()
    sns.scatterplot(data=df, x=coluna, y='resultado')
    plt.title(f'Dispersão da coluna {coluna} em relação à coluna alvo')
    plt.savefig(f'DataMiningSamples/5-AnaliseDescritiva/Graficos2/Dispersao/dispersao_{coluna}.png')
    plt.close()