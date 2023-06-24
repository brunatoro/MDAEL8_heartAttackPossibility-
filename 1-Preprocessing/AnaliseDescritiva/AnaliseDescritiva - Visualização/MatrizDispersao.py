import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Faz a leitura do arquivo
    input_file = 'DataMiningSamples/0-Datasets/heartClear.data'
    names = ['Idade',
             'Sexo',
             'Dor no peito',
             'Pressão arterial',
             'Colesterol',
             'Glicemia jejum',
             'Eletrocardiograma em rp',
             'Frequência cardíaca máx',
             'Angina induzida por exercício',
             'Depressão de ST ',
             'Pico do segmento ST',
             'Nº de v.s principais',
             'Talassemia',
             'Resultado'] 

    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas
    
    # Plota um gráfico de correlação
    corr = df.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='coolwarm', annot=True)
    plt.title('Matriz de Correlação')
    plt.show()
    

if __name__ == '__main__':
    main()