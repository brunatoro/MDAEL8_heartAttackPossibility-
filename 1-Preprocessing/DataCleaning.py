import pandas as pd
import numpy as np

def main():
    # Faz a leitura do arquivo
    names = ['idade',
             'sexo',
             'dor no peito',
             'pressão arterial em repouso',
             'colesterol sérico',
             'açucar no sangue em jejum',
             'resultados eletrocardiográficos em repouso',
             'frequência cardíaca máxima',
             'angina induzida por exercício',
             'Depressão de ST ',
             'pico do segmento ST',
             'nº de vasos sanguíneos principais',
             'talassemia',
             'resultado'] 
    features = ['idade',
             'sexo',
             'dor no peito',
             'pressão arterial em repouso',
             'colesterol sérico',
             'açucar no sangue em jejum',
             'resultados eletrocardiográficos em repouso',
             'frequência cardíaca máxima',
             'angina induzida por exercício',
             'Depressão de ST ',
             'pico do segmento ST',
             'nº de vasos sanguíneos principais',
             'talassemia',
             'resultado'] #caracteristicas que quero trabalhar
    output_file = 'DataMiningSamples/0-Datasets/heartClear.data' #nome do arquivo que fica armazenado as mudanças
    input_file = 'DataMiningSamples/0-Datasets/heart.data'
    df = pd.read_csv(input_file,         # Nome do arquivo com dados #df =  data framing
                     names = names,      # Nome das colunas 
                     usecols = features, # Define as colunas que serão  utilizadas
                     na_values='?')      # Define que ? será considerado valores ausentes
    
    df_original = df.copy()
    # Imprime as 20 primeiras linhas do arquivo
    print("PRIMEIRAS 20 LINHAS\n")
    print(df.head(20))
    print("\n")        

    # Imprime informações sobre dos dados
    print("INFORMAÇÕES GERAIS DOS DADOS\n")
    print(df.info())
    print("\n")
    
    # Imprime uma analise descritiva sobre dos dados
    print("DESCRIÇÃO DOS DADOS\n")
    print(df.describe())
    print("\n")
    
    # Imprime a quantidade de valores faltantes por coluna
    print("VALORES FALTANTES\n")
    print(df.isnull().sum())
    print("\n")    
    
    columns_missing_value = df.columns[df.isnull().any()]
    print(columns_missing_value)
    method = 'median' # number or median or mean or mode
    
    for c in columns_missing_value:
        UpdateMissingValues(df, c, method)
    
    print(df.describe())
    print("\n")
    print(df.head(15))
    print(df_original.head(15))
    print("\n")
    
    # Salva arquivo com o tratamento para dados faltantes
    df.to_csv(output_file, header=False, index=False)  
    

def UpdateMissingValues(df, column, method="median", number=0):
    if method == 'number':
        # Substituindo valores ausentes por um número
        df[column].fillna(number, inplace=True)

    elif method == 'median':
        # Substituindo valores ausentes pela mediana 
        median = df['dor no peito'].median()
        df[column].fillna(median, inplace=True)
        median2 = df['pico do segmento ST'].median()
        df[column].fillna(median2, inplace=True)

    elif method == 'mean':
        # Substituindo valores ausentes pela média
        mean = round(df[column].mean(), 0)  # Modificação na linha da média
        df[column].fillna(mean, inplace=True)

    elif method == 'mode':
        # Substituindo valores ausentes pela moda
        mode = df[column].mode()[0]
        df[column].fillna(mode, inplace=True)
    


if __name__ == "__main__":
    main()