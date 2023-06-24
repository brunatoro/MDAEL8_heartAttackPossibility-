import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # Faz a leitura do arquivo
input_file = '0-DataMiningSamples/Datasets/heartClear.data'
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
target = 'resultado'
df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas   
# Separar as variáveis dependentes da variável de saída
X = df.drop('resultado', axis=1)
y = df['resultado']

# Aplicar a normalização Min-Max às variáveis dependentes
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Criar um novo dataframe com as variáveis dependentes normalizadas e a variável de saída
df_norm = pd.DataFrame(X_norm, columns=X.columns)
df_norm['resultado'] = y

# Salvar a base de dados normalizada em um novo arquivo csv
df_norm.to_csv('dados_coracao_normalizados.csv', index=False)