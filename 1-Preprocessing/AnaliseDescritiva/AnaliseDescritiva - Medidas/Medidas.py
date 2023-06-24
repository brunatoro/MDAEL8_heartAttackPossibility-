import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Faz a leitura do arquivo
    input_file = 'DataMiningSamples/0-Datasets/heartClear.data'
    names = ['idade', 'sexo', 'dor no peito', 'pressão arterial em repouso', 'colesterol sérico', 'açucar no sangue em jejum', 'resultados eletrocardiográficos em repouso', 'frequência cardíaca máxima', 'angina induzida por exercício', 'Depressão de ST ', 'pico do segmento ST', 'nº de vasos sanguíneos principais', 'talassemia', 'resultado']
    target = 'idade'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas
    
    # Medidas de tendência central
    media = df[target].mean()
    mediana = df[target].median()
    moda = df[target].mode()[0]

    # Medidas de dispersão
    variancia = df[target].var()
    desvio_padrao = df[target].std()
    coeficiente_variacao = (desvio_padrao / media) * 100
    amplitude = df[target].max() - df[target].min()

    # Medidas de posição relativa
    Q1 = df[target].quantile(0.25)
    Q2 = df[target].quantile(0.5)
    Q3 = df[target].quantile(0.75)

    print("\n-----------------------------------------\n")
    print("Média: {:.2f}".format(media))
    print("Mediana: {:.2f}".format(mediana))
    print("Moda : {:.2f}".format(moda))
    print("\n-----------------------------------------\n")
    print("Variância : {:.2f}".format(variancia))
    print("Desvio padrão : {:.2f}".format(desvio_padrao))
    print("Coeficiente de variação (%): {:.2f}".format(coeficiente_variacao))
    print("Amplitude: {:.2f}".format(amplitude))
    print("\n-----------------------------------------\n")
    print('Z Score:\n{}\n'.format((df[target] - df[target].mean())/df[target].std())) # Z Score
    print("Primeiro quartil: {:.2f}".format(Q1))
    print("Segundo quartil (Mediana): {:.2f}".format(Q2))
    print("Terceiro quartil: {:.2f}".format(Q3))
    print("\n-----------------------------------------\n")
    
    print("\nMedidas de Associação\n")
    print('Covariância: \n{}\n'.format(df.cov())) # Covariância
    print('\nCorrelação: \n{}'.format(df.corr())) # Correlação
 
    # Plota o histograma dos valores da coluna Glucose
    #plt.hist(df['Glucose'], bins=20)
    #plt.title('Distribuição de Glucose')
    #plt.xlabel('Glucose')
    #plt.ylabel('Frequência')
    #plt.show()

if __name__== '__main__':
    main()