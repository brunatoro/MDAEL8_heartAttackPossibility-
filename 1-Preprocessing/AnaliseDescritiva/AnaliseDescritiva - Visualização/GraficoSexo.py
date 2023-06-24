import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Faz a leitura do arquivo
    input_file = 'DataMiningSamples/0-Datasets/heartClear.data'
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

    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas                      

    label = ['Homens','Mulheres']
    cores = ['#20B2AA', '#FF69B4']  # Tons suaves de rosa e azul
    Homens = df['sexo'].value_counts()[1]
    Mulheres = df['sexo'].value_counts()[0]
    total = Homens + Mulheres
    y = np.array([Homens, Mulheres])
    porcentagens = ['Homens: {:.2f}%'.format((Homens*100)/total), 'Mulheres: {:.2f}%'.format((Mulheres*100)/total)]
    plt.pie(y , labels=porcentagens, colors=cores, autopct= lambda x: '{:.0f}'.format(x*y.sum()/100, startangle=90))
    plt.title('Total de Resultados Homens x Mulheres')
    plt.show() 
    print("Total de Homens: {}".format(Homens))
    print("Total de Mulheres: {}".format(Mulheres))
   

if __name__ == "__main__":
    main()
