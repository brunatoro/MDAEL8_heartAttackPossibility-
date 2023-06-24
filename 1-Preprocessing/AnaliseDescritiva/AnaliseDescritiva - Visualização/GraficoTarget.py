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
                     names=names)   # Nome das colunas                      

    label = ['Maiores chances de infarto', 'Menores chances de infarto']
    cores = ['#20B2AA', '#FF69B4']  # Tons suaves de rosa e azul
    MaioresChances = df['resultado'].value_counts()[1]
    MenoresChances = df['resultado'].value_counts()[0]
    total = MaioresChances + MenoresChances
    y = np.array([MaioresChances, MenoresChances])
    
    # Calcula as porcentagens
    porcentagem_maiores = (MaioresChances * 100) / total
    porcentagem_menores = (MenoresChances * 100) / total
    
    # Exibe os resultados numéricos
    total_text = "Total: {}\n".format(total)
    maiores_text = "Maiores: {} ({:.2f}%)\n".format(MaioresChances, porcentagem_maiores)
    menores_text = "Menores: {} ({:.2f}%)\n".format(MenoresChances, porcentagem_menores)
    #print(total_text + maiores_text + menores_text)
    
    # Cria o gráfico de pizza com as porcentagens e os valores inteiros
    porcentagens = ["{} ({:.2f}%)".format(value, (value * 100) / total) for value in y]
    plt.pie(y, labels=label, colors=cores, autopct='%1.1f%%', startangle=90)
    plt.title('Total de Resultados\n' + maiores_text + menores_text)
    plt.legend(porcentagens)
    plt.show()

if __name__ == "__main__":
    main()
