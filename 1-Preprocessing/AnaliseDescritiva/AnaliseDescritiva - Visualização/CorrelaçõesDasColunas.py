import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lendo o arquivo csv
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

    df = pd.read_csv(input_file, names = names)

    # 1 - Criando uma nova coluna 'idade_categoria'
    binsIdade = [29, 40, 50, 60, 77] # limites das categorias
    labelsIdade = ['29-39', '40-49', '50-59', '60-77'] # rótulos das categorias
    df['df_idade'] = pd.cut(df['idade'], bins=binsIdade, labels=labelsIdade)

    # 2 - Criando uma nova coluna 'sexo'
    labelsSexo = ['Mulher', 'Homem'] # rótulos das categorias
    df['df_sexo'] = df['sexo'].map({0: labelsSexo[0], 1: labelsSexo[1]})

    # 3 - Criação da coluna 'tipo_de_dor_no_peito'
    binsDorNoPeito = [0, 1, 2, 3, 4]  # limites das categorias
    labelsDorNoPeito = ['Angina típica', 'Angina atípica', 'Dor não anginosa', 'Assintomático']  # rótulos das categorias
    df['df_tipo_de_dor_no_peito'] = pd.cut(df['dor no peito'], bins=binsDorNoPeito, labels=labelsDorNoPeito)

    # 4 - Classificando a coluna 'pressão arterial em repouso' de acordo com os critérios fornecidos
    binsPressãoArterial = [130, 139, 159, 179, 180, float('inf')] # limites das categorias
    labelsPressaoArterial = ['94-130', '131-139', '140-159', '160-179', '180'] # rótulos das categorias
    df['df_pressão_arterial'] = pd.cut(df['pressão arterial em repouso'], bins=binsPressãoArterial, labels=labelsPressaoArterial)

    # 5 - Classificando a coluna 'colesterol sérico' de acordo com os critérios fornecidos
    binsColesterolSerico = [199, 230, 564, float('inf')] # limites das categorias
    labelsColesterolSerico = ['Normal', 'Limítrofe', 'Alto'] # rótulos das categorias
    df['df_colesterol_sérico'] = pd.cut(df['colesterol sérico'], bins=binsColesterolSerico, labels=labelsColesterolSerico)

    # 6 - Criando uma nova coluna 'açucar no sangue em jejum'
    labelsAçucar = ['Verdadeiro', 'Falso'] # rótulos das categorias
    df['df_açucar_no_sangue_em_jejum'] = df['açucar no sangue em jejum'].map({0: labelsAçucar[0], 1: labelsAçucar[1]})

    # 7 - Criação da coluna 'resultados_eletrocardiograficos'
    binsResultadosEletro = [-1, 0, 1, 2]  # limites das categorias
    labelsResultadosEletro = ['Normal', 'Anormalidade da onda ST-T', 'Mostrando hipertrofia ventricular esquerda']  # rótulos das categorias
    df['df_resultados_eletrocardiográficos_em_repouso'] = pd.cut(df['resultados eletrocardiográficos em repouso'], bins=binsResultadosEletro, labels=labelsResultadosEletro)

    # 8 - Criando uma nova coluna 'batimentos_categoria'
    binsBatimentos = [0, 59, 100, 202] # limites das categorias
    labelsBatimentos = ['Baixos', 'Normais', 'Altos'] # rótulos das categorias
    df['df_frequência_cardíaca_máxima'] = pd.cut(df['frequência cardíaca máxima'], bins=binsBatimentos, labels=labelsBatimentos)

    # 9 - Criação da coluna 'angina induzida por exercício'
    labelsAngina = ['Não', 'Sim']  # rótulos das categorias
    df['df_angina_induzida_por_exercício'] = df['angina induzida por exercício'].map({0: labelsAngina[0], 1: labelsAngina[1]})

    # 11 - Criação da coluna 'inclinacao_segmento_ST'
    binsInclinacaoST = [0, 1, 2, 3]  # limites das categorias
    labelsInclinacaoST = ['Ascendente', 'Plano', 'Descendente']  # rótulos das categorias
    df['df_pico_do_segmento_ST'] = pd.cut(df['pico do segmento ST'], bins=binsInclinacaoST, labels=labelsInclinacaoST)

    # 12 -  Criando a coluna 'vasos_coloridos'
    binsVasos = [-1, 0, 1, 2, 3]
    labelsVasos = ['Não identificado', '1 vaso', '2 vasos', '3 vasos']
    df['df_nº_de_vasos_sanguíneos_principais'] = pd.cut(df['nº de vasos sanguíneos principais'], bins=binsVasos, labels=labelsVasos)

    # 13 - Criação da coluna 'thal'
    binsThal = [0, 1, 2, 3]
    labelsThal = ['Defeito fixo', 'Fluxo sanguíneo normal', 'Defeito reversível']
    df['df_talassemia'] = pd.cut(df['talassemia'], bins=binsThal, labels=labelsThal)


    #14 - Alvo
    df['df_resultado'] = df['resultado'].map({0: 'Menor chance de ataque do cardiaco', 1: 'Maior chance de ataque cardiaco'})

    # Definindo a paleta de cores
    cores = ['#4169E1', '#DB7093']  # Substitua as cores conforme necessário

    # Define a paleta de cores do Seaborn
    sns.set_palette(cores)    

    # Criando gráfico de barras para a categoria 'idade_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='df_idade', hue='df_resultado')
    plt.title('Relação dos Resultados por Idade')
    plt.xlabel('Idade')
    plt.ylabel('Número de Pessoas')
    plt.savefig('1-Preprocessing/AnaliseDescritiva/GraficosHeartDissease/grafico_faixa_etaria.png')

    # Criando gráfico de barras para a categoria 'sexo_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='df_sexo', hue='df_resultado')
    plt.title('Relação dos Resultados por Sexo')
    plt.xlabel('Sexo')
    plt.ylabel('Número de Pessoas')
    plt.savefig('1-Preprocessing/AnaliseDescritiva/GraficosHeartDissease/grafico_sexo.png')

    # Criando gráfico de barras para a categoria 'tipo_de_dor_no_peito_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='df_tipo_de_dor_no_peito', hue='df_resultado')
    plt.title('Relação dos por Tipo de Dor no Peito')
    plt.xlabel('Tipo de Dor no Peito')
    plt.ylabel('Número de Pessoas')
    plt.savefig('1-Preprocessing/AnaliseDescritiva/GraficosHeartDissease/grafico_dor_no_peito.png')

    # Criando gráfico de barras para a categoria 'pressão_arterial_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='df_pressão_arterial', hue='df_resultado')
    plt.title('Relação dos Resultados por Pressão Arterial')
    plt.xlabel('Pressão Arterial')
    plt.ylabel('Número de Pessoas')
    plt.savefig('1-Preprocessing/AnaliseDescritiva/GraficosHeartDissease/grafico_pressao_arterial.png')

    # Criando gráfico de barras para a categoria 'colesterol_sérico_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='df_colesterol_sérico', hue='df_resultado')
    plt.title('Relação dos Resultados por Colesterol Sérico')
    plt.xlabel('Colesterol Sérico')
    plt.ylabel('Número de Pessoas')
    plt.savefig('1-Preprocessing/AnaliseDescritiva/GraficosHeartDissease/grafico_colesterol_serico.png')

    # Criando gráfico de barras para a categoria 'açucar_no_sangue_em_jejum_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='df_açucar_no_sangue_em_jejum', hue='df_resultado')
    plt.title('Relação dos Resultados por Açúcar no Sangue em Jejum')
    plt.xlabel('Açúcar no Sangue em Jejum')
    plt.ylabel('Número de Pessoas')
    plt.savefig('1-Preprocessing/AnaliseDescritiva/GraficosHeartDissease/grafico_acucar_sangue_jejum.png')

    # Criando gráfico de barras para a categoria 'resultados_eletrocardiográficos_em_repouso_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='df_resultados_eletrocardiográficos_em_repouso', hue='df_resultado')
    plt.title('Relação dos Resultados por Resultados Eletrocardiográficos em Repouso')
    plt.xlabel('Resultados Eletrocardiográficos em Repouso')
    plt.ylabel('Número de Pessoas')
    plt.savefig('1-Preprocessing/AnaliseDescritiva/GraficosHeartDissease/grafico_resultados_eletrocardiograficos.png')

    # Criando gráfico de barras para a categoria 'frequência_cardíaca_máxima_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='df_frequência_cardíaca_máxima', hue='df_resultado')
    plt.title('Relação dos Resultados por Frequência Cardíaca Máxima')
    plt.xlabel('Frequência Cardíaca Máxima')
    plt.ylabel('Número de Pessoas')
    plt.savefig('1-Preprocessing/AnaliseDescritiva/GraficosHeartDissease/grafico_frequencia_cardiaca_maxima.png')

    # Criando gráfico de barras para a categoria 'angina_induzida_por_exercício_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='df_angina_induzida_por_exercício', hue='df_resultado')
    plt.title('Relação dos Resultados por Angina Induzida por Exercício')
    plt.xlabel('Angina Induzida por Exercício')
    plt.ylabel('Número de Pessoas')
    plt.savefig('1-Preprocessing/AnaliseDescritiva/GraficosHeartDissease/grafico_angina_induzida_exercicio.png')

    # Criando gráfico de barras para a categoria 'pico_do_segmento_ST_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='df_pico_do_segmento_ST', hue='df_resultado')
    plt.title('Relação dos Resultados por Pico do Segmento ST')
    plt.xlabel('Pico do Segmento ST')
    plt.ylabel('Número de Pessoas')
    plt.savefig('1-Preprocessing/AnaliseDescritiva/GraficosHeartDissease/grafico_pico_segmento_ST.png')

    # Criando gráfico de barras para a categoria 'nº_de_vasos_sanguíneos_principais_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='df_nº_de_vasos_sanguíneos_principais', hue='df_resultado')
    plt.title('Relação dos Resultados por Número de Vasos Sanguíneos Principais')
    plt.xlabel('Número de Vasos Sanguíneos Principais')
    plt.ylabel('Número de Pessoas')
    plt.savefig('1-Preprocessing/AnaliseDescritiva/GraficosHeartDissease/grafico_numero_vasos_sanguineos.png')

    # Criando gráfico de barras para a categoria 'talassemia_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='df_talassemia', hue='df_resultado')
    plt.title('Relação dos Resultados por Talassemia')
    plt.xlabel('Talassemia')
    plt.ylabel('Número de Pessoas')
    plt.savefig('1-Preprocessing/AnaliseDescritiva/GraficosHeartDissease/grafico_talassemia.png')


    # Lista de todas as colunas do DataFrame
    colunas = ['df_idade', 'df_sexo', 'df_tipo_de_dor_no_peito', 'df_pressão_arterial',
            'df_colesterol_sérico', 'df_açucar_no_sangue_em_jejum', 
            'df_resultados_eletrocardiográficos_em_repouso', 'df_frequência_cardíaca_máxima',
            'df_angina_induzida_por_exercício', 'df_pico_do_segmento_ST',
            'df_nº_de_vasos_sanguíneos_principais', 'df_talassemia']


    # Loop for para criar os gráficos
    for coluna1 in colunas:
        for coluna2 in colunas:
            if coluna1 != coluna2:  # Evita criar gráficos repetidos
                # Criação do gráfico de barras
                plt.figure(figsize=(10, 6))
                sns.countplot(data=df, x=coluna1, hue=coluna2)
                plt.title(f'Distribuição de Resultados por {coluna1} e {coluna2}')
                plt.xlabel(coluna1)
                plt.ylabel('Número de Pessoas')
                plt.legend(title=coluna2)
                plt.savefig(f'1-Preprocessing/Graficos/grafico_{coluna1}_e_{coluna2}.png')  # Salva o gráfico em um arquivo PNG
                #plt.show()  # Mostra o gráfico


if __name__ == "__main__":
    main()