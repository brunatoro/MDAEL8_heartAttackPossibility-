import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lendo o arquivo csv
def main():
    # Faz a leitura do arquivo
    
    input_file = 'DataMiningSamples/0-Datasets/heartClear.data'
    names = ['idade', 'sexo', 'dor no peito', 'pressão arterial em repouso', 'colesterol sérico', 'açucar no sangue em jejum', 'resultados eletrocardiográficos em repouso', 'frequência cardíaca máxima', 'angina induzida por exercício', 'Depressão de ST ', 'pico do segmento ST', 'nº de vasos sanguíneos principais', 'talassemia', 'resultado']
    target = 'Resultado'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas  

    # Criando uma nova coluna 'idade_categoria'
    binsIdade = [29, 40, 50, 60, 77] # limites das categorias
    labelsIdade = ['29-39', '40-49', '50-59', '60-77'] # rótulos das categorias
    df['conjunto_idade'] = pd.cut(df['Idade'], bins=binsIdade, labels=labelsIdade)

    # Criando uma nova coluna 'imc_categoria'
    binsImc = [0, 18.5, 25, 30, 35, 40] # limites das categorias
    labelsImc = ['Abaixo do Peso', 'Normal', 'Pré-obesidade', 'Obesidade Classe I', 'Obesidade Classe II'] # rótulos das categorias
    df['imc_categoria'] = pd.cut(df['IMC'], bins=binsImc, labels=labelsImc)

    # Criando uma nova coluna 'insulina_categoria'
    binsInsulina = [0, 140, 199, 200] # limites das categorias
    labelsInsulina = ['Saudável', 'Resistência à Insulina', 'Possível Diabético'] # rótulos das categorias
    df['insulina_categoria'] = pd.cut(df['Insulina'], bins=binsInsulina, labels=labelsInsulina)

     # Classificando a coluna 'Glucose' de acordo com os critérios fornecidos
    binsGlucose = [0, 99, 125, 200, float('inf')] # limites das categorias
    labelsGlucose = ['Glicemia de jejum normal', 'Glicemia de jejum alterada', 'Pré-Diabetes', 'Alerta'] # rótulos das categorias
    df['glucose_categoria'] = pd.cut(df['Glucose'], bins=binsGlucose, labels=labelsGlucose)

    # Classificando a coluna 'Glucose' de acordo com critério específico para Glicemia de jejum baixa ou hipoglicemia
     #df.loc[df['Glucose'] <= 70, 'glucose_categoria'] = 'Glicemia de jejum baixa ou hipoglicemia'

     # Classificando a coluna 'pressao Arterial' de acordo com os critérios fornecidos
    binsPressaoArterial = [0, 80, 85, 90, 99, 100, 109, 110, float('inf')] # limites das categorias
    labelsPressaoArterial = ['Normal limítrofe', 'Normal limítrofe', 'Hipertensão leve', 'Hipertensão leve', 'Hipertensão moderada', 'Hipertensão moderada', 'Hipertensão grave', 'Hipertensão sistólica isolada', 'Hipertensão sistólica isolada'] # rótulos das categorias
    df['pressao_arterial_categoria'] = pd.cut(df['pressao Arterial'], bins=binsPressaoArterial, labels=labelsPressaoArterial[:-1], ordered=False)

    # Classificando a coluna 'Expessura da Pele' de acordo com os critérios fornecidos
    binsExpessuraPele = [0, 5, 9, 10, 15, 16, 20, float('inf')] # limites das categorias
    labelsExpessuraPele = ['Muito fina', 'Fina', 'Fina', 'Moderada', 'Moderada', 'Espessa', 'Muito espessa'] # rótulos das categorias
    df['expessura_pele_categoria'] = pd.cut(df['Expessura da Pele'], bins=binsExpessuraPele, labels=labelsExpessuraPele, ordered=False)


    # Classificando a coluna 'Número de Gestações' de acordo com os critérios fornecidos
    binsGestacoes = [-1, 0, 1, 2, float('inf')] # limites das categorias
    labelsGestacoes = ['Nulípara', 'Primípara', 'Secundípara', 'Multípara'] # rótulos das categorias
    df['numero_gestacoes_categoria'] = pd.cut(df['Número Gestações'], bins=binsGestacoes, labels=labelsGestacoes)

    # Mapeando a coluna 'Resultado' para categorias 'Não Diabético' e 'Diabético'
    df['Resultado'] = df['Resultado'].map({0: 'Não Diabético', 1: 'Diabético'})

    # Classificando a coluna 'Função Pedigree Diabete' de acordo com os critérios fornecidos
    binsPedigree = [0, 1.0, 2.0, float('inf')] # limites das categorias
    labelsPedigree = ['Baixa predisposição genética', 'Moderada predisposição genética', 'Alta predisposição genética'] # rótulos das categorias
    df['funcao_pedigree_categoria'] = pd.cut(df['Função Pedigree Diabete'], bins=binsPedigree, labels=labelsPedigree)

        
 # Criando gráfico de barras para a categoria 'conjunto_idade'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='conjunto_idade', hue='Resultado')
    plt.title('Distribuição de Resultados por Faixa Etária')
    plt.xlabel('Faixa Etária')
    plt.ylabel('Número de Pessoas')
    plt.savefig('grafico_faixa_etaria.png')  # Salva o gráfico em um arquivo PNG

    # Criando gráfico de barras para a categoria 'imc_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='imc_categoria', hue='Resultado')
    plt.title('Distribuição de Resultados por Categoria de IMC')
    plt.xlabel('Categoria de IMC')
    plt.ylabel('Número de Pessoas')
    plt.savefig('grafico_categoria_imc.png')  # Salva o gráfico em um arquivo PNG

    # Criando gráfico de barras para a categoria 'insulina_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='insulina_categoria', hue='Resultado')
    plt.title('Distribuição de Resultados por Categoria de Insulina')
    plt.xlabel('Categoria de Insulina')
    plt.ylabel('Número de Pessoas')
    plt.savefig('grafico_categoria_insulina.png')  # Salva o gráfico em um arquivo PNG

    # Criando gráfico de barras para a categoria 'glucose_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='glucose_categoria', hue='Resultado')
    plt.title('Distribuição de Resultados por Categoria de Glicemia de Jejum')
    plt.xlabel('Categoria de Glicemia de Jejum')
    plt.ylabel('Número de Pessoas')
    plt.savefig('grafico_categoria_glicemia.png')  # Salva o gráfico em um arquivo PNG

    # Criando gráfico de barras para a categoria 'pressaoArterial_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='pressao_arterial_categoria', hue='Resultado')
    plt.title('Distribuição de Resultados por Categoria de Pressão Arterial')
    plt.xlabel('Categoria de Pressão Arterial')
    plt.ylabel('Número de Pessoas')
    plt.savefig('grafico_categoria_pressao_arterial.png')  # Salva o gráfico em um arquivo PNG

    # Criando gráfico de barras para a categoria 'expessura_pele_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='expessura_pele_categoria', hue='Resultado')
    plt.title('Distribuição de Resultados por Categoria de Espessura de Pele')
    plt.xlabel('Categoria de Espessura de Pele')
    plt.ylabel('Número de Pessoas')
    plt.savefig('grafico_categoria_expessura_pele.png')  # Salva o gráfico em um arquivo PNG

    # Criando gráfico de barras para a categoria 'numero_gestacoes_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='numero_gestacoes_categoria', hue='Resultado')
    plt.title('Distribuição de Resultados por Categoria de Número de Gestações')
    plt.xlabel('Categoria de Número de Gestações')
    plt.ylabel('Número de Pessoas')
    plt.savefig('grafico_categoria_numero_gestacoes.png')  # Salva o gráfico em um arquivo PNG

    # Criando gráfico de barras para a categoria 'funcao_pedigree_categoria'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='funcao_pedigree_categoria', hue='Resultado')
    plt.title('Distribuição de Resultados por Categoria de Função Pedigree Diabete')
    plt.xlabel('Categoria de Função Pedigree Diabete')
    plt.ylabel('Número de Pessoas')
    plt.savefig('grafico_categoria_funcao_pedigree.png')  # Salva o gráfico em um arquivo PNG

    # Lista de todas as colunas do DataFrame
    colunas = ['conjunto_idade', 'imc_categoria', 'insulina_categoria', 'glucose_categoria',
            'pressao_arterial_categoria', 'expessura_pele_categoria', 'numero_gestacoes_categoria',
            'funcao_pedigree_categoria']

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