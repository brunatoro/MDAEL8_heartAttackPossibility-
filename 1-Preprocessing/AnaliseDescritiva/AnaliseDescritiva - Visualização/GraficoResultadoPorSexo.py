import pandas as pd
import matplotlib.pyplot as plt


input_file = 'DataMiningSamples/0-Datasets/heartClear.data'
names = ['idade', 'sexo', 'dor no peito', 'pressão arterial em repouso', 'colesterol sérico', 'açucar no sangue em jejum', 'resultados eletrocardiográficos em repouso', 'frequência cardíaca máxima', 'angina induzida por exercício', 'Depressão de ST ', 'pico do segmento ST', 'nº de vasos sanguíneos principais', 'talassemia', 'resultado']
df = pd.read_csv(input_file, names=names)

# Contagem de homens e mulheres na base de dados
Homens = df['sexo'].value_counts()[1]
Mulheres = df['sexo'].value_counts()[0]

# Contagem de homens e mulheres com maiores chances de ataque cardíaco
HomensMaioresChances = df[(df['sexo'] == 1) & (df['resultado'] == 1)]['sexo'].count()
MulheresMaioresChances = df[(df['sexo'] == 0) & (df['resultado'] == 1)]['sexo'].count()

# Cálculo das porcentagens
total = Homens + Mulheres
homens_chances_percent = HomensMaioresChances / Homens * 100
mulheres_chances_percent = MulheresMaioresChances / Mulheres * 100

# Impressão dos resultados
print("Total de Homens: {}".format(Homens))
print("Total de Mulheres: {}".format(Mulheres))
print("Homens com Maiores Chances de Ataque Cardíaco: {} ({:.2f}%)".format(HomensMaioresChances, homens_chances_percent))
print("Mulheres com Maiores Chances de Ataque Cardíaco: {} ({:.2f}%)".format(MulheresMaioresChances, mulheres_chances_percent))

# Dados para o gráfico
labels = ['Homens', 'Mulheres']
sizes = [HomensMaioresChances, MulheresMaioresChances]
explode = (0, 0.1)  # Explodir o segundo pedaço do gráfico (Mulheres)

# Cores para cada pedaço do gráfico
colors = ['#ff9999', '#66b3ff']

# Configurações do gráfico de pizza
fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
       shadow=True, startangle=90)
ax.axis('equal')  # Faz o gráfico de pizza circular

# Título do gráfico
ax.set_title('Relação de Maiores Chances de Ataque Cardíaco\nHomens x Mulheres')

# Exibir o gráfico
plt.show()