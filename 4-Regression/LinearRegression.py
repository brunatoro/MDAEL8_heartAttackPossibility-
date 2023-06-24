import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# carrega os dados
input_file = '0-Datasets/heartClear.data'
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
df = pd.read_csv(input_file, names=names)

target_names = ['Menor chance', 'Maior chance']

# separa em set de treino e teste
X = df.drop('resultado', axis=1)
y = df['resultado']

# normaliza as variáveis de entrada
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# utiliza a seleção de variáveis RFE
linreg = LinearRegression()
selector = RFE(linreg, n_features_to_select=5)
selector.fit(X_norm, y)
X_sel = selector.transform(X_norm)

# separa em set de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.33, random_state=42)

# utiliza a validação cruzada
scores = cross_val_score(linreg, X_sel, y, cv=5)
print('Acurácia com validação cruzada: %.2f +/- %.2f' % (scores.mean(), scores.std()))

# treina o modelo
regr = LinearRegression()
regr.fit(X_train, y_train)

# avalia o modelo
r2_train = regr.score(X_train, y_train)
r2_test = regr.score(X_test, y_test)
print('R2 no set de treino: %.2f' % r2_train)
print('R2 no set de teste: %.2f' % r2_test)

# plota o gráfico de dispersão e a linha de regressão
plt.scatter(X_test[:,0], y_test, color='blue')
y_pred = regr.predict(X_test)
plt.plot(X_test[:,0], y_pred, color='red', linewidth=2)
plt.xlabel('Número Gestações')
plt.ylabel('Resultado')
plt.title('Regressão Linear')
plt.show()

# faz as predições no set de teste
y_pred = regr.predict(X_test)

# plota o gráfico de dispersão com a reta de regressão
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Valor Real')
plt.ylabel('Valor Previsto')
plt.title('Regressão Linear')
plt.show()

# faz as previsões para os dados de teste
y_pred = regr.predict(X_test)

# cria o gráfico de dispersão
plt.scatter(y_test, y_pred)

# traça a linha de regressão
x = range(0, 350, 10)
y = x
plt.plot(x, y, color='red')

# adiciona rótulos e título
plt.xlabel('Valor Real')
plt.ylabel('Valor Previsto')
plt.title('Gráfico de Dispersão e Regressão Linear')
y_pred = regr.predict(X_test)
abs_error = mean_absolute_error(y_pred, y_test)
print('Erro absoluto no set de treino: %.2f' % abs_error)

# exibe o gráfico
plt.show()