import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# Carregar o conjunto de dados
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
             'talassemia']
target = 'resultado'
df = pd.read_csv(input_file, names=names)

# Dados de treinamento
X = df[features].values
y = df[target].values

# Normalização dos dados
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Criação do classificador SVM
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X_normalized, y)

# Função para plotar o hiperplano de separação em 3D
def plot_hyperplane(clf, ax):
    # Extrai os coeficientes e intercepto do hiperplano
    w = clf.coef_[0]
    a = -w[0] / w[1]
    
    # Determina as coordenadas do hiperplano
    xx = np.linspace(np.min(X_normalized[:, 0]), np.max(X_normalized[:, 0]), 10)
    yy = np.linspace(np.min(X_normalized[:, 1]), np.max(X_normalized[:, 1]), 10)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = (-w[0] * XX - w[1] * YY - clf.intercept_[0]) / w[2]
    
    # Plota o hiperplano em 3D
    ax.plot_surface(XX, YY, ZZ, alpha=0.5, color='gray')

# Plotagem dos pontos de dados e hiperplano em 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Rótulos das classes
labels = np.unique(y)
for label in labels:
    indices = np.where(y == label)
    ax.scatter(X_normalized[indices, 0], X_normalized[indices, 1], X_normalized[indices, 2], label=f'Classe {label}')

plot_hyperplane(clf, ax)

# Configurações do gráfico
ax.set_xlabel('Idade')
ax.set_ylabel('Pressão arterial em repouso')
ax.set_zlabel('Colesterol sérico')
ax.set_title('Classificação SVM em 3D StandardScaler')
ax.legend()
plt.show()
