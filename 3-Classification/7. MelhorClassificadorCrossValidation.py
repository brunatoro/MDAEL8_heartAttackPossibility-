import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
from keras.models import Sequential
from keras.layers import Dense
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
from tabulate import tabulate
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier

# Carregar a base de dados
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
             'talassemia',
             ] #caracteristicas que quero trabalhar
target = 'resultado'

data = pd.read_csv(input_file, names=names)

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.25, random_state=0)

# Aplicar o Smote para balancear a base de dados
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN
knn_classifier = KNeighborsClassifier()
knn_grid = {'n_neighbors': [3, 5, 7],
            'metric': ['euclidean', 'manhattan']}
knn_grid_search = GridSearchCV(knn_classifier, knn_grid, scoring='accuracy', cv=5)
knn_grid_search.fit(X_train, y_train)
knn_classifier = knn_grid_search.best_estimator_

# SVM
svm_classifier = SVC(kernel='poly', C=1)

# Redes Neurais
rna_classifier = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)

# Árvore de Decisão
dt_classifier = DecisionTreeClassifier(max_depth=5)  # Definir uma profundidade máxima para a árvore

classifiers = {
    'KNN': knn_classifier,
    'SVM': svm_classifier,
    'RNA': rna_classifier,
    'Árvore de Decisão': dt_classifier
}

table_data = []
metrics = ['accuracy', 'f1_weighted', 'precision_macro']
for classifier_name, classifier in classifiers.items():
    cv_results = cross_validate(classifier, X_train, y_train, scoring=metrics, cv=10)
    accuracy_scores = cv_results['test_accuracy']
    f1_scores = cv_results['test_f1_weighted']
    precision_scores = cv_results['test_precision_macro']
    accuracy_mean = np.mean(accuracy_scores)
    f1_mean = np.mean(f1_scores)
    precision_mean = np.mean(precision_scores)
    classifier_data = [classifier_name, round(accuracy_mean, 2), round(f1_mean, 2), round(precision_mean, 2)]
    table_data.append(classifier_data)

# Printar tabela com os resultados
table_headers = ["Classificador", "Acurácia", "F1-Score", "Precision"]
print(tabulate(table_data, headers=table_headers, tablefmt="grid"))

# Printar matriz de confusão
for classifier_name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    confusion_matrix_data = confusion_matrix(y_test, predictions)
    print(f"\nMatriz de Confusão - {classifier_name}:")
    print(tabulate(confusion_matrix_data, headers=range(2), tablefmt="grid"))

# Gráfico comparativo de desempenho dos classificadores em 2D
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(classifiers))
width = 0.2

rects1 = ax.bar(x, [row[1] for row in table_data], width, label='Acurácia')
rects2 = ax.bar(x + width, [row[2] for row in table_data], width, label='F1-Score')
rects3 = ax.bar(x + 2 * width, [row[3] for row in table_data], width, label='Precision')

ax.set_title('Comparativo de Desempenho dos Classificadores')
ax.set_xlabel('Classificadores')
ax.set_ylabel('Desempenho')
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels([row[0] for row in table_data])
ax.legend()

# Anotar as métricas no gráfico
for rect1, rect2, rect3 in zip(rects1, rects2, rects3):
    height1 = rect1.get_height()
    height2 = rect2.get_height()
    height3 = rect3.get_height()
    ax.annotate(f'Acurácia: \n{round(height1, 2)}', xy=(rect1.get_x() + rect1.get_width() / 2, height1),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom')
    ax.annotate(f'F1-Score: \n{round(height2, 2)}', xy=(rect2.get_x() + rect2.get_width() / 2, height2),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom')
    ax.annotate(f'Precision: \n{round(height3, 2)}', xy=(rect3.get_x() + rect3.get_width() / 2, height3),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Gráfico comparativo de desempenho dos classificadores em 2D (horizontal)
fig, ax = plt.subplots(figsize=(10, 6))

y = np.arange(len(classifiers))
height = 0.2

rects1 = ax.barh(y, [row[1] for row in table_data], height, label='Acurácia')
rects2 = ax.barh(y + height, [row[2] for row in table_data], height, label='F1-Score')
rects3 = ax.barh(y + 2 * height, [row[3] for row in table_data], height, label='Precision')

ax.set_title('Comparativo de Desempenho dos Classificadores')
ax.set_xlabel('Desempenho')
ax.set_ylabel('Classificadores')
ax.set_yticks(y + 1.5 * height)
ax.set_yticklabels([row[0] for row in table_data])
ax.legend()

# Anotar as métricas no gráfico
for rect1, rect2, rect3 in zip(rects1, rects2, rects3):
    width1 = rect1.get_width()
    width2 = rect2.get_width()
    width3 = rect3.get_width()
    ax.annotate(f'Acurácia: {round(width1, 2)}', xy=(width1, rect1.get_y() + rect1.get_height() / 2),
                xytext=(3, 0), textcoords="offset points",
                ha='left', va='center')
    ax.annotate(f'F1-Score: {round(width2, 2)}', xy=(width2, rect2.get_y() + rect2.get_height() / 2),
                xytext=(3, 0), textcoords="offset points",
                ha='left', va='center')
    ax.annotate(f'Precision: {round(width3, 2)}', xy=(width3, rect3.get_y() + rect3.get_height() / 2),
                xytext=(3, 0), textcoords="offset points",
                ha='left', va='center')

plt.tight_layout()
plt.show()