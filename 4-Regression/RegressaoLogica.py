# Importando as Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
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

check = df['resultado'].value_counts(normalize=True) * 100
print(check)

# Criando o Modelo - Scikit-learn

# Matriz de Variáveis
X = df.drop(columns='resultado')
y = df['resultado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print('Dados de Treino: ', len(X_train))
print('Dados de Teste: ', len(X_test))

# Regressão Logistica
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Avaliação do Modelo
Previsao = logistic_model.predict(X_test)
print('Matriz Confusão: \n', confusion_matrix(y_test, Previsao), '\n')

f, ax = plt.subplots()
ax = sns.heatmap(confusion_matrix(y_test, Previsao), annot=True)
ax.set_title('Matriz Confusão', fontsize=10, loc='left', pad=13)
plt.tight_layout()
plt.show()

# Métricas de Classificação - Relatório de Clasfficação
print('Relatório de Clasfficação - Regressão Logistica: \n\n', classification_report(y_test, Previsao))

# Previsão Balanceada
print('Score (Treino): ', round(logistic_model.score(X_train, y_train), 2))
print('Score (Teste): ', round(logistic_model.score(X_test, y_test), 2))

# Validação Cruzada
Validacao_Cruzada = cross_val_score(logistic_model, X, y, cv=5)
print(Validacao_Cruzada)

# Random Forest Classifier
forest_model = RandomForestClassifier(max_depth=3)
forest_model.fit(X_train, y_train)

Previsao_forest = forest_model.predict(X_test)
print('Relatório de Clasfficação - Random Forest: \n\n', classification_report(y_test, Previsao_forest))

# SVM
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)

Previsao_svm = svm_model.predict(X_test)
print('Relatório de Clasfficação - SVM: \n\n', classification_report(y_test, Previsao_svm))