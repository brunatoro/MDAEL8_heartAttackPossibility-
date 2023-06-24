import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

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

# Verificar se a base de dados é supervisionada ou não supervisionada
if target in data.columns:
    is_supervised = True
    target_values = data[target].unique()
    num_classes = len(target_values)
    class_counts = data[target].value_counts()

    print("A base de dados é supervisionada.")
    print("Número de classes:", num_classes)
    print("Classes:", target_values)
    print("Contagem de instâncias em cada classe:")
    print(class_counts)
else:
    is_supervised = False
    print("A base de dados é não supervisionada.")

# Dividir o conjunto de dados entre atributos de entrada e rótulos/target
X = data[features]
y = data[target]

# Dividir o conjunto de dados entre treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de árvore de decisão
model = DecisionTreeClassifier(max_leaf_nodes=5)
model.fit(X_train, y_train)

# Avaliar o modelo
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Acurácia do modelo:", accuracy)
print("Recall do modelo:", recall)
print("F1-score do modelo:", f1)

# Criar um gráfico da árvore de decisão
fig, ax = plt.subplots(figsize=(12, 8))
tree.plot_tree(model,
               feature_names=features,
               class_names=[str(class_name) for class_name in model.classes_],
               filled=True,
               rounded=True,
               impurity=False,
               ax=ax)
text_objects = ax.findobj(plt.Text)
for text in text_objects:
    text_str = text.get_text()
    if text_str.startswith('X'):
        text.set_text(text_str + ' = True')
    elif text_str.startswith('X') and '<=' in text_str:
        text.set_text(text_str.replace('X', ''))
    elif text_str.startswith('X') and '>' in text_str:
        text.set_text(text_str.replace('X', '') + ' = False')
plt.show()

# Salvar o gráfico em um arquivo PDF
fig.savefig("arvore_decisao_heart.pdf")
