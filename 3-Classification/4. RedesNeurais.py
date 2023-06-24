import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tabulate import tabulate

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
             'talassemia',
             ] #caracteristicas que quero trabalhar
target = 'resultado'
df = pd.read_csv(input_file, names=names)

# Dados de exemplo
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalização Min-Max
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42, solver='adam', activation='relu')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)



metrics = [['Acurácia', accuracy], ['F1-Score', f1]]
print(tabulate(metrics, headers=['Métrica', 'Valor']))

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title('Rede Neural')
ax.set_xlabel('Camada')
ax.set_ylabel('Neurônio')
ax.grid(True)

# Coordenadas dos neurônios
neuron_coords = []

# Coordenadas das entradas
input_coords = []

# Coordenadas das saídas
output_coords = []

# Camadas ocultas
for i, layer_sizes in enumerate(model.hidden_layer_sizes):
    layer_label = f'Camada Oculta {i+1}'
    neurons = layer_sizes
    
    # Posição da camada oculta no gráfico
    x = i + 0.5
    
    for j in range(neurons):
        y = (neurons - 1) / 2 - j
        neuron_coords.append((x, y))
        
        if i == 0:
            input_coords.append((x-0.5, y))
        elif i == len(model.hidden_layer_sizes) - 1:
            output_coords.append((x+0.5, y))
        
        # Plot do neurônio
        circle = plt.Circle((x, y), radius=0.1, facecolor='white', edgecolor='black')
        ax.add_patch(circle)
        ax.annotate(f'Neurônio {j+1}\n{layer_label}', (x, y), ha='center', va='center')

# Camada de saída
output_label = 'Camada de Saída'
neurons = model.n_outputs_

# Posição da camada de saída no gráfico
x = len(model.hidden_layer_sizes) + 1.5

for j in range(neurons):
    y = (neurons - 1) / 2 - j
    neuron_coords.append((x, y))
    output_coords.append((x, y))
    
    # Plot do neurônio
    circle = plt.Circle((x, y), radius=0.1, facecolor='white', edgecolor='black')
    ax.add_patch(circle)
    ax.annotate(f'Neurônio {j+1}\n{output_label}', (x, y), ha='center', va='center')

# Conexões entre neurônios
for i, (x, y) in enumerate(neuron_coords):
    if i < len(neuron_coords) - neurons:
        outputs = neuron_coords[i+neurons:]
    else:
        outputs = output_coords
    
    for output in outputs:
        ax.arrow(x, y, output[0]-x, output[1]-y, head_width=0.05, head_length=0.1, fc='black', ec='black')

# Legendas das entradas e saídas
for coord in input_coords:
    ax.annotate('Entrada', coord, ha='center', va='center', xytext=(-1, 0), textcoords='offset points')

for coord in output_coords:
    ax.annotate('Saída', coord, ha='center', va='center', xytext=(1, 0), textcoords='offset points')

plt.xlim(-1, len(model.hidden_layer_sizes) + 2)
plt.ylim(-(max(neurons for neurons in model.hidden_layer_sizes) - 1) / 2 - 1, (max(neurons for neurons in model.hidden_layer_sizes) - 1) / 2 + 1)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()