import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tabulate import tabulate

# Load the dataset
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

# Split the dataset into training and testing sets
X = df.drop('resultado', axis=1)
y = df['resultado']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a neural network classifier with 2 hidden layers each of 4 neurons
nn = MLPClassifier(hidden_layer_sizes=(4, 4), max_iter=1000)

# Train the neural network classifier
nn.fit(X_train, y_train)

# Use the trained neural network classifier to predict results for the test set
y_pred_nn = nn.predict(X_test)

# Create and train decision tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Use trained decision tree classifier to make predictions on the test set
y_pred_dt = dt.predict(X_test)

# Create and train KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Use trained KNN classifier to make predictions on the test set
y_pred_knn = knn.predict(X_test)

# Create and train SVM classifier
svm = SVC()
svm.fit(X_train, y_train)

# Use trained SVM classifier to make predictions on the test set
y_pred_svm = svm.predict(X_test)

# Calculate the classification report and confusion matrix for each classifier
print("MLPClassifier classification report:")
classification_report_nn = classification_report(y_test, y_pred_nn, output_dict=True)
print(tabulate(pd.DataFrame(classification_report_nn).transpose(), headers='keys'))

print("\nConfusion Matrix:")
print(tabulate(confusion_matrix(y_test, y_pred_nn), headers=['Predicted 0', 'Predicted 1']))

print("\nDecisionTreeClassifier classification report:")
classification_report_dt = classification_report(y_test, y_pred_dt, output_dict=True)
print(tabulate(pd.DataFrame(classification_report_dt).transpose(), headers='keys'))

print("\nConfusion Matrix:")
print(tabulate(confusion_matrix(y_test, y_pred_dt), headers=['Predicted 0', 'Predicted 1']))

print("\nKNeighborsClassifier classification report:")
classification_report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
print(tabulate(pd.DataFrame(classification_report_knn).transpose(), headers='keys'))

print("\nConfusion Matrix:")
print(tabulate(confusion_matrix(y_test, y_pred_knn), headers=['Predicted 0', 'Predicted 1']))

print("\nSVM classification report:")
classification_report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
print(tabulate(pd.DataFrame(classification_report_svm).transpose(), headers='keys'))

print("\nConfusion Matrix:")
print(tabulate(confusion_matrix(y_test, y_pred_svm), headers=['Predicted 0', 'Predicted 1']))