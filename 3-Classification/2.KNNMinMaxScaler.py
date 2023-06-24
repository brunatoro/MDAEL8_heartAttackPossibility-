# Initial imports
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from tabulate import tabulate


# Calculate distance between two points
def minkowski_distance(a, b, p=2):
    # Store the number of dimensions
    dim = len(a)
    # Set initial distance to 0
    distance = 0
    # Calculate Minkowski distance using parameter p
    for d in range(dim):
        distance += abs(a[d] - b[d])**p
    distance = distance**(1/p)
    return distance


def knn_predict(X_train, X_test, y_train, y_test, k, p):
    # Make predictions on the test data
    # Need output of 1 prediction per test data point
    y_hat_test = []
    for test_point in X_test:
        distances = []
        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)
        # Store distances in a dataframe
        df_dists = pd.DataFrame(data=distances, columns=['dist'], index=y_train.index)
        # Sort distances and only consider the k closest points
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]
        # Create counter object to track the labels of k closest neighbors
        counter = Counter(y_train[df_nn.index])
        # Get the most common label of all the nearest neighbors
        prediction = counter.most_common()[0][0]
        # Append prediction to output list
        y_hat_test.append(prediction)
    return y_hat_test


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
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

    target_names = ['Menores chances', 'Maiores chances']

    # Separating out the features
    X = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:, target]

    print("Total samples: {}".format(X.shape[0]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test samples: {}".format(X_test.shape[0]))

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # STEP 1 - TESTS USING knn classifier written from scratch
    # Make predictions on test dataset using knn classifier
    y_hat_test = knn_predict(X_train, X_test, y_train, y_test, k=5, p=2)

    # Get test accuracy score
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Accuracy K-NN from scratch: {:.2f}%".format(accuracy))
    print("F1 Score K-NN from scratch: {:.2f}%".format(f1))

    # Get test confusion matrix
    cm = confusion_matrix(y_test, y_hat_test)
    plot_confusion_matrix(cm, target_names, False, "Confusion Matrix - K- MinMaxScaler")
    plot_confusion_matrix(cm, target_names, True, "Confusion Matrix - K-NN normalized MinMaxScaler")

    # STEP 2 - TESTS USING knn classifier from scikit-learn
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_hat_test = knn.predict(X_test)

    # Get test accuracy score
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Accuracy K-NN from scikit-learn: {:.2f}%".format(accuracy))
    print("F1 Score K-NN from scikit-learn: {:.2f}%".format(f1))
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5)
    clusters = kmeans.fit_predict(X_train)
    
    silhouette_avg = silhouette_score(X_train, clusters)
    print("Silhouette Score:", silhouette_avg)

    # Get test confusion matrix
    cm = confusion_matrix(y_test, y_hat_test)
    plot_confusion_matrix(cm, target_names, False, "Confusion Matrix - K-NN sklearn MinMaxScaler")
    plot_confusion_matrix(cm, target_names, True, "Confusion Matrix - K-NN sklearn normalized MinMaxScaler")
    plt.show()

    # STEP 3 - Results tabulation
    results = []
    results.append(['Model', 'Accuracy', 'F1 Score'])
    results.append(['K-NN from scratch', "{:.2f}%".format(accuracy), "{:.2f}%".format(f1)])

    # Print results
    print(tabulate(results, headers='firstrow'))

    # STEP 3 - Confusion matrix tabulation
    cm_scratch = confusion_matrix(y_test, y_hat_test)

    # Create headers for tabulation
    headers = ['Model', 'Confusion Matrix']
    model_names = ['K-NN from scratch']

    # Create data for tabulation
    data = []
    for model, cm in zip(model_names, [cm_scratch]):
        cm_table = tabulate(cm, headers='keys', tablefmt='grid')
        data.append([model, cm_table])

    # Print confusion matrices
    print(tabulate(data, headers, tablefmt='grid'))


if __name__ == '__main__':
    main()