#Implementation of Kmeans from scratch and using sklearn
#Loading the required modules 
import numpy as np
from scipy.spatial.distance import cdist 
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import silhouette_score, homogeneity_score
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
 
#Defining our kmeans function from scratch
def KMeans_scratch(x, k, no_of_iterations, distance_metric='euclidean'):
    idx = np.random.choice(len(x), k, replace=False)
    centroids = x[idx, :]

    distances = cdist(x, centroids, distance_metric)
    points = np.array([np.argmin(i) for i in distances])

    for _ in range(no_of_iterations): 
        centroids = []
        for idx in range(k):
            temp_cent = x[points==idx].mean(axis=0) 
            centroids.append(temp_cent)

        centroids = np.vstack(centroids)
        distances = cdist(x, centroids, distance_metric)
        points = np.array([np.argmin(i) for i in distances])

    return points



def show_digitsdataset(digits):
    fig = plt.figure(figsize=(6, 6))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
       # ax.imshow(digits.axes[i], cmap=plt.cm.binary, interpolation='nearest')
        # label the image with the target value
        ax.text(0, 7, str(digits.resultado[i]))

    #fig.show()


def plot_samples(projected, labels, title):    
    fig = plt.figure()
    u_labels = np.unique(labels)
    for i in u_labels:
        plt.scatter(projected[labels == i , 0] , projected[labels == i , 1] , label = i,
                    edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('tab10', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.legend()
    plt.title(title)

 
def main():
    #Load dataset Digits
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
             'resultado'] #caracteristicas que quero trabalhar
    target = 'resultado'
    digits = pd.read_csv(input_file,         # Nome do arquivo com dados #df =  data framing
                     names = names,      # Nome das colunas 
                     usecols = features, # Define as colunas que serão  utilizadas
                     na_values='?')      # Define que ? será considerado valores ausentes
    
    
    # Normalizar os dados usando StandardScaler
    scaler = StandardScaler()
    normalized_digits = scaler.fit_transform(digits)

    # Aplicar o PCA aos dados normalizados
    pca = PCA(2)
    projected = pca.fit_transform(normalized_digits)
    print(pca.explained_variance_ratio_)
    print(projected.shape)

    plot_samples(projected, digits.resultado, 'Original Labels')
 
    #Applying our kmeans function from scratch
    labels = KMeans_scratch(projected,6,5)

    # Usando a medida de distância de Manhattan (L1)
    labels_manhattan = KMeans_scratch(projected, 6, 5, distance_metric='cityblock')
    
    # Visualizar os resultados
    plot_samples(projected, digits.resultado, 'Clusters Sexo KMeans (Manhattan)')
    
    #Visualize the results 
    plot_samples(projected, digits.resultado, 'Clusters Sexo KMeans from scratch')

    #Applying sklearn kemans function
    # Aplicando a função K-means do sklearn
    kmeans = KMeans(n_clusters=2).fit(projected)
    centers = kmeans.cluster_centers_
    score = silhouette_score(projected, kmeans.labels_)
    homogeneity = homogeneity_score(digits.resultado, kmeans.labels_)
    inertia = kmeans.inertia_
    print("Silhouette Score:", score)
    print("Homogeneity Score:", homogeneity)
    print("Inertia:", inertia)

    #Visualize the results sklearn
    plot_samples(projected, kmeans.labels_, 'Clusters Labels KMeans from sklearn')

    plt.show()
    
 

if __name__ == "__main__":
     main()