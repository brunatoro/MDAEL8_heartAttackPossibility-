#Implementation of Kmeans from scratch and using sklearn
#Loading the required modules 
import numpy as np
from scipy.spatial.distance import cdist 
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score, homogeneity_score




def plot_samples(projected, labels, title):    
    fig = plt.figure(figsize=(10, 6))
    u_labels = np.unique(labels)
    colors = sns.color_palette("husl", len(u_labels))

    for i, color in zip(u_labels, colors):
        plt.scatter(projected[labels == i , 0] , projected[labels == i , 1] , label = f'Cluster {i}',
                    edgecolor='none', alpha=0.7, s=50, c=[color])
    
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.legend()
    plt.title(title)

    plt.show()


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
             'resultado'] #caracteristicas que quero trabalhar
    target = 'resultado'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                    names = names) # Nome das colunas
        
    normalized_df = (df[features] - df[features].min()) / (df[features].max() - df[features].min())
    #normalized_df = df[features].apply(zscore)
    #normalized_df = df[features] / (10 ** np.ceil(np.log10(df[features].abs().max())))
    
     # Separating out the features
    x = df.loc[:, features].values
    
    x = MinMaxScaler().fit_transform(x)
    normalizedDf = pd.DataFrame(data = x, columns = features)
    normalizedDf = pd.concat([normalizedDf, df[[target]]], axis = 1)

    # Apply PCA to the normalized data
    pca = PCA(2)
    projected = pca.fit_transform(normalizedDf)

    
    #Applying sklearn GMM function
    gm  = GaussianMixture(n_components=2).fit(projected)
    print(gm.weights_)
    print(gm.means_)
    x = gm.predict(projected)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(projected, x)
    print("Silhouette Score:", silhouette_avg)

     # Calculate homogeneity score
    homogeneity = homogeneity_score(df[target], x)
    print("Homogeneity Score:", homogeneity)

     # Visualize the results sklearn
    plot_samples(projected, x, 'Clusters Labels GMM')
    plt.show()
    #Visualize the results sklearn

if __name__ == "__main__":
    main()