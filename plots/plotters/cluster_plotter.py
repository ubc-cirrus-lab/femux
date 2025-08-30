import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

def plot_cluster(df, clusters, centroids, x, y, pt1, pt2):
    plt.title('{} vs {}'.format(x,y))
    for i in range(clusters):
        plt.scatter(df[i][x], df[i][y], label='cluster_{}'.format(i+1))
    
    # plot centroids for the data
    plt.scatter(centroids[:, pt1], centroids[:, pt2], c='yellow', s=50)
    plt.xlabel('{}'.format(x))
    plt.ylabel('{}'.format(y))
    plt.legend()
    plt.savefig('output_plots/k_means_plot_{}_vs_{}.pdf'.format(x,y))

def plot_3d_cluster(df, clusters, centroids, kmeans, best_max_cluster_forecaster, cluster_index = -1):
    plt.title('Clustering with best cluster forecasters labeled (# of clusters={})'.format(len(df['Clusters_{}'.format(cluster_index)].unique())))
    # fig = plt.figure(figsize = (8,8))
    # ax = plt.figure().add_subplot(projection='3d')
    ax = plt.axes(projection='3d')
    ax.grid()
    # print("len of labels ", len(df['EMALinearity']))
    ax.scatter(df['EMALinearity'], df['EMAStationarity'], df['Harmonics'], c=df['Clusters_{}'.format(cluster_index)].astype(float), s=50, alpha=0.5) # , label=df['Clusters'].astype(float)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', s=50)
    for i in range(len(df['Clusters_{}'.format(cluster_index)].unique())):
        ax.text(centroids[i][0], centroids[i][1], centroids[i][2], best_max_cluster_forecaster[i])
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    ax.set_xlabel('EMALinearity')
    ax.set_ylabel('EMAStationarity')
    ax.set_zlabel('Harmonics')
    plt.savefig('output_plots/k_means_3d_plot.pdf')
