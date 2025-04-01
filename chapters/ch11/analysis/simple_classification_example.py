import math
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
from node2vec import Node2Vec




import os
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from util.graphdb_base import GraphDBBase


class EvaluateEmbedding():

    def train(self, train_dataset):
        X = train_dataset.embeddings.values.tolist()
        y = train_dataset.label.values.tolist()

        self.scaler = StandardScaler().fit(X)
        X_std = self.scaler.transform(X)

        clf = LogisticRegressionCV(random_state=0, solver='liblinear', multi_class='ovr', max_iter=1000)
        self.model = clf.fit(X_std, y)

    def evaluate(self, test_dataset):
        X_test = test_dataset.embeddings.values.tolist()
        y_test = test_dataset.label.values.tolist()
        X_test_std = self.scaler.transform(X_test)

        prediction = self.model.predict(X_test_std)
        gold = y_test
        print("Gold:\t\t", gold)
        print("Predicted:\t", list(prediction))


        weighted = precision_recall_fscore_support(gold, prediction, average='weighted')
        print('Precision:', weighted[0], 'Recall:', weighted[1], 'f-score:', weighted[2])
        MUX = confusion_matrix(gold, prediction)
        print("Confusion Matrix:\t", MUX)

def set_club_colors(G):
    for node in G.nodes(data=True):
        # Mr. Hi = 'purple', Officier = 'blue'
        color = '#00fff9'
        if node[1]['club'] == 'Mr. Hi':
            color = '#e6e6fa'
        node[1]['color'] = color


def compute_degree_embeddings(G):
    # using degree as embedding
    embeddings = np.array(list(dict(G.degree()).values()))
    embeddings = [[i] for i in embeddings]
    return embeddings

def compute_specific_degree_embeddings(G):
    clubs = nx.get_node_attributes(G, "club")
    mr_hi_degree = [[clubs[c] for c in G.neighbors(i)].count('Mr. Hi') for i in G.nodes()]
    officer_degree = [[clubs[c] for c in G.neighbors(i)].count('Officer') for i in G.nodes()]
    degree = list(dict(G.degree()).values())
    embeddings = [[degree[i], mr_hi_degree[i], officer_degree[i]] for i in G.nodes]
    return embeddings

def compute_complex_embeddings(G):
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4, seed=0)
    model = node2vec.fit(window=10, min_count=1, batch_words=4, seed=0)
    embeddings = [model.wv.get_vector(i) for i in G.nodes]
    return embeddings

def draw_and_save_graph_picture(G):
    set_club_colors(G)
    layout_position = nx.spring_layout(G, k=8 / math.sqrt(G.order()))
    colors = [n[1]['color'] for n in G.nodes(data=True)]
    nx.draw_networkx(G, pos=layout_position, node_color=colors)
    plt.axis('off')
    plt.savefig("Karate_Graph.svg", format="SVG", dpi=1000)
    plt.savefig("Karate_Graph.png", format="PNG", dpi=1000)
    plt.show()

if __name__ == '__main__':
    start = time.time()
    G = nx.karate_club_graph()
    draw_and_save_graph_picture(G)

    # retrieve the labels for each node
    labels = np.asarray([G.nodes[i]['club'] != 'Mr. Hi' for i in G.nodes]).astype(np.int64)

    #embeddings = compute_degree_embeddings(G)
    #embeddings = compute_complex_embeddings(G)
    embeddings = compute_specific_degree_embeddings(G)

    df = pd.DataFrame({
        'nodeId': G.nodes,
        'embeddings': embeddings,
        'label': labels
    })

    train, test = train_test_split(df, test_size=0.4, random_state=0)

    classifier = EvaluateEmbedding()
    classifier.train(train)
    classifier.evaluate(test)
    end = time.time() - start
    print("Time to complete:", end)
