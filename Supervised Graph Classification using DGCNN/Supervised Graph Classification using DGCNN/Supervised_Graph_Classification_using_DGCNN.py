import pandas as pd
import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph
from sklearn import model_selection
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
from scipy.io import loadmat
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

raw_data = loadmat('C:/Users/Anthony Sirico/Documents/GitHub/Geomteric-Deep-Learning/template1-lib5-eqns-CR-RESULTS-SET1-FINAL.mat', squeeze_me=True)
data = raw_data['Graphs']
csv_data = pd.read_csv('C:/Users/Anthony Sirico/Documents/GitHub/Geomteric-Deep-Learning/graph_data.csv', converters={'Ln': eval, 'ln2': eval})

A = data['A']
graph_labels = pd.DataFrame()
graph_labels['perf'] = raw_data['Objective']
binary = []

for i in range(len(graph_labels)):
    if graph_labels['perf'][i] <= .03:
        binary.append(1)
    else:
        binary.append(-1)
graph_labels['perf2'] = binary
graph_labels = graph_labels.pop('perf2')
graph_labels = pd.get_dummies(graph_labels, drop_first=True)
print(graph_labels.value_counts().to_frame())

nx_graphs = []
for i in range(len(A)):
    nx_graphs.append(nx.Graph(A[i]))

for i in range(len(nx_graphs)):
    for j in range(nx_graphs[i].number_of_nodes()):
        nx_graphs[i].nodes[j]['feature'] = csv_data['ln2'][i][j]

nx_graphs_final = []
for i in range(len(nx_graphs)):
    nx_graphs_final.append(nx.relabel_nodes(nx_graphs[i], csv_data['Ln'][i]))

sg_graphs = []
for i in range(len(nx_graphs_final)):
    sg_graphs.append(StellarGraph.from_networkx(nx_graphs_final[i], node_features='feature'))
    
summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in sg_graphs],
    columns=["nodes", "edges"],
)
summary.describe().round(1)

generator = PaddedGraphGenerator(graphs=sg_graphs)

k = 35
layer_sizes = [32, 32, 32, 1]

dgcnn_model = DeepGraphCNN(
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    k=k,
    bias=False,
    generator=generator,
)
x_inp, x_out = dgcnn_model.in_out_tensors()

x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)

x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

x_out = Flatten()(x_out)

x_out = Dense(units=128, activation="relu")(x_out)
x_out = Dropout(rate=0.5)(x_out)

predictions = Dense(units=1, activation="sigmoid")(x_out)

model = Model(inputs=x_inp, outputs=predictions)

model.compile(
    optimizer=Adam(lr=0.0001), loss=binary_crossentropy, metrics=["acc"],
)

train_graphs, test_graphs = model_selection.train_test_split(
    graph_labels, train_size=0.8, test_size=None)

train_gen = generator.flow(
    list(train_graphs.index - 1),
    targets=train_graphs.values,
    batch_size=50,
    symmetric_normalization=False,
)

test_gen = generator.flow(
    list(test_graphs.index - 1),
    targets=test_graphs.values,
    batch_size=1,
    symmetric_normalization=False,
)

history = model.fit(
    train_gen, epochs=200, verbose=1, validation_data=test_gen, shuffle=True,
)

sg.utils.plot_history(history)
test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

