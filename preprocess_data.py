import json
import os
import pickle
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from custom_dataset import CustomGraphDataset
from sklearn.calibration import LabelEncoder
from torch_geometric.utils.convert import from_networkx


def read_subfolder(path: str, label):
    sequences = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r") as f:
                lines = f.readlines()
                # print(lines)
                # there is only one line
                line = list(map(int, lines[0].strip().split()))
                sequences.append(line)
                labels.append(label)
    return sequences, labels


def read_adfa_data(path: str):
    """
    sub_dir: data folder full path (e.g. "/../.../ADFA/Training_Data_Master")
    Read all files in the data folder and return a list of sequences.
    """
    sequences = []
    labels = []

    # check labels when reading attack data to be able to assign label to 1
    label = 0  # indicates benign
    if "Attack_Data_Master" in path:
        label = 1
        for sub_folder in list(sorted(os.listdir(path))):
            sub_folder_path = os.path.join(path, sub_folder)
            if os.path.isdir(sub_folder_path):
                # print("processing folder: ", sub_folder)
                sub_folder_sequences, sub_folder_labels = read_subfolder(
                    sub_folder_path, label=label
                )
                # print(f"len of sequences: {sub_folder} = {len(sub_folder_sequences)}")

                sequences.extend(sub_folder_sequences)
                labels.extend(sub_folder_labels)

        # return a list of sequences, and labels for the attack data
        # print(f"Read {len(sequences)} sequences from {path}")
        return sequences, labels

    # return a list of sequences, and labels for the benign data

    sequences, labels = read_subfolder(path, label=label)
    # print(f"Read {len(sequences)} sequences from {path}")
    return sequences, labels


def fetch_sequence_data(folder_path):
    # make sure "ADFA" folder in the parent directory of this project's folder [ie., your codes]
    current_directory = Path(os.getcwd())
    parent_path = current_directory.parent.absolute()
    # print(current_directory.parent.absolute())

    full_data_folder_path = os.path.join(parent_path, folder_path)

    adfa_sub_folders = [
        "Training_Data_Master",
        "Validation_Data_Master",
        "Attack_Data_Master",
    ]

    benign_training_data_path = os.path.join(full_data_folder_path, adfa_sub_folders[0])
    benign_validation_data_path = os.path.join(
        full_data_folder_path, adfa_sub_folders[1]
    )

    attack_data_path = os.path.join(full_data_folder_path, adfa_sub_folders[2])

    # # read the sub folders
    benign_train_sequences, benign_train_labels = read_adfa_data(
        benign_training_data_path
    )

    benign_val_sequences, benign_val_labels = read_adfa_data(
        benign_validation_data_path
    )

    attack_sequences, attack_labels = read_adfa_data(attack_data_path)

    # combine all data
    data = benign_train_sequences + benign_val_sequences + attack_sequences
    labels = benign_train_labels + benign_val_labels + attack_labels

    return data, labels


def sequence_to_graph(L: List, graph_label=None, vocabs=None):
    """
    Convert a sequence of (integers) to a graph.
    Currently, we are using already encoded set of integers that represent system calls. If raw data is used, it will be necessary to encode the data first using a dictionary.
    """
    # create a graph
    G = nx.DiGraph()
    for i in range(len(L) - 1):
        # u & v are the system calls integers from raw data ...but want to map them to encoded represented.e.g. system call 102 may be represented by 83
        u, v = L[i], L[i + 1]
        if vocabs is not None:
            # extract their key from values (u,v)
            u, v = vocabs[u], vocabs[v]
        # edge = (L[i], L[i + 1])
        # if edge is not in the graph
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=1)

        # if edge is in the graph, just update the weight
        else:
            # u, v = edge
            G[u][v]["weight"] += 1

    # convert networkx graph to pyg graph data
    # nodes = torch.tensor(list(G.nodes), dtype=torch.long)
    # node_attr = [nodes]
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    degree_centrality = nx.degree_centrality(G)
    # eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
    pagerank = nx.pagerank(G, weight="weight")

    # Define nodes and their features
    node_features = []
    for node in G.nodes():
        betweenness = betweenness_centrality[node]
        closeness = closeness_centrality[node]
        degree = degree_centrality[node]
        # eigenvector = eigenvector_centrality[node]
        pr = pagerank[node]
        # print(syscall, f'katz: {katz}', f'betweenness: {betweenness}', f'closeness: {closeness}', f'degree: {degree}', f'eigenvector: {eigenvector}', f'pagerank: {pr}')

        features = [node, betweenness, closeness, degree, pr]

        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)
    # edge_index = list(G.edges())
    # edge_features = [edge_counter[edge] for edge in edge_index]
    # edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # edge_features = torch.tensor(edge_features, dtype=torch.float)

    # graph = Data(
    #     x=x, edge_index=edge_index, num_nodes=num_nodes, edge_features=edge_features
    # )

    # convert graph to pytorch geometric data
    data = from_networkx(G)
    data.x = x
    if graph_label is not None:
        data.y = graph_label

    # validate the data
    data.validate(raise_on_error=True)

    return G, data


def fetch_graph_data(sequences, labels, vocabs):
    # convert each seq to graph
    graphs = []
    for i in range(len(sequences)):
        nx_graph_G, pyg_graph_data = sequence_to_graph(
            sequences[i], graph_label=labels[i], vocabs=vocabs
        )
        graphs.append(pyg_graph_data)
    return graphs


def build_vocabs(sequences):
    vocabs = []
    for seq in sequences:
        vocabs.extend(seq)
    # get number of unique vocabularies
    # vocab_size = len(set(vocabs))

    vocabs = set(vocabs)
    syscall_mappings = dict()
    for idx, syscall in enumerate(list(vocabs)):
        syscall_mappings[syscall] = idx

    print("vocab_size = ", len(syscall_mappings.keys()))

    return syscall_mappings


def save_graph_data(data, file_name):
    file_name = file_name
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


# def load_graph_data(file_name):
#     with open(file_name, "rb") as f:
#         graph_data = pickle.load(f)
#     return graph_data



def load_graph_data(file_path, training=False):
    """
    Args:
        file_name (str): file name with .pkl extension
    """

    with open(file_path, "rb") as f:
        graph_data = pickle.load(f)

        # shuffle(graph_data)
        # print(type(graph_data))
        graphs = [data for data in graph_data]
        labels = [data.y for data in graph_data]
        # train_graphs, test_graphs, train_labels, test_labels = train_test_split(
        #     graphs, labels, test_size=0.3, random_state=42, stratify=labels
        # )

        # Binarize labels
        # train_labels = ["normal" if label == 0 else "malware" for label in train_labels]
        # test_labels = ["normal" if label == 0 else "malware" for label in test_labels]

        labels = ["normal" if label == 0 else "malware" for label in labels]

        # Encode labels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        # test_labels = label_encoder.transform(test_labels)

        vocab_size = 175

        graph_dataset = CustomGraphDataset(graphs, classes=2, training=True)
        # test_dataset = CustomGraphDataset(test_graphs, classes=2, training=False)
        # return train_dataset, test_dataset, vocab_size, label_encoder
        return graph_dataset, vocab_size, label_encoder

def save_sequence_dataset(data,  labels, file_name):
    with open(file_name, "w") as f:
        data = {"sequences": data, "labels": labels}
        json.dump(data, f)

def load_sequence_dataset(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
        sequences = data["sequences"]
        labels = data["labels"]
    return sequences, labels

if __name__ == "__main__":

    # set reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # load data
    data_folder_path = "/Users/gere/Desktop/research projects/gan_models/ADFA"
    sequences, labels = fetch_sequence_data(data_folder_path)
    vocabs = build_vocabs(sequences)
    # vocabs = None

    ## train - test split the data

    train_data, test_data, train_labels, test_labels = train_test_split(
        sequences, labels, random_state=42, test_size=0.2, stratify=labels, shuffle=True
    )

    # save train and test data in pkl files
    save_sequence_dataset(sequences, labels, "data/train_dataset.json")
    save_sequence_dataset(test_data, test_labels, "/data/test_dataset.json")


    # load datasets

    train_data, train_labels = load_sequence_dataset("data/train_dataset.json")
    test_data, test_labels = load_sequence_dataset("data/test_dataset.json")
    print("data size = ", len(train_data))
    # print("test data size = ", len(test_data))

    # # encode sequences to graphs


    train_graph_data = fetch_graph_data(train_data, train_labels, vocabs=vocabs)
    test_graph_data = fetch_graph_data(test_data, test_labels, vocabs=vocabs)

    print(train_graph_data[0])

    # # save data
    file_name_train_gd = "data/train_graph_data.pkl"
    file_name_test_gd = "data/test_graph_data.pkl"

    save_graph_data(train_graph_data, file_name_train_gd)
    save_graph_data(test_graph_data, file_name_test_gd) 


    # load graph data

    train_graph_dataset, vocab_size, _ = load_graph_data("data/train_graph_data.pkl", training=True)
    test_graph_dataset,_, _ = load_graph_data("data/test_graph_data.pkl", training=False)

    # print(train_graph_data[10])