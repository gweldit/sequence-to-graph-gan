
import pickle
import re

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric
from custom_dataset import CustomGraphDataset
from file_reader import (load_and_print_dataset, read_all_sequences,
                         save_sequence_data)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from syscall_file_reader import SyscallFileReader
from torch_geometric.data import Data

RELEVANT_SYSCALLS = {
    "recvfrom", "write", "ioctl", "read", "sendto", "writev", "close", "socket", "bind", "connect",
    "mkdir", "access", "chmod", "open", "fchown", "rename", "unlink", "umask", "recvmsg", "sendmsg",
    "getdents64", "epoll_wait", "dup", "pread", "pwrit", "fcntl64"
}


class SyscallFileReader:
    def __init__(self, filter_calls=False):
        self.filter_calls = filter_calls

    def read(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Determine the format by checking the first non-empty line
        first_non_empty_line = next((line for line in lines if line.strip()), '').strip()

        # If the line is numeric (possibly space-separated), treat it as a pre-processed sequence
        if first_non_empty_line.replace(' ', '').isnumeric():
            # Assuming pre-processed sequences are space-separated
            syscalls = [int(num) for num in first_non_empty_line.split()]
        else:
            # Define a regular expression pattern for syscalls (assuming syscalls start with word characters and end with '(')
            syscall_pattern = re.compile(r'^\w+\(')

            syscalls = []

            for line in lines:
                # Strip whitespace from the beginning and end of the line
                line = line.strip()

                # Use regular expression to find a match at the start of the line
                match = syscall_pattern.match(line)
                if match:
                    syscall_name = match.group().rstrip('(')  # Extract syscall name and remove trailing '('

                    if self.filter_calls:
                        # Include only relevant syscalls if filtering is enabled
                        if syscall_name in self.RELEVANT_SYSCALLS:
                            syscalls.append(syscall_name)
                    else:
                        # Include all syscalls if no filtering is applied
                        syscalls.append(syscall_name)

        return syscalls

class GraphEncoder:
    """Class to encode system call sequences into graph structures."""
    def __init__(self):
        """Initialize an empty dictionary for the vocabulary."""
        # Initialize an empty dictionary for the vocabulary
        self.syscall_vocabs = {}
        # Assign a special token for unknown syscalls
        self.unknow_token = len(self.syscall_vocabs)

    def build_vocabulary(self, sequence_of_syscalls):
        """Build vocabulary from the list of syscalls."""

        for syscall in sequence_of_syscalls:
            if syscall not in self.syscall_vocabs:
                # Assign a new token to the syscall
                
                self.syscall_vocabs[syscall] = len(self.syscall_vocabs)

    def encode(self, sequence_of_syscalls):
        """Encode a sequence of system calls into a graph."""

        if not sequence_of_syscalls:
            raise ValueError("Sequence is empty.")

        # Ensure the vocabulary is built
        self.build_vocabulary(sequence_of_syscalls)

        unique_syscalls = list(set(sequence_of_syscalls))
        num_nodes = len(unique_syscalls)
        node_mapping = {syscall: i for i, syscall in enumerate(unique_syscalls)}

        G = nx.DiGraph()
        edge_counter = {}
        for i in range(len(sequence_of_syscalls) - 1):
            src = node_mapping[sequence_of_syscalls[i]]
            dst = node_mapping[sequence_of_syscalls[i + 1]]
            edge = (src, dst)
            edge_counter[edge] = edge_counter.get(edge, 0) + 1
            if G.has_edge(src, dst):
                G[src][dst]["weight"] += 1
            else:
                G.add_edge(src, dst, weight=1)

        # Define nodes and their features
        node_features = []
        if num_nodes > 1:
            # Compute centrality measures and other features
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            pagerank = nx.pagerank(G, weight="weight")

            if num_nodes > 2:
                katz_centrality = nx.katz_centrality_numpy(G, weight="weight")
            else:
                katz_centrality = {node: 0 for node in G}

            for syscall in unique_syscalls:
                node_idx = node_mapping[syscall]
                token = self.syscall_vocabs[syscall]
                katz = katz_centrality[node_idx]
                betweenness = betweenness_centrality[node_idx]
                closeness = closeness_centrality[node_idx]
                pr = pagerank[node_idx]
                # syscall_type_encoding = get_syscall_type_encoding(syscall)

                # Append features to the node features, including the syscall
                # features = (
                #     [token] + syscall_type_encoding + [katz, betweenness, closeness, pr]
                #     )
                features = (
                    [token]  + [katz, betweenness, closeness, pr]
                )
                node_features.append(features)
        else:
            # Handle singleton graph, assign default centrality measures
            for syscall in unique_syscalls:
                token = self.syscall_vocabs[syscall]
                # syscall_type_encoding = get_syscall_type_encoding(syscall)
                # features = [token] + syscall_type_encoding + [0, 0, 0, 0]
                features = [token] + [0, 0, 0, 0]

                node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = list(G.edges())
        edge_features = [edge_counter[edge] for edge in edge_index]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_features, dtype=torch.float)

        graph = Data(
            x=x, edge_index=edge_index, num_nodes=num_nodes, edge_features=edge_features
        )
        graph.validate(raise_on_error=True)

        return graph, node_mapping
    

    def fetch_graphs(self, sequences, labels):
        """Fetch PyTorch geometric graph data from sequences and labels."""
        graphs = []
        for sequence, label in zip(sequences, labels):
            graph, node_mapping = self.encode(sequence)
            # graph.y = torch.tensor([label], dtype=torch.long)
            data = {"graph":graph, "label":label}
            graphs.append(data)
        return graphs, node_mapping
    

    def save_graph_data(self, graph_data,output_filename):
        """saves graphs data into pkl file"""
        with open(output_filename, "wb") as f:
            pickle.dump(graph_data, f)


    @staticmethod
    def load_graph_data(file_path:str, vocab_size:int, training:bool, label_encoder=None):
        """
        Load graph data from a pickle file.

        Args:
            file_path (str): The path to the pickle file.
            vocab_size (int): The size of the vocabulary.
            training (bool): A boolean indicating if the data is for training or testing.
            label_encoder (LabelEncoder, optional): The label encoder. Defaults to None.

        Returns:
            tuple: A tuple containing the training and test datasets, the vocabulary size, and the label encoder.
        """
        
        with open(file_path, "rb") as f:
            graph_data = pickle.load(f)
            print(len(graph_data))

            graphs = [data['graph'] for data in graph_data]
            labels = [data['label'] for data in graph_data]
            
            # Binarize labels
            labels = ['normal' if label == 'normal' else 'malware' for label in labels]

            # Encode labels
            if label_encoder is None:
                # Create a LabelEncoder: when we encode the training labels
                label_encoder = LabelEncoder()
                labels = label_encoder.fit_transform(labels)
            else:
                # Use the same encoder to encode the test labels
                labels = label_encoder.transform(labels)

            for graph, label in zip(graphs, labels):
                graph.y = torch.tensor([label], dtype=torch.long)


            dataset = CustomGraphDataset(graphs, len(label_encoder.classes_), training=training)
    
            return dataset, vocab_size, label_encoder
        
    @staticmethod
    def load_data(dataset_path):
        """ this function will be removed soon. """
        graph_data, vocab_size = load_dataset(dataset_path)

        graphs = [data['graph'] for data in graph_data]
        labels = [data['label'] for data in graph_data]

        train_graphs, test_graphs, train_labels, test_labels = train_test_split(
            graphs, labels, test_size=0.3, random_state=42, stratify=labels
        )

        # Binarize labels
        train_labels = ['normal' if label == 'normal' else 'malware' for label in train_labels]
        test_labels = ['normal' if label == 'normal' else 'malware' for label in test_labels]

        # Encode labels
        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(train_labels)
        test_labels = label_encoder.transform(test_labels)

        for graph, label in zip(train_graphs, train_labels):
            graph.y = torch.tensor([label], dtype=torch.long)

        for graph, label in zip(test_graphs, test_labels):
            graph.y = torch.tensor([label], dtype=torch.long)

        train_dataset = CustomGraphDataset(train_graphs, len(label_encoder.classes_), training=True)
        test_dataset = CustomGraphDataset(test_graphs, len(label_encoder.classes_), training=False)
        return train_dataset, test_dataset, vocab_size, label_encoder


    def plot_graph(self, file_path, filter_calls, graph, node_mapping):
        g = torch_geometric.utils.to_networkx(graph, to_undirected=False)
        pos = nx.spring_layout(g)
        reverse_mapping = {i: syscall for syscall, i in node_mapping.items()}


        # Extract edge labels based on the edge_features
        edge_labels = {
            tuple(graph.edge_index[:, i].tolist()): int(graph.edge_features[i].item())
            for i in range(graph.edge_features.size(0))
        }

        # Define color map for nodes


        # Set the figure size
        plt.figure(figsize=(10, 6))  # Adjust the size as needed

        # Draw nodes with different colors and labels
        nx.draw_networkx_nodes(
            g,
            pos,
            node_size=100,
        )
        labels = {node: reverse_mapping[node] for node in g.nodes()}
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=8)

        # Draw edges with labels
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8)

        plt.axis("off")
        plt.savefig(f"{file_path}.png") if not filter_calls else plt.savefig(
            f"{file_path}_filtered.png"
        )


def test(file_path, filter_calls, plot):
    reader = SyscallFileReader(filter_calls)
    encoder = GraphEncoder()

    # Read syscall
    syscalls = reader.read(file_path)
    graph, node_mapping = encoder.encode(syscalls=syscalls)
    # print(syscalls)

    if plot:
        # Generate graph plot
        print(graph)
        encoder.plot_graph(file_path, filter_calls, graph, node_mapping)


    

if __name__ == "__main__":
    # fetch sequence data
    data_folder_path = "/Users/gere/Desktop/research projects/gan_models"
    sequences, labels = read_all_sequences(data_folder_path)
    # split data into train and test sets
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        sequences, labels, test_size=0.3, random_state=42, stratify=labels
    )
    # save the sequences and their labels to a pkl files
    save_sequence_data(train_sequences, train_labels, "data/train_dataset.json")
    save_sequence_data(test_sequences, test_labels, "data/test_dataset.json")
    # load the sequences and their labels from the pkl files
    train_sequences, train_labels = load_and_print_dataset("data/train_dataset.json", print_data=False)
    test_sequences, test_labels = load_and_print_dataset("data/test_dataset.json", print_data=False)

    # build vocabulary
    encoder = GraphEncoder()
    # build vocabulary using only the training sequences
    encoder = GraphEncoder()
    for sequence in train_sequences:
        encoder.build_vocabulary(sequence)
    
    # fetch graphs
    train_graphs, node_mappings = encoder.fetch_graphs(train_sequences, train_labels)
    print("vocab size in training dataset: ", len(encoder.syscall_vocabs))
    test_graphs, _ = encoder.fetch_graphs(test_sequences, test_labels)
    encoder.save_graph_data(train_graphs, "data/train_graph_data.pkl")
    encoder.save_graph_data(test_graphs, "data/test_graph_data.pkl") 

    # load graphs
    train_graphs, vocab_size, label_encoder = encoder.load_graph_data("data/train_graph_data.pkl", vocab_size=len(encoder.syscall_vocabs.keys()), training=True)
    test_graphs, vocab_size,label_encoder = encoder.load_graph_data("data/test_graph_data.pkl", vocab_size=len(encoder.syscall_vocabs.keys()), training=False, label_encoder=label_encoder)
