import glob
import json
import os
from collections import Counter
from pathlib import Path


def read_sequences_from_file(file_path):
    with open(file_path, 'r') as file:
        sequence = [int(num) for num in file.read().strip().split()]
    return sequence

def read_sequences_from_folder(folder_path):
    sequences = []
    for file_path in glob.glob(os.path.join(folder_path, '*.txt')):
        sequences.append(read_sequences_from_file(file_path))
    return sequences

def read_sequences_from_folder_with_subfolders(folder_path):
    sequences = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                sequences.append(read_sequences_from_file(file_path))
    return sequences

def read_all_sequences(parent_folder):
    data_dir = os.path.join(parent_folder, "ADFA")
    path_sub_folders = [os.path.join(data_dir, sub_folder)  for sub_folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, sub_folder))]
    print("sub folders :", path_sub_folders)

    sequences = []
    labels = []
    for folder_path in path_sub_folders:
        if "Attack" in folder_path:
            data = read_sequences_from_folder_with_subfolders(folder_path)
            sequences.extend(data)
            labels.extend(["malware"] * len(data))


        else:
            data = read_sequences_from_folder(folder_path)
            sequences.extend(data)
            labels.extend(["normal"] * len(data))
    return sequences, labels


def save_sequence_data( sequences, labels, output_file_path):
    # Serialize all sequence data and vocabulary to a single file
    with open(output_file_path, 'w') as f:
        data_to_save = {
            'sequence_data': sequences,
            # 'vocab_size': len(encoder.syscall_vocabs),
            # 'vocab': encoder.syscall_vocabs,
            'labels': labels
            # 'vocab': syscall_vocab, 
        }
        json.dump(data_to_save, f)

    return output_file_path


def load_and_print_dataset(file_path, print_data):
    with open(file_path, 'r') as f:
        loaded_data = json.load(f)

    sequences = loaded_data['sequence_data']
    labels = loaded_data['labels']
    # vocab_size = loaded_data['vocab_size']
    # syscall_vocab = loaded_data['vocab']
    # max_seq_len = loaded_data['max_seq_len']
    if print_data:
        for idx, (sequence, label) in enumerate(zip(sequences, labels)):
            print(f"Sequence: {sequence}\nLabel: {label}\n")
        # print(f"Vocabulary Size: {vocab_size}")
        # print('Vocab: ', syscall_vocab)
        # print(f"Max Length: {max_seq_len}")

    return sequences, labels

def fetch_graphs(encoder, sequences, labels):
    graphs = []
    for sequence, label in zip(sequences, labels):
        graphs.append(encoder.sequence_to_graph(sequence, label))
    return graphs


    
if __name__ == "__main__":
    # Paths to folders : two normal data folder and third attack folder containing subfolders
    curr_dirr = os.path.dirname(os.getcwd())

    parent_dirr =Path(curr_dirr).parent
    # print("parent dir :", curr_dirr)
    all_sequences, labels = read_all_sequences(parent_folder=parent_dirr)

    # save the data


    print(Counter(labels))
    # print(Counter(labels))

    # split data into training and testing
