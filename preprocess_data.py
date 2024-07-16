import os
from pathlib import Path



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