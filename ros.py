import torch
from custom_dataset import CustomGraphDataset
from torch_geometric.loader import DataLoader

def create_balanced_dataloader(dataset, classes, batch_size):
    """
    Creates a balanced DataLoader with oversampled minority classes.

    Parameters:
    - dataset: The dataset where each graph data has features and `data.y` (labels)
    - batch_size: The batch size for DataLoader

    Returns:
    - A DataLoader with balanced samples
    """
    # Extract all labels from the dataset
    labels = torch.tensor([data.y for data in dataset])
    
    # Step 1: Calculate Class Counts and Determine Target Count
    class_counts = torch.bincount(labels)
    max_count = class_counts.max().item()

    # Step 2: Create Indices to Balance Classes
    indices = []
    for class_label in range(len(class_counts)):
        # Get indices of all samples for the current class
        class_indices = (labels == class_label).nonzero(as_tuple=True)[0]

        # Calculate how many more samples are needed to reach max_count
        num_samples_to_add = max_count - len(class_indices)

        # Randomly sample indices from the current class to reach max_count
        sampled_indices = class_indices[torch.randint(0, len(class_indices), (num_samples_to_add,))]

        # Add original and duplicated indices to the list
        indices.extend(class_indices.tolist())
        indices.extend(sampled_indices.tolist())

    # Step 3: Extract the balanced dataset using sampled indices
    data_resampled = [dataset[i] for i in indices]
    labels_resampled = torch.tensor([dataset[i].y for i in indices])

    # Step 4: Create DataLoader with the balanced dataset
    balanced_dataset = CustomGraphDataset(data_resampled, classes)
    dataloader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


