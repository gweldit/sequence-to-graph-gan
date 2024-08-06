import argparse

import numpy as np
import torch
from graph_encoder import GraphEncoder as enc
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.loader import DataLoader

from models import GNNModel
from utils import ExperimentTracker, set_random_seeds


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a GNN model on a graph dataset")

    # Dataset and Experiment Configuration
    # parser.add_argument("--dataset_path", type=str, default='data/processed_graphs.pkl', help="Path to the dataset")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment")

    # Model Parameters
    parser.add_argument("--model", type=str, choices=['MLP', 'GCN', 'Sage', 'GIN', 'GAT', 'GATv2'], default='GIN', help="Type of GNN encoder model to use")
    parser.add_argument("--hidden_channels", type=int, default=512, help="Number of hidden channels in the model")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the model")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Dimensionality of the embedding layer")
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout rate for the model")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads in GAT model, applicable only if GAT model is selected")
    parser.add_argument("--norm", type=str, default='batch', choices=[None, 'batch', 'layer'], help="Type of normalization to use, applicable to most models")

    # Optimization Parameters
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--max_lr", type=float, default=0.01, help="Maximum learning rate for OneCycleLR scheduler")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for optimizer")

    # add boolean argument to perform oversampling of the minority class
    parser.add_argument("--use_fake_data",  action='store_true',default=False, help="use fake data flag.")



    # Other Parameters
    args = parser.parse_args()
    args.experiment_name = args.model + "_balanced" if args.use_fake_data else args.model

    return args


def initialize_model(vocab_size, num_node_features, num_classes, num_steps_per_epoch, args, device):
    model = GNNModel(vocab_size, args.embedding_dim, num_node_features, args.hidden_channels, args.num_layers, num_classes, args.dropout, 'relu',
                     model_type=args.model, heads=args.heads, norm=args.norm).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, pct_start=0.3, steps_per_epoch=num_steps_per_epoch, epochs=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()
    return model, optimizer, scheduler, criterion


def train_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # print("max token = ", data.x[:,0].max())
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss.append(loss.item())
    return np.mean(total_loss)


def evaluate_model(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(data.y.cpu().numpy())
    return np.array(preds), np.array(labels)


def main():
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Call the function to set random seeds right at the beginning of main
    set_random_seeds()

    # train_dataset, test_dataset, vocab_size, label_encoder = load_data(args.dataset_path)
    vocab_size = 175

    train_dataset, vocab_size, label_encoder = enc.load_graph_data("data/train_graph_data.pkl", vocab_size=vocab_size,training=True, )

    # print(train_dataset[0])


    # if oversampling / use fake data is true
    if args.use_fake_data is True:
        print("Using fake data to train GNN model")
        fake_dataset, vocab_size, label_encoder = enc.load_graph_data("data/fake_graph_data.pkl", vocab_size=vocab_size,training=True, label_encoder=label_encoder)

        # combine train_dataset and fake_dataset
        train_dataset = train_dataset + fake_dataset    

        print("type of train_dataset,", type(train_dataset[0]))
        # print("sample dataset = ",  ([train_dataset[i].x[:,0].min().item() for i in range(len(train_dataset))]))
        
        print("len of train dataset = ", len(train_dataset))
    else:
        print("len of training dataset = ", len(train_dataset))
        
    test_dataset, vocab_size, _ = enc.load_graph_data("data/test_graph_data.pkl", vocab_size=vocab_size, training=False,label_encoder=label_encoder)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Extract `num_node_features` and `num_classes` to pass to `initialize_model`
    num_node_features = train_dataset[0].num_node_features
    num_classes = len(label_encoder.classes_)
    num_steps_per_epoch = len(train_loader)

    model, optimizer, scheduler, criterion = initialize_model(vocab_size, num_node_features, num_classes, num_steps_per_epoch, args, device)

    experiment_tracker = ExperimentTracker(model, optimizer, scheduler, label_encoder, args)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        train_preds, train_labels = evaluate_model(model, train_loader, device)
        test_preds, test_labels = evaluate_model(model, test_loader, device)
        current_lr = optimizer.param_groups[0]['lr']

        # Update the metrics tracker with the results from this epoch
        experiment_tracker.update_and_save(epoch, loss, train_preds, train_labels, test_preds, test_labels, current_lr)
        print(experiment_tracker)


if __name__ == "__main__":
    main()