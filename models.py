import torch
import torch.nn.functional as F
from torch_geometric.graphgym import init_weights
from torch_geometric.nn import GAT, GCN, GIN, MLP, GraphSAGE, global_mean_pool


def count_parameters_in_millions(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6


class GNNModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        in_channels,
        hidden_channels,
        num_layers,
        out_channels,
        dropout,
        act,
        model_type="GNN",
        heads=1,
        norm=None,
        use_embedding=True,
    ):
        super(GNNModel, self).__init__()
        self.use_embedding = use_embedding

        if use_embedding:
            # Embedding layer for syscall tokens
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

            # Adjust in_channels to account for the new embedding dimension
            # Assume the original in_channels includes the syscall token as a one-hot encoded vector
            adjusted_in_channels = (
                in_channels - 1 + embedding_dim
            )  # Subtract 1 for the token, add embedding_dim
        else:
            # No adjustment needed if not using embeddings
            adjusted_in_channels = in_channels

        # GNN model definition
        if model_type == "MLP":
            self.gnn = MLP(
                in_channels=adjusted_in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=num_layers,
                dropout=0.0,
                act=act,
                norm=norm,
            )
        elif model_type == "GCN":
            self.gnn = GCN(
                adjusted_in_channels,
                hidden_channels,
                num_layers=num_layers,
                dropout=0.0,
                act=act,
                norm=norm,
            )
        elif model_type == "Sage":
            self.gnn = GraphSAGE(
                adjusted_in_channels,
                hidden_channels,
                num_layers=num_layers,
                dropout=0.0,
                act=act,
                norm=norm,
            )
        elif model_type == "GIN":
            self.gnn = GIN(
                adjusted_in_channels,
                hidden_channels,
                num_layers=num_layers,
                dropout=0.0,
                act=act,
                norm=norm,
            )
        elif model_type in ["GAT", "GATv2"]:
            self.gnn = GAT(
                adjusted_in_channels,
                hidden_channels,
                num_layers=num_layers,
                dropout=0.0,
                act=act,
                norm=norm,
                heads=heads,
                v2="v2" in model_type,
            )

        # Final fully connected layer
        self.fc = torch.nn.Linear(hidden_channels, out_channels, bias=True)
        self.dropout = dropout
        self.act = getattr(F, act)

        # Initialize weights
        init_weights(self)

        # Print model size
        print(f"Model size: {count_parameters_in_millions(self)}M")

    def forward(self, x, edge_index, batch):
        if self.use_embedding:
            # Embedding syscall tokens (assuming they are the first feature)
            syscall_embeddings = self.embedding(
                x[:, 0].long()
            )  # Assume syscall tokens are the first feature and are stored as long type

            # Concatenate the embeddings with the rest of the features
            x = torch.cat([syscall_embeddings, x[:, 1:]], dim=1)

        # Apply GNN layers
        x = self.gnn(x, edge_index)

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Apply dropout and the final fully connected layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)

        return x
