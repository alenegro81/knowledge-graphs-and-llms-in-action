
import torch
from torch_geometric.nn import to_hetero
from ch14.model.gnn_model import GraphConvModel, GAT, SAGE, GIN

class HeteroBaseModel(torch.nn.Module):
    def __init__(self, metadata, input_dim, hidden_dim, out_dim, base_model):
        """
        A generic heterogeneous graph model wrapper.

        Args:
            metadata (tuple): Metadata of the heterogeneous graph.
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden feature dimension.
            out_dim (int): Output feature dimension.
            base_model (torch.nn.Module): Base homogeneous model (e.g., GCN, GAT, SAGE, GIN).
        """
        super(HeteroBaseModel, self).__init__()
        # Define the base homogeneous model
        self.base_model = base_model(input_dim, hidden_dim, out_dim)
        # Convert it to a heterogeneous model using metadata
        self.hetero_model = to_hetero(self.base_model, metadata=metadata)

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass for heterogeneous graph data.

        Args:
            x_dict (dict): Dictionary of node features for each node type.
            edge_index_dict (dict): Dictionary of edge indices for each edge type.

        Returns:
            dict: Node embeddings for each node type.
        """
        return self.hetero_model(x_dict, edge_index_dict)

# ------------------------- #
#     HeteroGraphConv       #
# ------------------------- #

class HeteroGraphConv(HeteroBaseModel):
    def __init__(self, metadata, input_dim, hidden_dim, out_dim):
        super(HeteroGraphConv, self).__init__(
            metadata,
            input_dim,
            hidden_dim,
            out_dim,
            GraphConvModel
        )

# --------------------- #
#       HeteroGAT       #
# --------------------- #

class HeteroGAT(HeteroBaseModel):
    def __init__(self, metadata, input_dim, hidden_dim, out_dim, num_heads=4):
        super(HeteroGAT, self).__init__(
            metadata,
            input_dim,
            hidden_dim * num_heads,
            out_dim,
            lambda in_dim, hidden_dim, out_dim: GAT(in_dim, hidden_dim, out_dim, num_heads=num_heads, add_self_loops=False),
        )

# --------------------- #
#      HeteroSAGE       #
# --------------------- #

class HeteroSAGE(HeteroBaseModel):
    def __init__(self, metadata, input_dim, hidden_dim, out_dim):
        super(HeteroSAGE, self).__init__(metadata, input_dim, hidden_dim, out_dim, SAGE)

# -------------------- #
#       HeteroGIN      #
# -------------------- #

class HeteroGIN(HeteroBaseModel):
    def __init__(self, metadata, input_dim, hidden_dim, out_dim):
        super(HeteroGIN, self).__init__(
            metadata,
            input_dim,
            hidden_dim,
            out_dim,
            lambda in_dim, hidden_dim, out_dim: GIN(in_dim, hidden_dim, out_dim)
        )
