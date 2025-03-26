import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GraphConv

class BaseGraphModel(torch.nn.Module):
    """
    A generic graph model with two convolution layers and ReLU activation.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden feature dimension.
        out_dim (int): Output feature dimension.
        conv_layer (torch.nn.Module): Graph convolution layer class (e.g., GCNConv, SAGEConv, GATConv, GINConv).
        **conv_kwargs: Additional keyword arguments for the convolution layer.
    """
    def __init__(self, input_dim, hidden_dim, out_dim, conv_layer, **conv_kwargs):
        super(BaseGraphModel, self).__init__()

        # Initialize the first convolution layer
        self.conv1 = conv_layer(input_dim, hidden_dim, **conv_kwargs)
        # Initialize the second convolution layer
        self.conv2 = conv_layer(hidden_dim, out_dim, **conv_kwargs)

    def forward(self, x, edge_index):
        """
        Forward pass for the graph model.

        Args:
            x (Tensor): Input node features.
            edge_index (Tensor): Edge indices in COO format.

        Returns:
            Tensor: Output node features.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# --------- #
# GCN Model #
# --------- #

class GCN(BaseGraphModel):
    def __init__(self, input_dim, hidden_dim, out_dim, add_self_loops=True):
        super(GCN, self).__init__(
            input_dim, 
            hidden_dim, 
            out_dim, 
            GCNConv,
            add_self_loops=add_self_loops
        )

# --------- #
# GraphConv #
# --------- #

class GraphConvModel(BaseGraphModel):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(GraphConvModel, self).__init__(
            input_dim,
            hidden_dim,
            out_dim,
            GraphConv)

# --------- #
# GAT Model #
# --------- #

class GAT(BaseGraphModel):
    def __init__(self, input_dim, hidden_dim, out_dim, num_heads=8, add_self_loops=True):
        # Adjust hidden_dim to account for concatenated heads in GAT
        concat_hidden_dim = hidden_dim * num_heads

        # Define the first GATConv layer
        conv1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            add_self_loops=add_self_loops,
        )

        # Define the second GATConv layer
        conv2 = GATConv(
            in_channels=concat_hidden_dim,
            out_channels=out_dim,
            heads=1,  # Single head for the final layer
            add_self_loops=add_self_loops,
            concat=False,  # Disable concatenation for the final output
        )

        # Initialize the parent class but skip conv_layer initialization
        super(BaseGraphModel, self).__init__()
        self.conv1 = conv1
        self.conv2 = conv2

# ---------- #
# SAGE Model #
# ---------- #

class SAGE(BaseGraphModel):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(SAGE, self).__init__(
            input_dim,
            hidden_dim,
            out_dim,
            SAGEConv)

# --------- #
# GIN Model #
# --------- #

class GIN(BaseGraphModel):
    def __init__(self, input_dim, hidden_dim, out_dim):
        # Create the MLP for the GINConv layers
        conv1 = torch.nn.Linear(input_dim, hidden_dim)
        conv2 = torch.nn.Linear(hidden_dim, out_dim)

        # Initialize the parent class but skip conv_layer initialization
        super(BaseGraphModel, self).__init__()

        # Override conv1 and conv2 for GIN-specific logic
        self.conv1 = GINConv(conv1)
        self.conv2 = GINConv(conv2)
