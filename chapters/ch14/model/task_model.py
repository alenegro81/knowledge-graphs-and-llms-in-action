import torch
import torch.nn.functional as F
from ch14.model.util_model import MovieLensEmbedding, DotProduct

# ---------------- #
# Node Classifiers #
# ---------------- #

class NodeClassifier(torch.nn.Module):
    def __init__(self, gnn_model):
        """
        Initializes the NodeClassifier with a given GNN model.

        Args:
            gnn_model (torch.nn.Module): A GNN model (e.g., GCN, SAGE, GAT, etc.).
                                         This model computes node embeddings.
        """
        super().__init__()
        self.gnn = gnn_model  # Backbone GNN for learning node representations

    def forward(self, x, edge_index):
        """
        Forward pass for the node classification task.

        Args:
            x (torch.Tensor): Node feature matrix of shape (num_nodes, num_features).
                              Each row represents the feature vector for a node.
            edge_index (torch.Tensor): Graph connectivity in COO format,
                                        with shape (2, num_edges).
                                        Each column represents an edge (source, target).

        Returns:
            torch.Tensor: Log-softmax of class probabilities for each node,
                          with shape (num_nodes, num_classes).
        """
        # Compute node embeddings using the GNN model
        x = self.gnn(x, edge_index)
        
        # Apply log-softmax to output class probabilities for each node
        return F.log_softmax(x, dim=1)

# ---------------- #
# Link Predictors  #
# ---------------- #

class MovieLensLinkPredictor(torch.nn.Module):
    """
    Link prediction model for MovieLens data using heterogeneous graph neural networks.

    Args:
        gnn_model (torch.nn.Module): A GNN model for heterogeneous graph data.
        data (dict): Dictionary containing graph data. Expected keys:
            - "user": User node data.
            - "movie": Movie node data.
            - "user", "rates", "movie": Edge data for the "rates" relation.
            - edge_index_dict: Dictionary mapping edge types to edge indices.
        hidden_channels (int): Dimension of the hidden channels used in embeddings and GNN layers.
    """
    def __init__(self, gnn_model, data, hidden_channels):
        super().__init__()
        
        # Initialize embeddings for the user and movie nodes
        self.embedding = MovieLensEmbedding(
            data["user"].num_nodes,
            data["movie"].num_nodes,
            hidden_channels
        )

        # Use the provided GNN model
        self.gnn = gnn_model(data.metadata(),
                              hidden_channels,
                              hidden_channels,
                              hidden_channels)

        # Define the final classifier
        self.classifier = DotProduct()

    def forward(self, data):
        """
        Forward pass to compute link predictions.

        Args:
            data (dict): Dictionary containing graph data. Expected keys:
                - "user": User node data.
                - "movie": Movie node data.
                - "user", "rates", "movie": Edge data for the "rates" relation.
                - edge_index_dict: Dictionary mapping edge types to edge indices.

        Returns:
            torch.Tensor: Predictions for the "rates" relation.
        """
        # Extract user and movie node features
        x_dict = self.embedding(data)

        # Pass node features through the GNN
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        # Compute predictions using the classifier
        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )

        return pred
