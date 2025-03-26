import torch

class MovieLensEmbedding(torch.nn.Module):
    """
    Embedding model for MovieLens data. Maps user and movie information to a shared latent space.

    Args:
        user_input_dim (int): Number of unique users (input dimension for user embedding).
        movie_input_dim (int): Number of unique movies (input dimension for movie embedding).
        out_dim (int): Dimension of the shared latent space (output dimension for embeddings).
    """
    def __init__(self, user_input_dim, movie_input_dim, out_dim):
        super().__init__()
        self.movie_lin = torch.nn.Linear(20, out_dim)  # 20 is the number of movie genres
        self.user_emb = torch.nn.Embedding(user_input_dim, out_dim)
        self.movie_emb = torch.nn.Embedding(movie_input_dim, out_dim)

    def forward(self, data):
        """
        Forward pass to compute user and movie embeddings.

        Args:
            data (dict): Dictionary containing user and movie information. Expected keys:
                - "user": Data for users, with attribute `node_id` (user IDs).
                - "movie": Data for movies, with attributes `x` (genre features) and `node_id` (movie IDs).

        Returns:
            dict: A dictionary with embeddings for "user" and "movie". Keys:
                - "user": User embeddings of shape `(num_users, out_dim)`.
                - "movie": Movie embeddings of shape `(num_movies, out_dim)`.
        """
        return {
            "user": self.user_emb(data["user"].node_id),
            "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        }


class DotProduct(torch.nn.Module):
    def forward(self, x_src, x_dst, edge_label_index):
        """
        Compute edge-level predictions using dot product between source and destination embeddings.

        Args:
            x_src (Tensor): Node embeddings for source nodes (e.g., users).
            x_dst (Tensor): Node embeddings for destination nodes (e.g., movies).
            edge_label_index (Tensor): Edge indices of shape (2, num_edges), specifying
                                       source and destination nodes for each edge.

        Returns:
            Tensor: Edge-level predictions of shape (num_edges,).
        """
        # Extract node embeddings for the edges
        edge_feat_src = x_src[edge_label_index[0]]  # Source node embeddings
        edge_feat_dst = x_dst[edge_label_index[1]]  # Destination node embeddings

        # Compute predictions using dot product between source and destination embeddings
        return (edge_feat_src * edge_feat_dst).sum(dim=-1)