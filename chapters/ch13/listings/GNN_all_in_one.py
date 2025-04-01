import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Union, Dict, Optional, Tuple


class GraphAttentionHead(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Linear(feature_dim * 2, 1)

    def forward(self, node_features, neighbor_features):
        # Assuming neighbor_features has shape [batch, nodes, neighbors, features]
        # and node_features has shape [batch, nodes, features]

        # Expand node_features to broadcast against each neighbor
        node_features_expanded = node_features.unsqueeze(2)  # [batch, nodes, 1, features]

        # Concatenate node features with each neighbor's features
        combined = torch.cat([
            node_features_expanded.expand_as(neighbor_features),
            neighbor_features
        ], dim=-1)  # [batch, nodes, neighbors, 2*features]

        # Compute attention coefficients
        attention_scores = self.attention(combined).squeeze(-1)  # [batch, nodes, neighbors]
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to aggregate neighbor features
        weighted_neighbors = neighbor_features * attention_weights.unsqueeze(-1)
        aggregated = weighted_neighbors.sum(dim=2)  # [batch, nodes, features]

        return aggregated


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, "Input dimension must be divisible by number of heads"

        self.attention_heads = nn.ModuleList([
            GraphAttentionHead(self.head_dim)
            for _ in range(num_heads)
        ])

    def forward(self, node_features, neighbor_features):
        # Split features into chunks for each head
        node_chunks = torch.chunk(node_features, self.num_heads, dim=-1)
        neighbor_chunks = torch.chunk(neighbor_features, self.num_heads, dim=-1)

        head_outputs = []
        for i, head in enumerate(self.attention_heads):
            head_out = head(node_chunks[i], neighbor_chunks[i])
            head_outputs.append(head_out)

        return torch.cat(head_outputs, dim=-1)


class TransformerStyleGraphAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.query_transform = nn.Linear(feature_dim, feature_dim)
        self.key_transform = nn.Linear(feature_dim, feature_dim)
        self.value_transform = nn.Linear(feature_dim, feature_dim)

        # Output projection
        self.output_transform = nn.Linear(feature_dim, feature_dim)

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, 4 * feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * feature_dim, feature_dim)
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(self, node_features, neighbor_features, mask=None):
        batch_size, num_nodes = node_features.size(0), node_features.size(1)
        max_neighbors = neighbor_features.size(2)

        # Reshape node_features to broadcast against neighbors
        node_features_expanded = node_features.unsqueeze(2)

        # Multi-head attention
        Q = self.query_transform(node_features_expanded)  # [batch, nodes, 1, feature_dim]
        K = self.key_transform(neighbor_features)  # [batch, nodes, neighbors, feature_dim]
        V = self.value_transform(neighbor_features)  # [batch, nodes, neighbors, feature_dim]

        # Reshape for multi-head attention
        if self.num_heads > 1:
            Q = Q.view(batch_size, num_nodes, 1, self.num_heads, self.head_dim).transpose(2, 3)
            K = K.view(batch_size, num_nodes, max_neighbors, self.num_heads, self.head_dim).transpose(2, 3)
            V = V.view(batch_size, num_nodes, max_neighbors, self.num_heads, self.head_dim).transpose(2, 3)

        # Calculate attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [..., 1, neighbors]
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            # Expand mask for broadcasting: [batch, nodes, neighbors] -> [batch, nodes, 1, neighbors]
            if self.num_heads > 1:
                mask = mask.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
            else:
                mask = mask.unsqueeze(2)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Normalize attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, V)  # [..., 1, feature_dim]

        # Reshape back from multi-head format
        if self.num_heads > 1:
            attended_values = attended_values.transpose(2, 3).contiguous()
            attended_values = attended_values.view(batch_size, num_nodes, 1, self.feature_dim)

        # Remove the expanded dimension
        attended_values = attended_values.squeeze(2)  # [batch, nodes, feature_dim]

        # Output projection
        attended_values = self.output_transform(attended_values)

        # Residual connection and layer normalization
        attended_values = self.layer_norm(node_features + attended_values)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(attended_values)
        output = self.ffn_norm(attended_values + ffn_output)

        return output


class StructuralGraphTransformer(nn.Module):
    def __init__(self, feature_dim, num_heads, max_degree=10):
        super().__init__()
        self.structural_encoding = nn.Embedding(max_degree + 1, feature_dim)
        self.attention = MultiHeadGraphAttention(feature_dim, num_heads)

    def forward(self, node_features, neighbor_features, degrees):
        structural_features = self.structural_encoding(degrees)
        enhanced_features = node_features + structural_features

        return self.attention(enhanced_features, neighbor_features)


class GraphSAGEUpdate(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.update_nn = nn.Linear(input_dim * 2, hidden_dim)

    def forward(self, node_features, aggregated_neighbor_features):
        combined = torch.cat([node_features, aggregated_neighbor_features], dim=-1)

        updated = self.update_nn(combined)
        return F.relu(updated)


class MeanAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, neighbor_features, mask=None):
        # neighbor_features: [batch, nodes, neighbors, features]
        if mask is not None:
            # Apply mask for proper mean calculation (avoid counting padded neighbors)
            mask = mask.unsqueeze(-1)  # [batch, nodes, neighbors, 1]
            neighbor_features = neighbor_features * mask
            return neighbor_features.sum(dim=2) / (mask.sum(dim=2) + 1e-10)
        else:
            return neighbor_features.mean(dim=2)


class MaxAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, neighbor_features, mask=None):
        # neighbor_features: [batch, nodes, neighbors, features]
        if mask is not None:
            # Set padded neighbors to very negative values
            mask = mask.unsqueeze(-1)  # [batch, nodes, neighbors, 1]
            neighbor_features = neighbor_features * mask + (1 - mask) * (-1e9)

        return neighbor_features.max(dim=2)[0]


class GatedGraphUpdate(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.update_gate = nn.Linear(feature_dim * 2, feature_dim)
        self.transform = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, node_features, aggregated_features):
        gate_input = torch.cat([node_features, aggregated_features], dim=-1)
        update_gate = torch.sigmoid(self.update_gate(gate_input))

        combined = torch.cat([node_features, aggregated_features], dim=-1)
        candidate = torch.tanh(self.transform(combined))

        return (1 - update_gate) * node_features + update_gate * candidate


class JumpingKnowledge(nn.Module):
    def __init__(self, feature_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim,
            batch_first=True
        )

    def forward(self, layer_representations):
        stacked = torch.stack(layer_representations, dim=1)

        output, _ = self.lstm(stacked)

        return output[:, -1, :]


class FlexibleGNN(nn.Module):
    """
    A flexible Graph Neural Network that can switch between different architectures.

    Architecture options:
    - 'gat': Graph Attention Network with multi-head attention
    - 'transformer': Transformer-style graph attention
    - 'structural': Structural Graph Transformer with degree-based encodings
    - 'graphsage': GraphSAGE with mean/max aggregation and linear update
    - 'gated': Gated Graph Network with GRU-like update mechanism
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 2,
            architecture: str = 'transformer',
            num_heads: int = 4,
            dropout: float = 0.1,
            max_degree: int = 10,
            aggregator: str = 'mean',
            jk_mode: bool = False,
            residual: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.architecture = architecture
        self.num_heads = num_heads
        self.jk_mode = jk_mode
        self.residual = residual

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Create aggregators if needed for certain architectures
        if architecture == 'graphsage':
            if aggregator == 'mean':
                self.aggregator = MeanAggregator()
            elif aggregator == 'max':
                self.aggregator = MaxAggregator()
            else:
                raise ValueError(f"Unknown aggregator: {aggregator}")

        # Create graph layers based on selected architecture
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if architecture == 'gat':
                self.layers.append(MultiHeadGraphAttention(hidden_dim, num_heads))
            elif architecture == 'transformer':
                self.layers.append(TransformerStyleGraphAttention(hidden_dim, num_heads, dropout))
            elif architecture == 'structural':
                self.layers.append(StructuralGraphTransformer(hidden_dim, num_heads, max_degree))
            elif architecture == 'graphsage':
                self.layers.append(GraphSAGEUpdate(hidden_dim, hidden_dim))
            elif architecture == 'gated':
                self.layers.append(GatedGraphUpdate(hidden_dim))
            else:
                raise ValueError(f"Unknown architecture: {architecture}")

        # Jumping Knowledge layer if enabled
        if jk_mode:
            self.jk_layer = JumpingKnowledge(hidden_dim, num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def _check_dimensions(self, node_features, neighbor_features, mask=None, degrees=None):
        """Validate input dimensions and reshape if necessary"""
        if len(node_features.shape) == 2:  # [nodes, features]
            node_features = node_features.unsqueeze(0)  # [1, nodes, features]

        if len(neighbor_features.shape) == 3:  # [nodes, neighbors, features]
            neighbor_features = neighbor_features.unsqueeze(0)  # [1, nodes, neighbors, features]

        if mask is not None and len(mask.shape) == 2:  # [nodes, neighbors]
            mask = mask.unsqueeze(0)  # [1, nodes, neighbors]

        if degrees is not None and len(degrees.shape) == 1:  # [nodes]
            degrees = degrees.unsqueeze(0)  # [1, nodes]

        return node_features, neighbor_features, mask, degrees

    def forward(
            self,
            node_features: torch.Tensor,
            neighbor_features: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            degrees: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the GNN.

        Args:
            node_features: Node feature tensor [batch_size, num_nodes, input_dim] or [num_nodes, input_dim]
            neighbor_features: Neighborhood features [batch_size, num_nodes, max_neighbors, input_dim] or
                               [num_nodes, max_neighbors, input_dim]
            mask: Optional mask for padding in neighbor lists [batch_size, num_nodes, max_neighbors] or
                  [num_nodes, max_neighbors]
            degrees: Node degrees for structural encoding [batch_size, num_nodes] or [num_nodes]

        Returns:
            Updated node features [batch_size, num_nodes, output_dim]
        """
        # Check and normalize dimensions
        node_features, neighbor_features, mask, degrees = self._check_dimensions(
            node_features, neighbor_features, mask, degrees
        )

        # Input projection
        x = self.input_proj(node_features)
        x = F.relu(x)
        x = self.dropout(x)

        # Store representations from each layer if using JK
        layer_outputs = [x] if self.jk_mode else []

        # Process through GNN layers
        for i, layer in enumerate(self.layers):
            if self.architecture == 'graphsage':
                # For GraphSAGE, first aggregate neighbors then update
                aggregated = self.aggregator(neighbor_features, mask)
                x_new = layer(x, aggregated)
            elif self.architecture == 'gated':
                # For gated updates, first aggregate neighbors then apply gate
                aggregated = self.aggregator(neighbor_features, mask)
                x_new = layer(x, aggregated)
            elif self.architecture == 'structural':
                # For structural transformer, need degrees
                if degrees is None:
                    raise ValueError("Degrees tensor is required for structural transformer")
                x_new = layer(x, neighbor_features, degrees)
            else:
                # GAT and Transformer architectures
                x_new = layer(x, neighbor_features, mask)

            # Apply residual connection if enabled (except for first layer)
            if self.residual and i > 0:
                x = x_new + x
            else:
                x = x_new

            x = self.dropout(x)

            if self.jk_mode:
                layer_outputs.append(x)

        # Apply Jumping Knowledge if enabled
        if self.jk_mode:
            x = self.jk_layer(layer_outputs)

        # Output projection
        output = self.output_proj(x)

        return output


# Example usage:
def create_flexible_gnn(config):
    """Factory function to create a FlexibleGNN with the specified configuration"""
    return FlexibleGNN(
        input_dim=config.get('input_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        output_dim=config.get('output_dim', 64),
        num_layers=config.get('num_layers', 2),
        architecture=config.get('architecture', 'transformer'),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1),
        max_degree=config.get('max_degree', 10),
        aggregator=config.get('aggregator', 'mean'),
        jk_mode=config.get('jk_mode', False),
        residual=config.get('residual', True)
    )


# Example configurations
gat_config = {
    'input_dim': 64,
    'hidden_dim': 128,
    'output_dim': 32,
    'architecture': 'gat',
    'num_heads': 8
}

transformer_config = {
    'input_dim': 64,
    'hidden_dim': 128,
    'output_dim': 32,
    'architecture': 'transformer',
    'num_heads': 4,
    'num_layers': 3
}

sage_config = {
    'input_dim': 64,
    'hidden_dim': 128,
    'output_dim': 32,
    'architecture': 'graphsage',
    'aggregator': 'mean'
}

structural_config = {
    'input_dim': 64,
    'hidden_dim': 128,
    'output_dim': 32,
    'architecture': 'structural',
    'num_heads': 4,
    'max_degree': 15
}

gated_config = {
    'input_dim': 64,
    'hidden_dim': 128,
    'output_dim': 32,
    'architecture': 'gated',
    'aggregator': 'max',
    'jk_mode': True
}


# Usage example:
def run_example():
    # Create a model with the transformer config
    model = create_flexible_gnn(transformer_config)

    # Generate some example data
    batch_size = 2
    num_nodes = 10
    max_neighbors = 5
    input_dim = 64

    # Create random node features: [batch_size, num_nodes, input_dim]
    node_features = torch.randn(batch_size, num_nodes, input_dim)

    # Create random neighbor features: [batch_size, num_nodes, max_neighbors, input_dim]
    neighbor_features = torch.randn(batch_size, num_nodes, max_neighbors, input_dim)

    # Create a mask for varying number of neighbors: [batch_size, num_nodes, max_neighbors]
    mask = torch.ones(batch_size, num_nodes, max_neighbors)
    for b in range(batch_size):
        for n in range(num_nodes):
            # Randomly mask some neighbors
            num_valid = torch.randint(1, max_neighbors + 1, (1,)).item()
            mask[b, n, num_valid:] = 0

    # Create random degree information: [batch_size, num_nodes]
    degrees = torch.randint(0, 10, (batch_size, num_nodes))

    # Forward pass
    output = model(node_features, neighbor_features, mask, degrees)

    print(f"Output shape: {output.shape}")
    return output


if __name__ == "__main__":
    run_example()