"""
Joint Demand Forecasting and Clustering Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalDemandEncoder(nn.Module):
    """Multi-layer LSTM encoder for temporal demand patterns"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.3):
        super(TemporalDemandEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Normalization and projection
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        
    def forward(self, sequences):
        """
        Args:
            sequences: [batch_size, seq_len, input_dim]
        Returns:
            lstm_outputs: [batch_size, seq_len, hidden_dim]
            embeddings: [batch_size, hidden_dim]
        """
        lstm_output, (final_hidden, _) = self.lstm(sequences)
        
        # Extract final embedding
        embeddings = final_hidden[-1]  # Last layer output
        embeddings = self.layer_norm(embeddings)
        embeddings = self.activation(self.projection(embeddings))
        
        return lstm_output, embeddings


class SoftClusteringLayer(nn.Module):
    """Differentiable soft clustering using Student's t-distribution"""
    
    def __init__(self, embedding_dim, n_clusters, alpha=1.0):
        super(SoftClusteringLayer, self).__init__()
        
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # Learnable cluster centroids
        self.centroids = nn.Parameter(
            torch.randn(n_clusters, embedding_dim) * 0.1
        )
        
    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch_size, embedding_dim]
        Returns:
            soft_assignments: [batch_size, n_clusters]
            distances: [batch_size, n_clusters]
        """
        # Compute distances to centroids
        expanded_embeddings = embeddings.unsqueeze(1)  # [B, 1, D]
        expanded_centroids = self.centroids.unsqueeze(0)  # [1, K, D]
        
        distances = torch.norm(
            expanded_embeddings - expanded_centroids, 
            dim=2, p=2
        )
        
        # Student's t-distribution for soft assignments
        numerator = (1.0 + distances**2 / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        soft_assignments = numerator / torch.sum(numerator, dim=1, keepdim=True)
        
        return soft_assignments, distances


class DemandForecastingHead(nn.Module):
    """Multi-layer forecasting head with residual connections"""
    
    def __init__(self, hidden_dim, output_dim, num_layers=3, dropout=0.2):
        super(DemandForecastingHead, self).__init__()
        
        layers = []
        input_dim = hidden_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim // 2
        
        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Skip connection
        self.skip_connection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, embeddings):
        main_output = self.network(embeddings)
        skip_output = self.skip_connection(embeddings)
        return main_output + 0.1 * skip_output


class JointDemandClusteringModel(nn.Module):
    """Complete joint model for demand forecasting and clustering"""
    
    def __init__(self, config):
        super(JointDemandClusteringModel, self).__init__()
        
        self.config = config
        
        # Core components
        self.encoder = TemporalDemandEncoder(
            input_dim=config.input_features,
            hidden_dim=config.hidden_dim,
            num_layers=config.lstm_layers,
            dropout=config.dropout
        )
        
        self.clustering = SoftClusteringLayer(
            embedding_dim=config.hidden_dim,
            n_clusters=config.n_clusters,
            alpha=config.cluster_alpha
        )
        
        self.forecasting = DemandForecastingHead(
            hidden_dim=config.hidden_dim,
            output_dim=config.forecast_horizon,
            num_layers=config.forecast_layers,
            dropout=config.dropout
        )
        
    def forward(self, sequences, return_embeddings=False):
        """
        Forward pass through complete model
        
        Args:
            sequences: [batch_size, seq_len, input_features]
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary containing predictions and cluster assignments
        """
        # Encode temporal sequences
        lstm_outputs, embeddings = self.encoder(sequences)
        
        # Generate demand forecasts
        demand_predictions = self.forecasting(embeddings)
        
        # Compute cluster assignments
        cluster_probs, cluster_distances = self.clustering(embeddings)
        
        output = {
            'demand_predictions': demand_predictions,
            'cluster_probabilities': cluster_probs,
            'cluster_distances': cluster_distances,
        }
        
        if return_embeddings:
            output.update({
                'embeddings': embeddings,
                'lstm_outputs': lstm_outputs
            })
            
        return output
    
    def get_cluster_assignments(self, sequences):
        """Get hard cluster assignments for analysis"""
        with torch.no_grad():
            outputs = self.forward(sequences)
            return torch.argmax(outputs['cluster_probabilities'], dim=1)


if __name__ == "__main__":
    # Simple test
    from types import SimpleNamespace
    
    config = SimpleNamespace(
        input_features=28,
        hidden_dim=256,
        lstm_layers=3,
        n_clusters=8,
        cluster_alpha=1.0,
        forecast_horizon=1,
        forecast_layers=3,
        dropout=0.3
    )
    
    model = JointDemandClusteringModel(config)
    test_input = torch.randn(32, 30, 28)  # batch_size=32, seq_len=30
    
    outputs = model(test_input)
    
    print("Model test successful!")
    print(f"Demand predictions shape: {outputs['demand_predictions'].shape}")
    print(f"Cluster probabilities shape: {outputs['cluster_probabilities'].shape}")
