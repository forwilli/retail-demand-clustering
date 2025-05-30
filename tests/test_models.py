
import unittest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.joint_model import JointDemandClusteringModel
from types import SimpleNamespace


class TestJointModel(unittest.TestCase):
    """Test cases for the joint demand clustering model"""
    
    def setUp(self):
        """Setup test configuration and model"""
        self.config = SimpleNamespace(
            input_features=28,
            hidden_dim=64,  # Smaller for testing
            lstm_layers=2,
            n_clusters=4,
            cluster_alpha=1.0,
            forecast_horizon=1,
            forecast_layers=2,
            dropout=0.1
        )
        
        self.model = JointDemandClusteringModel(self.config)
        self.batch_size = 8
        self.seq_len = 30
        
    def test_model_forward_pass(self):
        """Test that model forward pass works"""
        # Create test input
        test_input = torch.randn(self.batch_size, self.seq_len, self.config.input_features)
        
        # Forward pass
        outputs = self.model(test_input)
        
        # Check output shapes
        self.assertEqual(
            outputs['demand_predictions'].shape,
            (self.batch_size, self.config.forecast_horizon)
        )
        self.assertEqual(
            outputs['cluster_probabilities'].shape,
            (self.batch_size, self.config.n_clusters)
        )
        
        # Check that cluster probabilities sum to 1
        cluster_sums = torch.sum(outputs['cluster_probabilities'], dim=1)
        self.assertTrue(torch.allclose(cluster_sums, torch.ones(self.batch_size), atol=1e-6))
        
    def test_cluster_assignments(self):
        """Test cluster assignment functionality"""
        test_input = torch.randn(self.batch_size, self.seq_len, self.config.input_features)
        
        assignments = self.model.get_cluster_assignments(test_input)
        
        # Check output shape and range
        self.assertEqual(assignments.shape, (self.batch_size,))
        self.assertTrue(torch.all(assignments >= 0))
        self.assertTrue(torch.all(assignments < self.config.n_clusters))
        
    def test_model_parameters(self):
        """Test that model has trainable parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)


if __name__ == '__main__':
    unittest.main()
