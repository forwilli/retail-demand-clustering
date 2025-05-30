
import argparse
import logging
import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.joint_model import JointDemandClusteringModel


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_random_seeds(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)


def create_dummy_data(config):
    """Create dummy data for testing purposes"""
    batch_size = config['training']['batch_size']
    seq_len = config['data']['sequence_length']
    input_features = config['model']['input_features']
    
    # Generate synthetic data
    X = torch.randn(batch_size * 10, seq_len, input_features)
    y = torch.randn(batch_size * 10, config['model']['forecast_horizon'])
    
    # Simple train/val split
    split_idx = int(len(X) * 0.8)
    
    train_data = torch.utils.data.TensorDataset(X[:split_idx], y[:split_idx])
    val_data = torch.utils.data.TensorDataset(X[split_idx:], y[split_idx:])
    
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader


def simple_train_loop(model, train_loader, val_loader, config, logger):
    """Simple training loop for demonstration"""
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    criterion = torch.nn.MSELoss()
    
    num_epochs = min(5, config['training']['num_epochs'])  # Limit for demo
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            
            # Simple forecasting loss only for demo
            loss = criterion(outputs['demand_predictions'], targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 5 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs['demand_predictions'], targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f'Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
    
    logger.info("Training completed!")
    
    # Save model
    os.makedirs(config['experiment']['save_dir'], exist_ok=True)
    torch.save(model.state_dict(), f"{config['experiment']['save_dir']}/model_demo.pth")
    logger.info("Model saved!")


def main():
    parser = argparse.ArgumentParser(description='Train joint demand forecasting model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging("DEBUG" if args.debug else "INFO")
    config = load_config(args.config)
    set_random_seeds(config['experiment']['seed'])
    
    logger.info("Configuration loaded successfully")
    
    # Create model
    from types import SimpleNamespace
    model_config = SimpleNamespace(**config['model'])
    model = JointDemandClusteringModel(model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    
    # Create dummy data (replace with real data loading)
    logger.info("Creating dummy data for demonstration...")
    train_loader, val_loader = create_dummy_data(config)
    logger.info(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Train model
    simple_train_loop(model, train_loader, val_loader, config, logger)


if __name__ == "__main__":
    main()
