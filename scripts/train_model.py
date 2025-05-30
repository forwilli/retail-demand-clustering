# Model Configuration for Joint Demand Forecasting and Clustering

model:
  # Architecture parameters
  input_features: 28              # Number of input features after engineering
  hidden_dim: 256                 # LSTM hidden dimension
  lstm_layers: 3                  # Number of LSTM layers
  forecast_layers: 3              # Number of forecasting head layers
  forecast_horizon: 1             # Number of time steps to predict
  n_clusters: 8                   # Number of product subtypes
  dropout: 0.3                    # Dropout probability
  cluster_alpha: 1.0              # Student's t-distribution parameter

training:
  # Training parameters
  batch_size: 64
  num_epochs: 150
  learning_rate: 0.001
  weight_decay: 0.01
  
  # Loss function weights
  lambda_cluster: 0.1             # Clustering loss weight
  lambda_entropy: 0.01            # Entropy regularization weight
  lambda_diversity: 0.05          # Cluster diversity weight
  
  # Optimization
  optimizer: "adamw"
  scheduler: "cosine"
  warmup_epochs: 10
  
  # Monitoring
  log_interval: 10
  save_interval: 50
  early_stopping_patience: 20

data:
  # Data processing parameters
  sequence_length: 30             # Input sequence length (days)
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  
  # Feature engineering
  lag_features: [1, 7, 14, 30]    # Lag periods for demand features
  rolling_windows: [7, 14, 30]    # Rolling statistics windows
  seasonal_features: true         # Include seasonal encoding
  promotion_features: true        # Include promotional features

hardware:
  device: "cuda"                  # "cuda" or "cpu"
  num_workers: 4                  # Data loader workers
  pin_memory: true               # Pin memory for faster GPU transfer

experiment:
  name: "baseline_experiment"
  seed: 42                       # Random seed for reproducibility
  save_dir: "results/"
  log_dir: "logs/"
