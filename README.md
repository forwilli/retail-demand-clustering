A joint learning approach for demand forecasting and product clustering in retail inventory management using LSTM neural networks.

## Overview

This repository implements a novel approach that simultaneously:
- Predicts future product demand using LSTM temporal encoders
- Discovers meaningful product subtypes through soft clustering
- Integrates both objectives in an end-to-end trainable framework

## Architecture

- **LSTM Encoder**: Multi-layer temporal sequence modeling
- **Soft Clustering**: Differentiable product subtype discovery  
- **Forecasting Head**: Multi-step demand prediction
- **Joint Loss**: Combined optimization objective

## Quick Start

### Installation
```bash
git clone https://github.com/YOUR-USERNAME/retail-demand-clustering.git
cd retail-demand-clustering
pip install -r requirements.txt
