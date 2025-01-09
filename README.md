# Synthetic Survey Data Generation and Evaluation

## Overview
Repository for replicating "Synthetic Survey Data Generation and Evaluation."

## Content Description

### Generator
- `Synthpop`: Synthetic data generation using traditional population modeling.
- `CTGAN`: Generative adversarial network for mixed-type tabular data generation.
- `TVAE`: Variational autoencoder for mixed-type tabular data generation.
- `REaDTabFormer`: Transformer-based synthetic tabular data generation.
- `SMOTE (baseline)`: Synthetic minority over-sampling technique for generating synthetic samples by interpolating between the selected instance and its k-nearest neighbors.

### Syntheval
- **Metrics**: 
  - General utility.
  - Target-specific utility.
  - Privacy measures.

- **Result Parser**: Parses and consolidates results from `Syntheval`.
