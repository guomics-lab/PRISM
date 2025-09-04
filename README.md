./src/PRISM.sh --data-path data/Alzheimer.csv --device cuda --gpu-id 1
# PRISM - Protein Data Imputation Framework

PRISM is a deep learning framework for missing protein data imputation, supporting both **DMF (Deep Matrix Factorization)** and **DCAE (Dilated Convolutional AutoEncoder)** methods.

## Features

- **Multiple Imputation Methods**: Support for DMF and DCAE algorithms
- **Flexible Architecture**: Configurable network architectures and hyperparameters
- **GPU Acceleration**: CUDA support with specific GPU selection
- **Comprehensive Logging**: TensorBoard integration for training monitoring
- **Early Stopping**: Prevent overfitting with configurable patience
- **Batch Processing**: Efficient batch training with customizable batch sizes

## Installation

### Requirements

- Python 3.12+
- CUDA-capable GPU (optional, but recommended)

### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch pytorch-lightning pandas numpy matplotlib seaborn tensorboard scipy scikit-learn
```

## Usage

### Quick Start

```bash
# Basic usage with DMF method
./src/PRISM.sh --data-path data/your_dataset.csv

# Use DCAE method with GPU 1
./src/PRISM.sh --data-path data/Alzheimer.csv --method DCAE --device cuda --gpu-id 1

# Custom parameters with early stopping
./src/PRISM.sh --data-path data/your_dataset.csv \
    --method DMF \
    --hidden-dims 512 256 128 \
    --embedding-dim 128 \
    --early-stopping \
    --max-epochs 100
```

### Available Parameters

#### Required Arguments
- `--data-path PATH`: Path to input CSV file

#### Method Selection
- `--method {DMF,DCAE}`: Imputation method (default: DMF)

#### General Network Parameters
- `--hidden-dims DIMS`: Hidden layer dimensions, space-separated (default: "256 128")
- `--batch-size SIZE`: Batch size for training (default: 1024)
- `--learning-rate RATE`: Learning rate (default: 0.001)
- `--weight-decay DECAY`: Weight decay for optimizer (default: 0.00001)
- `--gradient-clip VALUE`: Gradient clipping value (default: 1.0)

#### DMF Specific Parameters
- `--embedding-dim DIM`: Embedding dimension (default: 64)

#### DCAE Specific Parameters
- `--latent-dim DIM`: Latent dimension (default: 64)
- `--num-encoder-blocks NUM`: Number of encoder blocks (default: 2)
- `--num-decoder-blocks NUM`: Number of decoder blocks (default: 2)
- `--dilation VALUE`: Dilation factor (default: 2)

#### Loss Weights
- `--mask-weight WEIGHT`: Weight for mask prediction loss (default: 0.5)
- `--reconstruction-weight WEIGHT`: Weight for reconstruction loss (default: 1.0)

#### Training Control
- `--max-epochs EPOCHS`: Maximum training epochs (default: 200)
- `--early-stopping`: Enable early stopping
- `--patience PATIENCE`: Patience for early stopping (default: 20)

#### Device Settings
- `--device {cpu,cuda,auto}`: Device to use (default: auto)
- `--gpu-id ID`: Specific GPU ID to use (0, 1, etc.)

#### Output Settings
- `--results-dir DIR`: Directory for saving results (default: ./results)
- `--log-interval INTERVAL`: Logging interval in steps (default: 50)
- `--progress-bar`: Show progress bar during training

## Data Format

The input CSV file should have the following format:
- First row: Header (will be skipped)
- First column: Sample IDs/names (will be skipped)
- Remaining columns: Protein expression data
- Missing values: Use 0, negative values, or NaN

Example:
```csv
Sample_ID,Protein_1,Protein_2,Protein_3,...
Sample_001,1.23,0.45,NaN,...
Sample_002,2.34,0,1.67,...
Sample_003,1.45,1.23,2.89,...
```

## Output Files

PRISM generates the following outputs in the results directory:

```
results/
├── checkpoints/           # Model checkpoints
├── logs/                 # TensorBoard logs
└── outputs/
    └── {METHOD}_{DATASET}_{TIMESTAMP}/
        ├── config.json          # Training configuration
        ├── imputed_data.csv     # Imputed protein data
        ├── training_metrics.csv # Training loss history
        └── model_final.ckpt    # Final trained model
```

## Examples

### Example 1: DMF with Custom Architecture
```bash
./src/PRISM.sh --data-path data/Alzheimer.csv \
    --method DMF \
    --hidden-dims 512 256 128 64 \
    --embedding-dim 128 \
    --mask-weight 0.3 \
    --learning-rate 0.0005 \
    --max-epochs 150 \
    --early-stopping \
    --progress-bar
```

### Example 2: DCAE with GPU Acceleration
```bash
./src/PRISM.sh --data-path data/Alzheimer.csv \
    --method DCAE \
    --device cuda \
    --gpu-id 1 \
    --latent-dim 128 \
    --num-encoder-blocks 3 \
    --num-decoder-blocks 3 \
    --dilation 4 \
    --batch-size 512
```

### Example 3: CPU Training with Custom Output Directory
```bash
./src/PRISM.sh --data-path data/Alzheimer.csv \
    --device cpu \
    --results-dir ./my_results \
    --max-epochs 50 \
    --log-interval 10
```

## Method Descriptions

### DMF (Deep Matrix Factorization)
- Uses row and column embeddings to capture latent patterns
- Suitable for collaborative filtering-style missing data
- Good for datasets with structured missing patterns

### DCAE (Dilated Convolutional AutoEncoder)
- Uses dilated convolutions to capture long-range dependencies
- Suitable for sequential or structured protein data
- Better for complex missing data patterns

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir results/logs
```

### Training Metrics
Monitor the following metrics:
- `train_loss`: Overall training loss
- `reconstruction_loss`: Data reconstruction quality
- `mask_loss`: Missing data pattern prediction accuracy

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `--batch-size`
   - Use `--device cpu` for CPU training

2. **Shape Mismatch Errors**
   - Check CSV format (ensure first column is skipped)
   - Verify data contains only numeric values

3. **Slow Training**
   - Use GPU acceleration with `--device cuda`
   - Increase `--batch-size` if memory allows

4. **Poor Performance**
   - Adjust `--mask-weight` (try 0.1-0.8)
   - Experiment with different `--hidden-dims`
   - Enable `--early-stopping`

### Getting Help

For help with parameters:
```bash
./src/PRISM.sh --help
```

## File Structure

```
PRISM/
├── README.md
├── requirements.txt
├── src/
│   ├── PRISM.sh              # Main training script
│   ├── train.py              # Python training interface
│   ├── datasets.py           # Data loading utilities
│   ├── models.py             # Model architectures
│   └── methods/
│       ├── DMF.py           # DMF implementation
│       └── DCAE.py          # DCAE implementation
└── data/
    └── your_datasets.csv
```

## Citation

If you use PRISM in your research, please cite:

```bibtex
@software{prism2024,
  title={PRISM: Protein Data Imputation Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/PRISM}
}
```

## License

This project is licensed under the MIT License 


## Changelog

### Version 0.0.1
- Initial release
- Support for DMF and DCAE methods
- GPU acceleration
- Comprehensive parameter configuration
- TensorBoard integration