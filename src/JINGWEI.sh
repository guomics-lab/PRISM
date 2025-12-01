#!/bin/bash
# filepath: /home/lizhaoxing/PRISM/src/PRISM.sh

# PRISM Training Script
# Supports DMF and DCAE imputation methods

# Set default values
METHOD="DMF"
DATA_PATH=""
HIDDEN_DIMS="256 128"
EMBEDDING_DIM=64
LATENT_DIM=64
NUM_ENCODER_BLOCKS=2
NUM_DECODER_BLOCKS=2
DILATION=2
MASK_WEIGHT=0.5
RECONSTRUCTION_WEIGHT=1.0
MAX_EPOCHS=200
BATCH_SIZE=1024
LEARNING_RATE=0.001
WEIGHT_DECAY=0.00001
GRADIENT_CLIP=1.0
EARLY_STOPPING=false
PATIENCE=20
RESULTS_DIR="./results"
LOG_INTERVAL=50
PROGRESS_BAR=false
DEVICE="auto"
GPU_ID=""

# Display help information
show_help() {
    cat << EOF
PRISM Training Script - Supports DMF and DCAE imputation methods

Usage:
    $0 --data-path <path> [options]

Required Arguments:
    --data-path PATH            Path to input CSV file

Method Selection:
    --method METHOD             Imputation method (DMF|DCAE, default: DMF)

General Network Parameters:
    --hidden-dims DIMS          Hidden layer dimensions, space-separated (default: "256 128")
    --batch-size SIZE           Batch size for training (default: 1024)
    --learning-rate RATE        Learning rate (default: 0.001)
    --weight-decay DECAY        Weight decay for optimizer (default: 0.00001)
    --gradient-clip VALUE       Gradient clipping value (default: 1.0)

DMF Specific Parameters:
    --embedding-dim DIM         Embedding dimension (default: 64)

DCAE Specific Parameters:
    --latent-dim DIM            Latent dimension (default: 64)
    --num-encoder-blocks NUM    Number of encoder blocks (default: 2)
    --num-decoder-blocks NUM    Number of decoder blocks (default: 2)
    --dilation VALUE            Dilation factor (default: 2)

Loss Weights:
    --mask-weight WEIGHT        Weight for mask prediction loss (default: 0.5)
    --reconstruction-weight W   Weight for reconstruction loss (default: 1.0)

Training Control:
    --max-epochs EPOCHS         Maximum training epochs (default: 200)
    --early-stopping            Enable early stopping
    --patience PATIENCE         Patience for early stopping (default: 20)

Output Settings:
    --results-dir DIR           Directory for saving results (default: ./results)
    --log-interval INTERVAL     Logging interval in steps (default: 50)
    --progress-bar              Show progress bar during training

Other:
    --help, -h                  Show this help message

Device Settings:
    --device DEVICE             Device to use (cpu|cuda|auto, default: auto)
    --gpu-id ID                 Specific GPU ID to use (0, 1, etc.)


Examples:
    # Train with DMF method
    $0 --data-path data/dataset.csv --method DMF --mask-weight 0.3 --max-epochs 100

    # Train with DCAE method, enable early stopping and progress bar
    $0 --data-path data/dataset.csv --method DCAE --early-stopping --progress-bar

    # Custom network architecture
    $0 --data-path data/dataset.csv --hidden-dims "512 256 128" --embedding-dim 128

    # Use specific GPU
    $0 --data-path data/dataset.csv --device cuda --gpu-id 1
    
    # Use CPU
    $0 --data-path data/dataset.csv --device cpu

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --hidden-dims)
            HIDDEN_DIMS="$2"
            shift 2
            ;;
        --embedding-dim)
            EMBEDDING_DIM="$2"
            shift 2
            ;;
        --latent-dim)
            LATENT_DIM="$2"
            shift 2
            ;;
        --num-encoder-blocks)
            NUM_ENCODER_BLOCKS="$2"
            shift 2
            ;;
        --num-decoder-blocks)
            NUM_DECODER_BLOCKS="$2"
            shift 2
            ;;
        --dilation)
            DILATION="$2"
            shift 2
            ;;
        --mask-weight)
            MASK_WEIGHT="$2"
            shift 2
            ;;
        --reconstruction-weight)
            RECONSTRUCTION_WEIGHT="$2"
            shift 2
            ;;
        --max-epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --gradient-clip)
            GRADIENT_CLIP="$2"
            shift 2
            ;;
        --early-stopping)
            EARLY_STOPPING=true
            shift
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --log-interval)
            LOG_INTERVAL="$2"
            shift 2
            ;;
        --progress-bar)
            PROGRESS_BAR=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$DATA_PATH" ]]; then
    echo "Error: --data-path parameter is required"
    echo "Use --help for usage information"
    exit 1
fi

# Validate data file exists
if [[ ! -f "$DATA_PATH" ]]; then
    echo "Error: Data file does not exist: $DATA_PATH"
    exit 1
fi

# Validate method parameter
if [[ "$METHOD" != "DMF" && "$METHOD" != "DCAE" ]]; then
    echo "Error: Method must be either DMF or DCAE"
    exit 1
fi


# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build Python command
PYTHON_CMD="python $SCRIPT_DIR/train.py"
PYTHON_CMD="$PYTHON_CMD --method $METHOD"
PYTHON_CMD="$PYTHON_CMD --data-path \"$DATA_PATH\""
PYTHON_CMD="$PYTHON_CMD --hidden-dims $HIDDEN_DIMS"
PYTHON_CMD="$PYTHON_CMD --embedding-dim $EMBEDDING_DIM"
PYTHON_CMD="$PYTHON_CMD --latent-dim $LATENT_DIM"
PYTHON_CMD="$PYTHON_CMD --num-encoder-blocks $NUM_ENCODER_BLOCKS"
PYTHON_CMD="$PYTHON_CMD --num-decoder-blocks $NUM_DECODER_BLOCKS"
PYTHON_CMD="$PYTHON_CMD --dilation $DILATION"
PYTHON_CMD="$PYTHON_CMD --mask-weight $MASK_WEIGHT"
PYTHON_CMD="$PYTHON_CMD --reconstruction-weight $RECONSTRUCTION_WEIGHT"
PYTHON_CMD="$PYTHON_CMD --max-epochs $MAX_EPOCHS"
PYTHON_CMD="$PYTHON_CMD --batch-size $BATCH_SIZE"
PYTHON_CMD="$PYTHON_CMD --learning-rate $LEARNING_RATE"
PYTHON_CMD="$PYTHON_CMD --weight-decay $WEIGHT_DECAY"
PYTHON_CMD="$PYTHON_CMD --gradient-clip $GRADIENT_CLIP"
PYTHON_CMD="$PYTHON_CMD --patience $PATIENCE"
PYTHON_CMD="$PYTHON_CMD --results-dir \"$RESULTS_DIR\""
PYTHON_CMD="$PYTHON_CMD --log-interval $LOG_INTERVAL"
PYTHON_CMD="$PYTHON_CMD --device $DEVICE"

if [[ -n "$GPU_ID" ]]; then
    PYTHON_CMD="$PYTHON_CMD --gpu-id $GPU_ID"
fi

if [[ "$EARLY_STOPPING" == "true" ]]; then
    PYTHON_CMD="$PYTHON_CMD --early-stopping"
fi

if [[ "$PROGRESS_BAR" == "true" ]]; then
    PYTHON_CMD="$PYTHON_CMD --progress-bar"
fi

# Display training configuration
echo "================================"
echo "PRISM Training Configuration"
echo "================================"
echo "Method: $METHOD"
echo "Data Path: $DATA_PATH"
echo "Hidden Dimensions: $HIDDEN_DIMS"
if [[ "$METHOD" == "DMF" ]]; then
    echo "Embedding Dimension: $EMBEDDING_DIM"
else
    echo "Latent Dimension: $LATENT_DIM"
    echo "Encoder Blocks: $NUM_ENCODER_BLOCKS"
    echo "Decoder Blocks: $NUM_DECODER_BLOCKS"
    echo "Dilation Factor: $DILATION"
fi
echo "Mask Weight: $MASK_WEIGHT"
echo "Reconstruction Weight: $RECONSTRUCTION_WEIGHT"
echo "Max Epochs: $MAX_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Weight Decay: $WEIGHT_DECAY"
echo "Gradient Clip: $GRADIENT_CLIP"
echo "Early Stopping: $EARLY_STOPPING"
if [[ "$EARLY_STOPPING" == "true" ]]; then
    echo "Patience: $PATIENCE"
fi
echo "Results Directory: $RESULTS_DIR"
echo "Show Progress Bar: $PROGRESS_BAR"
echo "Device: $DEVICE"
if [[ -n "$GPU_ID" ]]; then
    echo "GPU ID: $GPU_ID"
fi
echo "================================"
echo

# Confirm execution
read -p "Start training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled"
    exit 0
fi

# Execute training
echo "Starting training..."
echo "Executing command: $PYTHON_CMD"
echo

eval $PYTHON_CMD

# Check execution result
if [[ $? -eq 0 ]]; then
    echo
    echo "================================"
    echo "Training completed successfully!"
    echo "Results saved in: $RESULTS_DIR"
    echo "================================"
else
    echo
    echo "================================"
    echo "Training failed!"
    echo "================================"
    exit 1
fi