# LLaMa 3 From Scratch

This project implements a LLaMa 3 model from scratch, including dataset creation, model architecture, and training pipeline.

## Project Structure

- `Dataset_creator.py`: Script for creating and processing the dataset
- `model.py`: Implementation of the LLaMa model architecture
- `Tokenizers.py`: Tokenizer utilities for encoding and decoding text
- `training_arc.py`: Training pipeline for the LLaMa model

## Features

- Implementation of LLaMa 3 architecture
- Custom dataset creation and processing
- Efficient training pipeline with gradient accumulation
- Dynamic learning rate scheduling
- Validation and checkpointing
- Text generation capabilities

## Requirements

- Python 3.x
- PyTorch
- Transformers
- NumPy
- tqdm

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/Pragateeshwaran/LLaMa-3-From-Scratch.git
   cd LLaMa-3-From-Scratch/src
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Hugging Face token in `Tokenizers.py`

## Usage

### Dataset Creation

Run the dataset creation script:

```
python Dataset_creator.py
```

### Training

Start the training process:

```
python training_arc.py
```

The training script includes:
- Automatic mixed precision (AMP) for efficient training
- Gradient accumulation for large batch sizes
- Learning rate scheduling
- Regular validation and model checkpointing
- Text generation samples during training

## Model Architecture

The LLaMa model is implemented in `model.py` and includes:
- Rotary positional embeddings
- RMSNorm for layer normalization
- Flash Attention (when available)
- Configurable number of layers, heads, and model dimensions

## Acknowledgements

This implementation is inspired by the LLaMa paper and various open-source implementations. Special thanks to the authors of the referenced papers:

- [2104.09864v5.pdf](https://arxiv.org/abs/2104.09864v5)
- [2307.09288v2.pdf](https://arxiv.org/abs/2307.09288v2)
- [2407.21783v1.pdf](https://arxiv.org/abs/2407.21783v1)

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
 