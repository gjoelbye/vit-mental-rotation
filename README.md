# ViT Mental Rotation

This repository contains the core components for vision transformer experiments on mental rotation tasks.

## Repository Structure

```
vit-mental-rotation/
├── src/
│   ├── __init__.py
│   └── model_wrapper.py          # Unified model wrapper for various vision models
├── plots/
│   ├── plot_results_figure_3.py  # Plotting script for Figure 3 results
│   └── plot_results_figure_4.py  # Plotting script for Figure 4 results
├── data_generation/
│   ├── generate_shapes.py         # Shape generation utilities
│   └── text_generation.ipynb     # Text generation experiments
├── extract_embeddings.py         # Extract embeddings from vision models
├── experiment.py                  # Main experiment script
└── README.md                     # This file
```

## Core Components

### 1. Extract Embeddings (`extract_embeddings.py`)
- Extracts embeddings from multiple layers of vision models
- Supports various models (CLIP, DINOv2, ViT, ConvNeXT)
- Handles batch processing efficiently
- Supports both CLS token and average pooling methods

### 2. Experiment Script (`experiment.py`)
- Main siamese network experiment for mental rotation tasks
- Supports k-fold cross-validation and predefined train/test splits

### 3. Model Wrapper (`src/model_wrapper.py`)
- Unified interface for loading different vision models
- Automatic preprocessing and format handling
- Supports CLIP, DINOv2, ViT, and ConvNeXT architectures

### 4. Data Generation (`data_generation/generate_shapes.py`)
- Utilities for generating synthetic shapes for mental rotation tasks

### 5. Plotting Scripts (`plots/`)
- `plot_results_figure_3.py`: Visualization for Figure 3 results
- `plot_results_figure_4.py`: Visualization for Figure 4 results

## Usage

### Extract Embeddings
```python
from extract_embeddings import LayerEmbeddingExtractor

extractor = LayerEmbeddingExtractor(
    model_name='clip',
    model_size='base',
    all_layers=True,
    pooling_method='cls'
)

embeddings = extractor.extract_embeddings(X, y)
```

### Run Experiments
```bash
python experiment.py --embedding_file your_embeddings.npy
```

### Generate Plots
```bash
python plots/plot_results_figure_3.py
python plots/plot_results_figure_4.py
```

## Dependencies

The code requires the following main dependencies:
- PyTorch
- PyTorch Lightning
- NumPy
- Scikit-learn
- Transformers (for CLIP and other models)
- Matplotlib/Seaborn (for plotting)

## Notes

This repository contains vision transformer experiments for mental rotation tasks. The code supports various model architectures and provides tools for embedding extraction, training, and result visualization.
