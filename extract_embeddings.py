#!/usr/bin/env python3
"""
Script to extract embeddings from multiple layers of vision models.
Efficiently processes images in batches and extracts both final outputs 
and intermediate layer representations (CLS tokens).

Leverages the UnifiedPreprocessor from model_wrapper for automatic
handling of various image formats including grayscale/RGB conversion,
different array layouts, and value normalization.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
from tqdm import tqdm
import argparse
from pathlib import Path

from src.model_wrapper import get_model, list_available_models


class LayerEmbeddingExtractor:
    """Extracts embeddings from specified layers of vision models.
    
    Supports two pooling methods:
    - 'cls': Extract CLS token (first token) from transformer outputs
    - 'avg': Average pool all tokens (including CLS token) from transformer outputs
    """
    
    def __init__(
        self, 
        model_name: str, 
        model_size: str, 
        layer_indices: Optional[List[int]] = None,
        all_layers: bool = False,
        device: str = 'auto',
        batch_size: int = 32,
        pooling_method: str = 'cls'
    ):
        """
        Initialize the embedding extractor.
        
        Args:
            model_name: Model type ('dinov2', 'dinov3', 'clip', 'vit', 'convnext')
            model_size: Model size ('base', 'large', etc.)
            layer_indices: List of layer indices to extract embeddings from
            all_layers: If True, extract from all available layers (ignores layer_indices)
            device: Device to use ('cuda', 'cpu', or 'auto')
            batch_size: Batch size for processing
            pooling_method: Method to pool tokens ('cls' for CLS token, 'avg' for average pooling)
        """
        self.model_name = model_name.lower()
        self.model_size = model_size
        self.all_layers = all_layers
        self.batch_size = batch_size
        self.pooling_method = pooling_method.lower()
        
        if self.pooling_method not in ['cls', 'avg']:
            raise ValueError(f"pooling_method must be 'cls' or 'avg', got '{pooling_method}'")
        
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Loading {model_name}-{model_size} on {self.device}...")
        self.model, self.preprocessor = get_model(model_name, model_size, self.device)
        
        # Determine layer indices
        if all_layers:
            layers = self._get_model_layers()
            if layers is None:
                raise ValueError(f"Layer extraction not supported for {self.model_name}")
            self.layer_indices = list(range(len(layers)))
            print(f"Extracting from all {len(self.layer_indices)} layers")
        else:
            self.layer_indices = layer_indices or []
            print(f"Extracting from specified layers: {self.layer_indices}")
        
        # Storage for intermediate embeddings
        self.layer_embeddings = {}
        self.hook_handles = []
        
        # Register hooks for specified layers (only if we have layers to extract from)
        if self.layer_indices:
            self._register_hooks()
        else:
            print("No intermediate layers to extract - will only extract final output")
        
    def _register_hooks(self):
        """Register forward hooks to capture intermediate layer outputs."""
        def create_hook(layer_idx: int):
            def hook_fn(module, input, output):
                # Extract embeddings from transformer outputs
                if hasattr(output, 'shape') and len(output.shape) == 3:
                    # Shape: (batch_size, seq_len, hidden_dim)
                    if self.pooling_method == 'cls':
                        # Extract CLS token (first token)
                        pooled_embedding = output[:, 0, :].detach().cpu()
                    elif self.pooling_method == 'avg':
                        # Average pool all tokens (excluding CLS token)
                        # pooled_embedding = output.mean(dim=1).detach().cpu()
                        pooled_embedding = output[:, 1:, :].mean(dim=1).detach().cpu()
                elif isinstance(output, tuple) and len(output) > 0:
                    # Handle cases where output is a tuple
                    if self.pooling_method == 'cls':
                        pooled_embedding = output[0][:, 0, :].detach().cpu()
                    elif self.pooling_method == 'avg':
                        # pooled_embedding = output[0].mean(dim=1).detach().cpu()
                        pooled_embedding = output[0][:, 1:, :].mean(dim=1).detach().cpu()
                else:
                    # Fallback: use the output as-is
                    pooled_embedding = output.detach().cpu()
                
                if layer_idx not in self.layer_embeddings:
                    self.layer_embeddings[layer_idx] = []
                self.layer_embeddings[layer_idx].append(pooled_embedding)
            return hook_fn
        
        # Get the appropriate layer structure based on model type
        layers = self._get_model_layers()
        
        if layers is None:
            raise ValueError(f"Layer extraction not supported for {self.model_name}")
        
        for layer_idx in self.layer_indices:
            if layer_idx >= len(layers):
                print(f"Warning: Layer {layer_idx} does not exist. Model has {len(layers)} layers.")
                continue
                
            hook_handle = layers[layer_idx].register_forward_hook(create_hook(layer_idx))
            self.hook_handles.append(hook_handle)
            
        print(f"Registered hooks for layers: {self.layer_indices}")
    
    def _get_model_layers(self):
        """Get the layer structure based on model type."""
        if self.model_name == 'clip':
            return self.model.model.transformer.resblocks
        elif self.model_name in ['dinov2', 'vit', 'vitmae']:
            return self.model.model.encoder.layer
        elif self.model_name == 'dinov3':
            return self.model.model.layer
        elif self.model_name == 'convnext':
            # ConvNext doesn't have transformer layers in the same way
            # You might need to specify which layers to hook into
            print("Warning: ConvNext layer extraction needs specific implementation")
            return None
        else:
            print(f"Warning: Layer extraction not supported for {self.model_name}")
            return None
    
    def extract_embeddings(self, X: np.ndarray, y: Optional[np.ndarray] = None, original_shape: Optional[tuple] = None) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from all specified layers and final output.
        
        Args:
            X: Input images in any supported format - UnifiedPreprocessor handles conversion
            y: Optional labels to include in output
            
        Returns:
            Dictionary with keys:
            - 'output': Final model output (N, output_dim)
            - 'layer_{i}': Intermediate layer embeddings (N, layer_dim)
            - 'y': Labels (N,) if provided
        """
        print(f"Input shape: {X.shape}")
        if y is not None:
            print(f"Labels shape: {y.shape}")
        N = X.shape[0]
        
        # Clear previous embeddings
        self.layer_embeddings = {}
        final_outputs = []
        
        print(f"Processing {N} images in batches of {self.batch_size}...")
        
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, N, self.batch_size), desc="Extracting embeddings"):
                batch_end = min(i + self.batch_size, N)
                batch_images = X[i:batch_end]
                
                # UnifiedPreprocessor handles all format conversions automatically
                batch_tensors = self.preprocessor(batch_images).to(self.device)
                
                # Forward pass (this triggers the hooks)
                output = self.model(batch_tensors)
                final_outputs.append(output.detach().cpu())
        
        # Combine all outputs
        results = {
            'output': torch.cat(final_outputs, dim=0).numpy().reshape(*original_shape, -1)
        }
        
        # Include labels if provided
        if y is not None:
            results['y'] = y
        
        # Combine intermediate layer embeddings
        for layer_idx in self.layer_indices:
            if layer_idx in self.layer_embeddings:
                layer_embeds = torch.cat(self.layer_embeddings[layer_idx], dim=0).numpy()
                results[f'layer_{layer_idx}'] = layer_embeds.reshape(*original_shape, -1)
                print(f"Layer {layer_idx} embeddings: {layer_embeds.shape}")
        
        print(f"Final output shape: {results['output'].shape}")
        if y is not None:
            print(f"Labels shape: {results['y'].shape}")
        return results
    
    def cleanup(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


def main():
    # parser = argparse.ArgumentParser(description='Extract embeddings from vision models')
    # parser.add_argument('--data_path', type=str, default='data.npy', 
    #                    help='Path to input data (.npy file)')
    # parser.add_argument('--output_path', type=str, default='embeddings.npy',
    #                    help='Path to save embeddings (.npy file)')
    # parser.add_argument('--model_name', type=str, default='clip',
    #                    choices=list(list_available_models().keys()),
    #                    help='Model type to use')
    # parser.add_argument('--model_size', type=str, default='base',
    #                    help='Model size to use')
    # parser.add_argument('--layers', type=int, nargs='*', default=[0, 6, 11],
    #                    help='Layer indices to extract embeddings from (ignored if --all_layers is used)')
    # parser.add_argument('--all_layers', action='store_true',
    #                    help='Extract embeddings from all available layers')
    # parser.add_argument('--batch_size', type=int, default=32,
    #                    help='Batch size for processing')
    # parser.add_argument('--device', type=str, default='auto',
    #                    choices=['auto', 'cuda', 'cpu'],
    #                    help='Device to use')
    
    # args = parser.parse_args()

    dataset_path = '/scratch/agjma/Sebastian/Datasets/'
    output_path = '/scratch/agjma/Sebastian/Embedding_Avg_NoCLS/'
    #dataset_list = ["blocks_0.npz", "blocks_10.npz", "blocks_15.npz", "blocks_20.npz", "blocks_30.npz", "blocks_free.npz", "fruit_all_same_0.npz", "fruit_all_same_20.npz", "fruit_all_same_40.npz", "fruit_all_same_60.npz", "words_normal.npz", "words_pseudo.npz", "words_random.npz"]
    #dataset_list = ["blocks_0.npz", "blocks_10.npz", "blocks_15.npz", "blocks_20.npz", "blocks_30.npz", "blocks_free.npz", "fruit_all_same_0.npz", "fruit_all_same_20.npz", "fruit_all_same_40.npz", "fruit_all_same_60.npz", "words_normal.npz", "words_pseudo.npz", "words_random.npz"]
    dataset_list = ["blocks2_free.npz"]
    model_list = ['dinov2', 'dinov3', 'clip', 'vit', 'vitmae']
    #model_list = model_list[::-1]
    model_size_list = ['base', 'large', 'huge']

    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Configuration settings
    batch_size = 256
    device = 'auto'
    all_layers = True
    pooling_method = 'avg'  # 'cls' for CLS token, 'avg' for average pooling
    
    # Loop over all combinations
    total_combinations = len(dataset_list) * len(model_list) * len(model_size_list)
    current_combination = 0
    
    for dataset_name in dataset_list:
        # Load dataset
        data_file = Path(dataset_path) / dataset_name
        print(f"\nLoading dataset: {dataset_name}")
        print("=" * 60)
        
        try:
            data = np.load(data_file, allow_pickle=True)
            X = data['X']
            y = data['y'] if 'y' in data else None
            
            # Assert expected shape format: (20000, 2, H, W, 3) or (20000, 2, H, W, 1)
            assert len(X.shape) == 5, f"Expected 5D array, got shape {X.shape}"
            assert X.shape[0] == 20000, f"Expected 20000 samples, got {X.shape[0]}"
            assert X.shape[1] == 2, f"Expected 2 images per sample, got {X.shape[1]}"
            assert X.shape[4] in [1, 3], f"Expected 1 or 3 channels, got {X.shape[4]}"
            
            print(f"Dataset loaded and validated: X.shape={X.shape}")
            if y is not None:
                print(f"Labels loaded: y.shape={y.shape}")
            
            # Store original shape for later reconstruction
            original_shape = X.shape[:2]  # (20000, 2)
            
            # Reshape for processing: (40000, H, W, C)
            X = X.reshape(-1, X.shape[2], X.shape[3], X.shape[4])
                
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue
        
        for model_name in model_list:
            for model_size in model_size_list:
                current_combination += 1
                print(f"\n[{current_combination}/{total_combinations}] Processing: {dataset_name} + {model_name}-{model_size}")
                print("-" * 60)
                
                # Skip invalid model-size combinations
                available_models = list_available_models()
                if model_name not in available_models:
                    print(f"Model {model_name} not available. Skipping...")
                    continue
                if model_size not in available_models[model_name]:
                    print(f"Model size {model_size} not available for {model_name}. Skipping...")
                    continue
                
                # Define output filename
                dataset_base = dataset_name.replace('.npz', '')
                output_filename = f"{dataset_base}_{model_name}_{model_size}_embeddings.npy"
                output_file = Path(output_path) / output_filename
                
                # Skip if file already exists
                if output_file.exists():
                    print(f"Output file {output_filename} already exists. Skipping...")
                    continue
                
                # Initialize extractor
                try:
                    extractor = LayerEmbeddingExtractor(
                        model_name=model_name,
                        model_size=model_size,
                        layer_indices=None,
                        all_layers=all_layers,
                        device=device,
                        batch_size=batch_size,
                        pooling_method=pooling_method
                    )
                except Exception as e:
                    print(f"Error initializing {model_name}-{model_size}: {e}")
                    continue
                
                # Extract embeddings
                try:
                    embeddings = extractor.extract_embeddings(X, y, original_shape)
                    
                    # Save results
                    print(f"Saving embeddings to {output_filename}...")
                    np.save(output_file, embeddings)
                    
                    # Print summary
                    print(f"\nExtraction Summary for {dataset_base} + {model_name}-{model_size}:")
                    print("=" * 50)
                    for key, value in embeddings.items():
                        if key == 'y':
                            print(f"{'labels':12}: {value.shape} (labels)")
                        else:
                            print(f"{key:12}: {value.shape}")
                    print(f"Saved to: {output_file}")
                    
                except Exception as e:
                    print(f"Error during extraction for {model_name}-{model_size}: {e}")
                finally:
                    extractor.cleanup()
    
    print(f"\n\nProcessing complete! Processed {current_combination} combinations.")
    print(f"Check output directory: {output_path}")

if __name__ == "__main__":
    # Example usage if run without arguments
    main()