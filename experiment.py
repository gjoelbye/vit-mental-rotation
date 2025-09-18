import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile
import shutil
import os
import logging
import warnings
import random
import argparse

# Suppress warnings and logs
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#torch.set_float32_matmul_precision('medium')

def set_seed(seed: int = 42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For PyTorch Lightning
    pl.seed_everything(seed, workers=True)

class EmbeddingPairDataset(Dataset):
    """Dataset for embedding pairs with optional GPU storage."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, device="cpu"):
        """Create the dataset.

        Args:
            embeddings: np.ndarray with shape (N, 2, embedding_dim)
            labels: np.ndarray with shape (N,)
            device: Target device for the returned tensors.
        """

        # Resolve the target device (fall back to CPU if CUDA not available)
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")  # gracefully degrade

        # Convert numpy arrays → torch tensors directly on the target device
        try:
            self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
            self.labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
        except Exception as e:
            print("Available CUDA devices:", torch.cuda.device_count())
            print("CUDA is available:", torch.cuda.is_available())
            print("torch.cuda.current_device():", torch.cuda.current_device() if torch.cuda.is_available() else "N/A")
            print("torch.cuda.get_device_name():", torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "N/A")
            print("Exception during tensor creation:", e)
            raise

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class SiameseNet(pl.LightningModule):
    """Siamese network using PyTorch Lightning."""
    
    def __init__(self, embedding_dim: int, learning_rate: float = 1e-3, dropout: float = 0.2,
                 warmup_epochs: int = 5, max_epochs: int = 100, debug_training: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.debug_training = debug_training

        # Shared processing tower for both embeddings
        self.shared_tower = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.BatchNorm1d(embedding_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2), nn.BatchNorm1d(embedding_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 128)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Track metrics for training curves
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
    def forward(self, x1, x2):
        # Process both embeddings through shared tower
        e1 = F.normalize(self.shared_tower(x1), p=2, dim=-1)
        e2 = F.normalize(self.shared_tower(x2), p=2, dim=-1)
        
        # Compute absolute difference
        diff = torch.abs(e1 - e2)
        
        # Classify
        logit = self.classifier(diff).squeeze(-1)
        return torch.sigmoid(logit)
    
    def _shared_step(self, batch, stage):
        """Shared step for training, validation, and test."""
        embeddings, labels = batch
        x1, x2 = embeddings[:, 0], embeddings[:, 1]
        
        y_hat = self(x1, x2)
        loss = F.binary_cross_entropy(y_hat, labels)
        
        # Calculate accuracy
        preds = (y_hat > 0.5).float()
        acc = (preds == labels).float().mean()
        
        return loss, acc
    
    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, 'train')
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, 'val')
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, 'test')
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        return {'loss': loss, 'acc': acc}
    
    def on_train_epoch_end(self):
        # Store metrics for training curves
        train_loss = self.trainer.logged_metrics.get('train_loss')
        train_acc = self.trainer.logged_metrics.get('train_acc')
        if train_loss is not None and train_acc is not None:
            self.train_losses.append(train_loss.item())
            self.train_accs.append(train_acc.item())
            
    def on_validation_epoch_end(self):
        # Store metrics for training curves
        val_loss = self.trainer.logged_metrics.get('val_loss')
        val_acc = self.trainer.logged_metrics.get('val_acc')
        if val_loss is not None and val_acc is not None:
            self.val_losses.append(val_loss.item())
            self.val_accs.append(val_acc.item())
            
    def get_training_history(self):
        """Return training history."""
        return {
            'train_loss': self.train_losses,
            'train_acc': self.train_accs,
            'val_loss': self.val_losses,
            'val_acc': self.val_accs,
            'epochs': list(range(len(self.train_losses)))
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.warmup_epochs
        )
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.max_epochs - self.warmup_epochs)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[self.warmup_epochs]
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


def run_baseline_mlp(embeddings: np.ndarray, labels: np.ndarray,
                     train_val_idx: np.ndarray, test_idx: np.ndarray,
                     config: Dict, k_folds: int) -> None:
    """Run baseline sklearn MLP as a sanity check."""
    try:
        # Prepare data for sklearn (concatenate pairs)
        train_X = np.concatenate([embeddings[train_val_idx][:, 0], embeddings[train_val_idx][:, 1]], axis=1)
        test_X = np.concatenate([embeddings[test_idx][:, 0], embeddings[test_idx][:, 1]], axis=1)
        
        train_y = labels[train_val_idx]
        test_y = labels[test_idx]
        
        # Train sklearn MLP
        validation_fraction = 1.0 - config['train_val_split']
        mlp = MLPClassifier(max_iter=100, random_state=config['seed'], early_stopping=True, 
                          validation_fraction=validation_fraction, n_iter_no_change=25)
        mlp.fit(train_X, train_y)
        
        # Get predictions and metrics
        train_pred = mlp.predict_proba(train_X)[:, 1]
        test_pred = mlp.predict_proba(test_X)[:, 1]
        
        train_acc_mlp = accuracy_score(train_y, (train_pred > 0.5).astype(int))
        test_acc_mlp = accuracy_score(test_y, (test_pred > 0.5).astype(int))
        
        train_loss_mlp = log_loss(train_y, train_pred)
        test_loss_mlp = log_loss(test_y, test_pred)
        
        # Calculate width to match fold format with padded numbers
        fold_num_width = len(str(k_folds))
        fold_format_width = len(f"Fold {k_folds:>{fold_num_width}}/{k_folds}:")
        baseline_label = f"{'Baseline':<{fold_format_width}}"
        
        print(f"{baseline_label}: Train (L: {train_loss_mlp:.2f}, A: {train_acc_mlp:.2f}), "
              f"Val (L:  N/A, A:  N/A), "
              f"Test (L: {test_loss_mlp:.2f}, A: {test_acc_mlp:.2f})")
        
    except Exception as e:
        print(f"Baseline MLP failed: {e}")


def train_fold(embeddings: np.ndarray, labels: np.ndarray, 
               train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
               embedding_dim: int, fold: int, layer_name: str,
               config: Dict) -> Dict:
    """Train a single fold of the siamese network with optional retraining."""
    
    # ------------------------------------------------------------------
    # Standardize embeddings using statistics from the *training* split
    # ------------------------------------------------------------------
    train_mean = embeddings[train_idx].mean(axis=(0, 1), keepdims=True)
    train_std = embeddings[train_idx].std(axis=(0, 1), keepdims=True) + 1e-8

    embeddings_norm = (embeddings - train_mean) / train_std

    # ------------------------------------------------------------------
    # Data preparation – optionally keep the full dataset on the GPU to
    # bypass CPU→GPU transfers (set CONFIG['gpu_dataset'] = True).
    # ------------------------------------------------------------------

    keep_dataset_on_gpu = config.get('gpu_dataset', False) and torch.cuda.is_available()
    target_device = "cuda" if keep_dataset_on_gpu else "cpu"
    
    train_dataset = EmbeddingPairDataset(embeddings_norm[train_idx], labels[train_idx], device=target_device)
    val_dataset = EmbeddingPairDataset(embeddings_norm[val_idx], labels[val_idx], device=target_device)
    test_dataset = EmbeddingPairDataset(embeddings_norm[test_idx], labels[test_idx], device=target_device)

    if keep_dataset_on_gpu:
        num_workers = 0
        pin_memory = False
    else:
        num_workers = min(4, os.cpu_count() or 1)  # Use available CPUs, max 4
        pin_memory = True

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                             shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    # ------------------------------------------------------------------
    # Multiple retrains per fold - keep the best one based on val accuracy
    # ------------------------------------------------------------------
    
    retrains_per_fold = config.get('retrains_per_fold', 1)
    best_val_acc = -1.0
    best_results = None
    best_retrain = 0
    
    for retrain in range(retrains_per_fold):
        # Set a different seed for each retrain to get different initialization
        retrain_seed = config['seed'] + fold * 1000 + retrain
        set_seed(retrain_seed)
        
        # Initialize model with the retrain-specific seed
        model = SiameseNet(
            embedding_dim=embedding_dim,
            learning_rate=config['learning_rate'],
            dropout=config['dropout'],
            warmup_epochs=config['warmup_epochs'],
            max_epochs=config['max_epochs'],
            debug_training=config.get('debug_training', False) # Pass debug_training
        )
        
        # Create temporary directory for checkpoints
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Callbacks
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=config['patience'],
                mode='min',
                verbose=False
            )
            
            checkpoint = ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                verbose=False,
                dirpath=temp_dir,
                filename='best_model'
            )
            
            # Trainer configuration - enable progress bar and logging if debugging
            debug_mode = config.get('debug_training', False)
            trainer = pl.Trainer(
                max_epochs=config['max_epochs'],
                callbacks=[early_stop, checkpoint],
                accelerator='auto',
                devices=1,
                logger=False,  # Keep logging disabled for clean output
                enable_progress_bar=debug_mode,  # Enable progress bar only in debug mode
                enable_model_summary=debug_mode,  # Enable model preview in debug mode
                deterministic=True,
                num_sanity_val_steps=0,  # Skip sanity check
                log_every_n_steps=50 if debug_mode else 999999,  # More frequent logging in debug mode
            )
            
            # Train
            trainer.fit(model, train_loader, val_loader)
            
            # Test with best model
            best_model_path = checkpoint.best_model_path
            if best_model_path is None:
                # If no checkpoint was saved (e.g., training stopped immediately), use current model
                best_model = model
            else:
                best_model = SiameseNet.load_from_checkpoint(
                    best_model_path, 
                    embedding_dim=embedding_dim,
                    learning_rate=config['learning_rate'],
                    dropout=config['dropout'],
                    debug_training=config.get('debug_training', False)
                )
            
            # Evaluate on all splits with best model
            train_results = trainer.test(best_model, train_loader, verbose=False)[0]
            val_results = trainer.test(best_model, val_loader, verbose=False)[0]
            test_results = trainer.test(best_model, test_loader, verbose=False)[0]
            
            # Get training history from the trained model
            history = model.get_training_history()
            
            # Find the epoch with minimum validation loss
            if history['val_loss'] and len(history['val_loss']) > 0:
                best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1  # +1 because epochs are 1-indexed
            else:
                # Fallback to current epoch if no validation history available
                best_epoch = max(1, trainer.current_epoch)
            
            # Prepare results for this retrain
            retrain_results = {
                'train_loss': train_results['test_loss'],
                'train_acc': train_results['test_acc'],
                'val_loss': val_results['test_loss'],
                'val_acc': val_results['test_acc'],
                'test_loss': test_results['test_loss'],
                'test_acc': test_results['test_acc'],
                'best_val_loss': checkpoint.best_model_score.item() if checkpoint.best_model_score is not None else None,
                'best_epoch': best_epoch,
                'training_history': history,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'test_size': len(test_idx),
                'retrain_number': retrain,
                'retrain_seed': retrain_seed
            }
            
            # Check if this retrain has the best validation accuracy so far
            current_val_acc = retrain_results['val_acc']
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                best_results = retrain_results.copy()
                best_retrain = retrain
                
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    # Add information about which retrain was selected
    if best_results is not None:
        best_results['selected_retrain'] = best_retrain
        best_results['total_retrains'] = retrains_per_fold
        best_results['best_val_acc_across_retrains'] = best_val_acc
        
    return best_results


def run_experiment_with_splits(embeddings: np.ndarray, labels: np.ndarray,
                              splits_list: List[Tuple[np.ndarray, np.ndarray]],
                              layer_name: str, config: Dict, 
                              experiment_type: str = "fold") -> Dict:
    """Run experiment with provided train/test splits.
    
    Args:
        embeddings: Embedding pairs
        labels: Labels
        splits_list: List of (train_val_indices, test_indices) tuples
        layer_name: Name of the layer
        config: Configuration dictionary
        experiment_type: "fold" for K-fold CV, "split" for predefined splits
    """
    
    print(f"\n--- {layer_name} --- (Shape: {embeddings.shape})")
    
    embedding_dim = embeddings.shape[-1]
    k_experiments = len(splits_list)
    
    # Initialize results storage
    layer_results = {}
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    test_losses = []
    test_accs = []
    
    for exp_idx, (train_val_idx, test_idx) in enumerate(splits_list):
        # Further split train_val into train and validation
        train_split = config['train_val_split']
        
        # For predefined splits, shuffle to ensure random train/val split
        # For K-fold, the indices are already from sklearn's split
        if experiment_type == "split":
            np.random.seed(config['seed'] + exp_idx)  # Ensure reproducible but different splits
            shuffled_train_val = np.random.permutation(train_val_idx)
            train_size = int(train_split * len(train_val_idx))
            train_idx = shuffled_train_val[:train_size]
            val_idx = shuffled_train_val[train_size:]
        else:
            np.random.seed(config['seed'] + exp_idx)  # Ensure reproducible splits
            
            # Use stratified sampling to maintain class balance in train/val split
            train_indices_relative, val_indices_relative = train_test_split(
                range(len(train_val_idx)),
                test_size=1.0 - train_split,
                random_state=config['seed'] + exp_idx,
                stratify=labels[train_val_idx]
            )
            
            # Convert relative indices back to absolute indices
            train_idx = train_val_idx[train_indices_relative]
            val_idx = train_val_idx[val_indices_relative]
        
        # Train experiment
        exp_results = train_fold(
            embeddings, labels,
            train_idx, val_idx, test_idx,
            embedding_dim, exp_idx, layer_name,
            config
        )
        
        # Store results for this experiment
        layer_results[f'{experiment_type}_{exp_idx}'] = exp_results
        
        train_losses.append(exp_results['train_loss'])
        train_accs.append(exp_results['train_acc'])
        val_losses.append(exp_results['val_loss'])
        val_accs.append(exp_results['val_acc'])
        test_losses.append(exp_results['test_loss'])
        test_accs.append(exp_results['test_acc'])
        
        # Calculate width needed for experiment numbers
        exp_num_width = len(str(k_experiments))
        exp_label = f"{experiment_type.capitalize()} {exp_idx+1:>{exp_num_width}}/{k_experiments}:"
        
        # Prepare retrain information if multiple retrains were used
        retrain_info = ""
        if config.get('retrains_per_fold', 1) > 1:
            retrain_info = f" [Retrain {exp_results['selected_retrain']+1}/{exp_results['total_retrains']}]"
        
        print(f"{exp_label} Train (L: {exp_results['train_loss']:.2f}, A: {exp_results['train_acc']:.2f}), "
              f"Val (L: {exp_results['val_loss']:.2f}, A: {exp_results['val_acc']:.2f}), "
              f"Test (L: {exp_results['test_loss']:.2f}, A: {exp_results['test_acc']:.2f}) "
              f"[Epoch: {exp_results['best_epoch']}]{retrain_info}")
    
    # Add summary statistics
    summary_key = 'k_folds' if experiment_type == 'fold' else 'k_splits'
    layer_results['summary'] = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'test_loss': test_losses,
        'test_acc': test_accs,
        'embedding_dim': embedding_dim,
        summary_key: k_experiments
    }
    
    summary = layer_results['summary']
    print(f"Summary: Train (L: {np.mean(summary['train_loss']):.2f}±{np.std(summary['train_loss']):.2f}, "
          f"A: {np.mean(summary['train_acc']):.2f}±{np.std(summary['train_acc']):.2f}), "
          f"Val (L: {np.mean(summary['val_loss']):.2f}±{np.std(summary['val_loss']):.2f}, "
          f"A: {np.mean(summary['val_acc']):.2f}±{np.std(summary['val_acc']):.2f}), "
          f"Test (L: {np.mean(summary['test_loss']):.2f}±{np.std(summary['test_loss']):.2f}, "
          f"A: {np.mean(summary['test_acc']):.2f}±{np.std(summary['test_acc']):.2f})")
    
    return layer_results


def run_kfold_experiment(embeddings: np.ndarray, labels: np.ndarray, 
                        layer_name: str, config: Dict) -> Dict:
    """Run K-fold cross validation experiment."""
    
    # K-fold cross validation
    k_folds = config['k_folds']
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config['seed'])
    
    # Prepare splits for the shared function
    splits_list = [(train_val_idx, test_idx) for train_val_idx, test_idx in skf.split(embeddings, labels)]
    
    return run_experiment_with_splits(embeddings, labels, splits_list, layer_name, config, "fold")


def run_predefined_splits_experiment(embeddings: np.ndarray, labels: np.ndarray,
                                   train_indices_list: List[np.ndarray], 
                                   test_indices_list: List[np.ndarray],
                                   layer_name: str, config: Dict) -> Dict:
    """Run experiment with predefined train/test splits."""
    
    # Prepare splits for the shared function
    splits_list = list(zip(train_indices_list, test_indices_list))
    
    return run_experiment_with_splits(embeddings, labels, splits_list, layer_name, config, "split")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run siamese network experiments on embeddings')
    parser.add_argument('--embedding_file', type=str, required=True,
                       help='Path to embedding file (.npy)')
    parser.add_argument('--splits_file', type=str, default=None,
                       help='Path to .npz file containing train/test indices (for predefined splits)')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug training output (epoch-by-epoch monitoring)')
    args = parser.parse_args()

    EMBEDDING_FOLDER = '/scratch/agjma/Sebastian/Embedding_Avg_NoCLS'
    
    # Validate input file
    embedding_file = args.embedding_file
    embedding_file = os.path.join(EMBEDDING_FOLDER, embedding_file)
    
    # Determine experiment mode based on whether splits_file is provided
    use_predefined_splits = args.splits_file is not None
    splits_file = None
    
    print(embedding_file)
    if not embedding_file.endswith('_embeddings.npy'):
        print(f"Error: Embedding file must end with '_embeddings.npy'")
        return
    
    if not os.path.exists(embedding_file):
        print(f"Error: Embedding file '{embedding_file}' does not exist")
        return
    
    base_name = os.path.basename(embedding_file).replace('_embeddings.npy', '')
    
    if use_predefined_splits:
        # splits_file = os.path.join(EMBEDDING_FOLDER, args.splits_file)
        splits_file = args.splits_file
        if not splits_file.endswith('.npz'):
            print(f"Error: Splits file must be .npz format")
            return
        if not os.path.exists(splits_file):
            print(f"Error: Splits file '{splits_file}' does not exist")
            return
    
    # Generate output path
    if use_predefined_splits:
        output_file = f'results_avg_1/{base_name}_siamese_results_predefined.npy'
    else:
        output_file = f'results_avg_1/{base_name}_siamese_results.npy'
    
    if os.path.exists(output_file) and not args.no_save:
        print(f"Error: Output file '{output_file}' already exists")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Training configuration
    CONFIG = {
        'k_folds': 10,
        'train_val_split': 0.9,  # 90% train, 10% validation within training set
        'max_epochs': 100, #100
        'batch_size': 256,
        'learning_rate': 1e-3,
        'dropout': 0.0,
        'patience': 30, #15,  # Early stopping patience
        'seed': 44,  # Random seed for reproducibility
        'warmup_epochs': 15, #5,
        'gpu_dataset': True,  # Set to True to keep entire dataset in GPU memory
        'retrains_per_fold': 1,  # Number of times to retrain each fold, keeping the best result
        'debug_training': args.debug # Set to True to enable debug output
    }
    
    # Set seed for reproducibility
    set_seed(CONFIG['seed'])
    
    # Load embeddings and splits
    print(f"Loading embeddings from {embedding_file}")
    
    if use_predefined_splits:
        print(f"Using predefined train/test splits from {splits_file}")
        print(f"Configuration: Predefined splits, {CONFIG['retrains_per_fold']} retrain(s) per split, "
              f"max_epochs={CONFIG['max_epochs']}, patience={CONFIG['patience']}")
    else:
        print(f"Configuration: {CONFIG['k_folds']}-fold CV, {CONFIG['retrains_per_fold']} retrain(s) per fold, "
              f"max_epochs={CONFIG['max_epochs']}, patience={CONFIG['patience']}")
    
    if CONFIG['debug_training']:
        print(f"Debug mode enabled: Will show epoch-by-epoch training progress")
    
    if CONFIG['retrains_per_fold'] > 1:
        print(f"Note: Using {CONFIG['retrains_per_fold']} retrains per fold - selecting best model based on validation accuracy")
    
    try:
        # Load embeddings from .npy file
        embedding_data = np.load(embedding_file, allow_pickle=True).item()
        labels = embedding_data['y']
        
        if use_predefined_splits:
            # Load splits from separate .npz file
            splits_data = np.load(splits_file, allow_pickle=True)
            train_indices_list = []
            test_indices_list = []
            
            for key in splits_data.files:
                if key.startswith('train_indices_'):
                    split_idx = int(key.replace('train_indices_', ''))
                    if split_idx >= len(train_indices_list):
                        train_indices_list.extend([None] * (split_idx + 1 - len(train_indices_list)))
                    train_indices_list[split_idx] = splits_data[key]
                elif key.startswith('test_indices_'):
                    split_idx = int(key.replace('test_indices_', ''))
                    if split_idx >= len(test_indices_list):
                        test_indices_list.extend([None] * (split_idx + 1 - len(test_indices_list)))
                    test_indices_list[split_idx] = splits_data[key]
            
            # Remove None entries and ensure lists are same length
            train_indices_list = [x for x in train_indices_list if x is not None]
            test_indices_list = [x for x in test_indices_list if x is not None]
            
            if len(train_indices_list) != len(test_indices_list):
                print(f"Error: Mismatch between number of train splits ({len(train_indices_list)}) and test splits ({len(test_indices_list)})")
                return
            
            if not train_indices_list:
                print("Error: No train/test indices found in splits file")
                return
                
    except Exception as e:
        print(f"Error loading files: {e}")
        return
    
    # Validate labels exist
    if 'y' not in embedding_data:
        print("Error: 'y' key not found in embedding file")
        return
    
    # Results storage
    all_results = {}
    
    # Check if there are any embedding layers to process
    embedding_keys = [key for key in embedding_data.keys() if key != 'y']
    if not embedding_keys:
        print("Error: No embedding layers found in file")
        return
    
    # Process each layer
    for key in embedding_keys:
        embeddings = embedding_data[key]
        
        # Validate embedding shape
        if len(embeddings.shape) != 3 or embeddings.shape[1] != 2:
            print(f"Warning: Skipping layer {key} - invalid shape {embeddings.shape}, expected (N, 2, embedding_dim)")
            continue
        
        # Validate embedding count matches labels
        if embeddings.shape[0] != len(labels):
            print(f"Warning: Skipping layer {key} - embedding count {embeddings.shape[0]} doesn't match label count {len(labels)}")
            continue
        
        # Run experiment for this layer
        if use_predefined_splits:
            layer_results = run_predefined_splits_experiment(
                embeddings, labels, train_indices_list, test_indices_list, key, CONFIG
            )
        else:
            layer_results = run_kfold_experiment(
                embeddings, labels, key, CONFIG
            )
        
        all_results[key] = layer_results
    
    # Check if any layers were successfully processed
    if not all_results:
        print("Error: No valid layers were processed")
        return
    
    # Save results
    print(f"\nSaving results to {output_file}")
    if not args.no_save:
        np.save(output_file, all_results)

if __name__ == "__main__":
    main()
    