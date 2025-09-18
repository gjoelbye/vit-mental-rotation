import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from collections import defaultdict

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.use14corefonts'] = False

# Set font family to Times-like serif fonts (using available system fonts)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Nimbus Roman', 'Times New Roman', 'Times', 'serif']
mpl.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern for math text


# Set working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from Notebooks/
os.chdir(project_root)

# Configuration constants
RESULTS_DIRECTORIES = ["results_avg_1", "results_avg_2", "results_avg_3"]

# Figure size constants - adjusted for 2x2 layout
FIGURE_WIDTH = 3.25  # Width in inches
FIGURE_HEIGHT_RATIO = 7.5/10  # Height as ratio of width (6/10 = 0.6) - taller for 2 rows
FIGURE_HEIGHT = FIGURE_WIDTH * FIGURE_HEIGHT_RATIO  # Calculated height
FIGURE_DPI = 600  # DPI for saving the plot

# Font size constants
TITLE_FONT_SIZE = 8
AXIS_LABEL_FONT_SIZE = 7
TICK_LABEL_FONT_SIZE = 7
LEGEND_FONT_SIZE = 8

# Axis label positioning constants
XLABEL_PAD = 2  # Distance between x-axis and x-label (default is usually ~6)
YLABEL_PAD = 2  # Distance between y-axis and y-label (default is usually ~6)

# Plot styling constants
LINE_WIDTH = 1.0
MARKER_SIZE = 1.5
LEGEND_MARKER_SIZE = 4
LINE_ALPHA = 0.8  # Transparency of plot lines (0.0 = transparent, 1.0 = opaque)
USE_LINE_STYLES = False  # Set to False to use solid lines for all datasets

# Dataset filtering configuration - now handled per row
# Row 0 (blocks): datasets starting with 'blocks'
# Row 1 (fruit): datasets starting with 'fruit'
BLOCKS_DATASETS = ['blocks2_0', 'blocks2_10', 'blocks_20', 'blocks_30', 'blocks2_free']
FRUIT_DATASETS = ['fruit_all_same_0', 'fruit_all_same_20', 'fruit_all_same_40', 'fruit_all_same_60']

# For backward compatibility, keep INCLUDE_DATASETS but it won't be used
INCLUDE_DATASETS = BLOCKS_DATASETS + FRUIT_DATASETS


# Dataset legend name mapping
DATASET_LEGEND_NAMES = {
    'blocks2_0': 'Shepard-Metzler ±0°',
    'blocks2_10': 'Shepard-Metzler ±10°',
    'blocks_5': 'Shepard-Metzler ±5°',
    'blocks_15': 'Shepard-Metzler ±15°',
    'blocks_20': 'Shepard-Metzler ±20°',
    'blocks_30': 'Shepard-Metzler ±30°',
    'blocks2_free': 'Shepard-Metzler Free',
    'fruit_all_same_0': 'Photo-Realistic 90°',
    'fruit_all_same_20': 'Photo-Realistic 70°',
    'fruit_all_same_40': 'Photo-Realistic 50°',
    'fruit_all_same_60': 'Photo-Realistic 30°',
    'words_normal': 'Text Normal',
    'words_random': 'Text Random', 
    'words_pseudo': 'Text Pseudo',
}

# Model title name mapping
MODEL_TITLE_NAMES = {
    'vit_base': 'ViT Base',
    'vit_large': 'ViT Large', 
    'vit_huge': 'ViT Huge',
    'dinov2_base': 'DINOv2 Base',
    'dinov2_large': 'DINOv2 Large',
    'dinov2_huge': 'DINOv2 Huge',
    'dinov3_base': 'DINOv3 Base',
    'dinov3_large': 'DINOv3 Large',
    'dinov3_huge': 'DINOv3 Huge',
    'clip_base': 'CLIP Base',
    'clip_large': 'CLIP Large',
    'clip_huge': 'CLIP Huge',
    'vitmae_base': 'ViTMAE Base',
    'vitmae_large': 'ViTMAE Large',
    'vitmae_huge': 'ViTMAE Huge'
}

MODEL_ORDER = [['dinov2_huge', 'dinov3_huge']]

# Dataset groups configuration - similar datasets get same color with different shades
# Each group gets a different base color and symbol
# Individual colors in INDIVIDUAL_DATASET_COLORS will override group colors for specific datasets
DATASET_GROUPS = {
    'words': ['words_normal', 'words_random', 'words_pseudo'],
    'fruit_all_same': ['fruit_all_same_0', 'fruit_all_same_20', 'fruit_all_same_40', 'fruit_all_same_60'],
    'blocks': ['blocks2_0', 'blocks_5', 'blocks2_10', 'blocks_15', 'blocks_20', 'blocks_30', 'blocks2_free']
}

DATASET_COLORS = {}

# Individual dataset color overrides - specify exact colors for specific datasets
INDIVIDUAL_DATASET_COLORS = {
    'words_normal': '#2E6B59',
    'words_random': '#458A74',
    'words_pseudo': '#7FBEA9',
    'blocks2_0': '#8D502D',
    'blocks2_10': '#A8633E',
    'blocks_20': '#C07A54',
    'blocks_30': '#CD8F67',
    'blocks2_free': '#D89B6F',
    'fruit_all_same_0': '#32639B',
    'fruit_all_same_20': '#4F7CB0',
    'fruit_all_same_40': '#729ECE',
    'fruit_all_same_60': '#96BCE3',
}

# Available symbols and line styles for different dataset groups
SYMBOLS = ['s', 'o', '^', 'v', 'D', 'P', '*', 'X', 'p', 'h']
LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5)), (0, (1, 1))]

def load_and_group_results():
    """Load results files from multiple directories and group them by model type and dataset."""
    model_groups = defaultdict(list)
    all_results_files = []
    
    # Collect files from all specified directories
    for results_dir in RESULTS_DIRECTORIES:
        if not os.path.exists(results_dir):
            print(f"Warning: {results_dir} directory not found")
            continue
        
        try:
            dir_files = os.listdir(results_dir)
        except PermissionError:
            print(f"Error: Permission denied accessing {results_dir} directory")
            continue
        
        # Add full paths for files ending with _siamese_results.npy
        dir_results_files = [f"{results_dir}/{file}" for file in dir_files 
                           if file.endswith("_siamese_results.npy")]
        all_results_files.extend(dir_results_files)
        print(f"Found {len(dir_results_files)} result files in {results_dir}")
    
    if not all_results_files:
        print("Warning: No result files found in any directory")
        return {}
    
    print(f"Total result files found: {len(all_results_files)}")
    
    # Flatten MODEL_ORDER to get all valid model keys
    all_models = [model for sublist in MODEL_ORDER for model in sublist]
    
    for results_file in all_results_files:
        filename = os.path.basename(results_file).replace("_siamese_results.npy", "")
        parts = filename.split("_")
        
        if len(parts) >= 3:
            dataset = '_'.join(parts[:-2])
            model = parts[-2]
            size = parts[-1]
            model_key = f"{model}_{size}"
            
            # Only include models in our predefined order
            if model_key in all_models:
                model_groups[model_key].append((results_file, dataset))
    
    # Sort model_groups according to flattened MODEL_ORDER
    return {k: model_groups[k] for k in all_models if k in model_groups}

def process_model_results(files_and_datasets):
    """Process results for a single model and return dataset-grouped results.
    
    Combines results from the same model/dataset across multiple directories
    by averaging the accuracy values and standard deviations.
    """
    dataset_groups = defaultdict(list)
    
    for results_file, dataset in files_and_datasets:
        dataset_groups[dataset].append(results_file)
    
    processed_results = {}
    
    for dataset, results_files in dataset_groups.items():
        if dataset in BLOCKS_DATASETS or dataset in FRUIT_DATASETS:
            pass
        else:
            continue

        all_raw_data = []  # Store raw test_acc data for proper std calculation
        all_val_acc = []
        n_hidden = None
        successful_files = []
        
        for results_file in results_files:
            try:
                results = np.load(results_file, allow_pickle=True).item()
                
                if n_hidden is None:
                    n_hidden = len(results) - 1
                
                # Extract mean accuracy per layer
                val_acc = [np.mean(results[f"layer_{i}"]["summary"]["test_acc"])
                          for i in range(n_hidden)]
                
                # Add output layer
                val_acc.append(np.mean(results["output"]["summary"]["test_acc"]))
                
                # Store raw data for proper std calculation
                raw_data = []
                for i in range(n_hidden):
                    raw_data.append(results[f"layer_{i}"]["summary"]["test_acc"])
                raw_data.append(results["output"]["summary"]["test_acc"])
                

                all_val_acc.append(val_acc)
                all_raw_data.append(raw_data)
                successful_files.append(results_file)
                
            except (FileNotFoundError, ValueError, KeyError) as e:
                print(f"Warning: Could not load {results_file}: {e}")
                continue
            except Exception as e:
                print(f"Error: Unexpected error loading {results_file}: {e}")
                continue
        
        if all_val_acc:  # Only add if we have valid results
            all_val_acc = np.array(all_val_acc)
            
            # Calculate combined standard deviation properly
            # Combine all raw data across files for each layer
            combined_std = []
            if not all_raw_data:
                print(f"Warning: No raw data available for dataset {dataset}")
                continue
            n_layers = len(all_raw_data[0])
            
            for layer_idx in range(n_layers):
                # Combine raw data from all files for this layer
                layer_data = []
                for file_data in all_raw_data:
                    layer_data.extend(file_data[layer_idx])
                combined_std.append(np.std(layer_data))
            
            processed_results[dataset] = {
                'mean_acc': np.mean(all_val_acc, axis=0),
                'mean_std': np.array(combined_std),
                'n_layers': len(all_val_acc[0]),
                'n_files_combined': len(successful_files)
            }
    
    return processed_results

def should_include_dataset(dataset_name, row_idx=None):
    """Check if a dataset should be included based on row-specific filtering."""
    if row_idx is None:
        # Backward compatibility - include all configured datasets
        return dataset_name in (BLOCKS_DATASETS + FRUIT_DATASETS)
    elif row_idx == 0:
        # First row: only blocks datasets
        return dataset_name in BLOCKS_DATASETS
    elif row_idx == 1:
        # Second row: only fruit datasets
        return dataset_name in FRUIT_DATASETS
    else:
        return False

def assign_dataset_colors_and_symbols(all_datasets):
    """Assign colors and symbols to datasets based on DATASET_COLORS and DATASET_GROUPS configuration."""
    
    dataset_info = {}  # Will store {dataset: {'color': color, 'symbol': symbol}}
    
    # Get fallback colors for ungrouped datasets
    cmap = plt.get_cmap("tab10")
    
    group_idx = 0
    ungrouped_idx = 0
    
    # First, process all groups to assign symbols, then handle individual color overrides
    # This ensures datasets with individual colors still get group-based symbols
    for group_name, group_datasets in DATASET_GROUPS.items():
        # Get base color from DATASET_COLORS or fallback to colormap
        if group_name in DATASET_COLORS:
            base_color = mcolors.to_rgba(DATASET_COLORS[group_name])
        else:
            base_color = cmap(group_idx % 10)
        
        symbol = SYMBOLS[group_idx % len(SYMBOLS)]
        line_style = LINE_STYLES[group_idx % len(LINE_STYLES)] if USE_LINE_STYLES else '-'
        
        # Find all datasets in this group that exist in our data
        group_datasets_found = [ds for ds in group_datasets if ds in all_datasets]
        
        # Separate datasets with individual colors from those using group colors
        group_color_datasets = [ds for ds in group_datasets_found if ds not in INDIVIDUAL_DATASET_COLORS]
        individual_color_datasets = [ds for ds in group_datasets_found if ds in INDIVIDUAL_DATASET_COLORS]
        
        # First, assign individual colors with group symbol
        for dataset in individual_color_datasets:
            color = mcolors.to_rgba(INDIVIDUAL_DATASET_COLORS[dataset])
            dataset_info[dataset] = {
                'color': color,
                'symbol': symbol,  # Use group symbol
                'linestyle': line_style  # Use group line style
            }
        
        # Then, create shades for datasets using group colors
        n_variants = len(group_color_datasets)
        
        if n_variants > 0:
            # Create shades from lighter to darker
            for i, dataset in enumerate(group_color_datasets):
                # Create shade by adjusting the brightness
                # Lighter shades for earlier datasets, darker for later ones
                shade_factor = 0.4 + 0.6 * (i / max(1, n_variants - 1))  # Range from 0.4 to 1.0
                
                # Convert to HSV, adjust value (brightness), convert back
                hsv_color = mcolors.rgb_to_hsv(base_color[:3])
                hsv_color[2] = hsv_color[2] * shade_factor
                shaded_color = mcolors.hsv_to_rgb(hsv_color)
                
                dataset_info[dataset] = {
                    'color': (*shaded_color, base_color[3] if len(base_color) > 3 else 1.0),
                    'symbol': symbol,
                    'linestyle': line_style
                }
            
        group_idx += 1
    
    # Handle datasets not in any group
    grouped_datasets = {ds for group_datasets in DATASET_GROUPS.values() for ds in group_datasets}
    ungrouped_datasets = [ds for ds in all_datasets if ds not in grouped_datasets and ds not in dataset_info]
    
    for dataset in ungrouped_datasets:
        # Check if this dataset has an individual color override
        if dataset in INDIVIDUAL_DATASET_COLORS:
            color = mcolors.to_rgba(INDIVIDUAL_DATASET_COLORS[dataset])
            # Use a unique symbol and line style for ungrouped individual overrides
            symbol_idx = (group_idx + ungrouped_idx) % len(SYMBOLS)
            line_style = LINE_STYLES[(group_idx + ungrouped_idx) % len(LINE_STYLES)] if USE_LINE_STYLES else '-'
        else:
            # Check if dataset matches any key in DATASET_COLORS
            matched_color = None
            matched_key = None
            for color_key, color_value in DATASET_COLORS.items():
                if color_key in dataset:
                    matched_color = mcolors.to_rgba(color_value)
                    matched_key = color_key
                    break
            
            if matched_color:
                # Use the specified color from DATASET_COLORS
                color = matched_color
                # Use symbol and line style based on the matched key
                symbol_idx = list(DATASET_COLORS.keys()).index(matched_key)
                line_style = LINE_STYLES[symbol_idx % len(LINE_STYLES)] if USE_LINE_STYLES else '-'
            else:
                # Use fallback color from colormap
                color_idx = (group_idx + ungrouped_idx) % 10
                symbol_idx = (group_idx + ungrouped_idx) % len(SYMBOLS)
                line_style = LINE_STYLES[(group_idx + ungrouped_idx) % len(LINE_STYLES)] if USE_LINE_STYLES else '-'
                color = cmap(color_idx)
                
        dataset_info[dataset] = {
            'color': color,
            'symbol': SYMBOLS[symbol_idx % len(SYMBOLS)],
            'linestyle': line_style
        }
        ungrouped_idx += 1
    
    return dataset_info

def setup_empty_subplot(ax, model_name, row_idx, col_idx, n_rows, n_cols):
    """Set up a subplot for models with no data available."""
    ax.text(0.5, 0.5, 'No Data\nAvailable', transform=ax.transAxes,
           ha='center', va='center', fontsize=14, alpha=0.5)
    
    # Add title inside the plot in upper left corner
    ax.text(0.06, 0.98, model_name, transform=ax.transAxes,
           ha='left', va='top', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    
    # Remove top and right spines (borders), keep left and bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylim(0.475, 1.025)
    
    # Add custom grid lines - both x and y axis custom
    # Custom y-axis grid lines (exclude the topmost one at y=1.0)
    ytick_positions = [0.5, 0.6, 0.7, 0.8, 0.9]  # Explicit positions to avoid sharey issues
    for ytick in ytick_positions:
        ax.axhline(y=ytick, color='gray', alpha=0.3, linestyle='--', linewidth=0.4, zorder=0)
    
    # Custom x-axis grid lines that stop at y=0.95
    major_xtick_positions = [0, 25, 50, 75, 100]  # Major x-ticks
    minor_xtick_positions = [12.5, 37.5, 62.5, 87.5]  # Minor x-ticks at midpoints
    
    for xtick in major_xtick_positions:
        ax.axvline(x=xtick, ymin=(0.475-0.475)/(1.025-0.475), ymax=(0.95-0.475)/(1.025-0.475), 
                  color='gray', alpha=0.3, linestyle='--', linewidth=0.4, zorder=0)
    
    for xtick in minor_xtick_positions:
        ax.axvline(x=xtick, ymin=(0.475-0.475)/(1.025-0.475), ymax=(0.95-0.475)/(1.025-0.475), 
                  color='gray', alpha=0.3, linestyle=':', linewidth=0.3, zorder=0)
    
    # Set standardized x-axis with percentage labels (same as data plots)
    ax.set_xlim(-5, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], rotation=0, ha="center", fontsize=TICK_LABEL_FONT_SIZE)
    # Add minor ticks
    ax.set_xticks([12.5, 37.5, 62.5, 87.5], minor=True)
    
    # Move tick marks inside the plot
    ax.tick_params(axis='both', direction='in', labelsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis='both', which='minor', direction='in')
    
    # Add labels only to appropriate subplots
    if row_idx == n_rows - 1:  # Bottom row
        ax.set_xlabel("Layer Position", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=XLABEL_PAD)
    if col_idx == 0:  # Left column
        ax.set_ylabel("Test Accuracy", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=YLABEL_PAD)

if __name__ == "__main__":
    # Load and group all results
    model_groups = load_and_group_results()
    
    # Note: We don't exit if no model_groups found, as we still want to show empty plot cells
    
    # Validate MODEL_ORDER structure
    if not MODEL_ORDER or not all(isinstance(col, list) for col in MODEL_ORDER):
        raise ValueError("MODEL_ORDER must be a non-empty list of lists")
    
    # Create 2x2 subplot grid: 2 rows (blocks/fruit) x 2 cols (dinov2_huge/dinov3_huge)
    n_cols = 2  # DINOv2 HUGE and DINOv3 HUGE
    n_rows = 2  # Blocks and Fruit
    
    # Build global color and symbol assignment for all datasets (filtered)
    all_datasets = sorted({ds for _, files in model_groups.items() for _, ds in files})
    # Apply dataset filtering to color assignment - include all datasets we want to show
    filtered_datasets = [ds for ds in all_datasets if should_include_dataset(ds)]
    
    dataset_info = assign_dataset_colors_and_symbols(filtered_datasets)
    
    # Track which datasets are actually plotted (after filtering)
    plotted_datasets = set()
    

    # Create subplot grid with dynamic dimensions - share x-axis within columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), sharex='col', sharey=True)
    
    
    # Handle case where n_cols = 1 (axes won't be 2D)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Define the models for our 2x2 grid
    models = ['dinov2_huge', 'dinov3_huge']
    
    # Iterate through the 2x2 grid
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            model_key = models[col_idx]  # Column determines the model
            # Use model title name from dictionary, fallback to formatted version
            model_name = MODEL_TITLE_NAMES.get(model_key, model_key.replace("_", " ").title())
            
            # Check if this model has any results
            if model_key not in model_groups:
                setup_empty_subplot(ax, model_name, row_idx, col_idx, n_rows, n_cols)
                continue
            
            # Process results for this model
            processed_results = process_model_results(model_groups[model_key])
            
            if not processed_results:
                setup_empty_subplot(ax, model_name, row_idx, col_idx, n_rows, n_cols)
                continue
            
            # Plot each dataset for this model
            for dataset, results in processed_results.items():
                # Apply row-specific dataset filtering
                if not should_include_dataset(dataset, row_idx):
                    continue

                mean_acc = results['mean_acc']
                mean_std = results['mean_std']
                n_layers = results['n_layers']
                
                # Map ALL layers (including output) to 0-100% range with even spacing
                x_positions = []
                for i in range(n_layers):
                    # Map all layers evenly across 0-100 range
                    if n_layers > 1:
                        percentage = (i / (n_layers - 1)) * 100  # 0 to 100
                    else:
                        percentage = 0  # Single layer goes at 0
                    x_positions.append(percentage)
                
                # Check if dataset is in dataset_info to prevent KeyError
                if dataset not in dataset_info:
                    print(f"Warning: Dataset '{dataset}' not found in dataset_info, skipping")
                    continue
                    
                color = dataset_info[dataset]['color']
                symbol = dataset_info[dataset]['symbol']
                line_style = dataset_info[dataset]['linestyle']
                
                standard_error = mean_std / np.sqrt(2)
                ax.fill_between(x_positions, mean_acc - standard_error, mean_acc + standard_error,
                               alpha=0.25, color=color)
                ax.plot(x_positions, mean_acc, marker=symbol, linestyle=line_style, linewidth=LINE_WIDTH,
                        color=color, markersize=MARKER_SIZE, alpha=LINE_ALPHA)
                #ax.plot(x_positions, mean_acc, linewidth=1, color=color, markersize=2)
                
                # Track that this dataset was actually plotted
                plotted_datasets.add(dataset)
            
            # Customize subplot
            # Add title inside the plot in upper left corner
            ax.text(0.06, 0.98, model_name, transform=ax.transAxes,
                   ha='left', va='top', fontsize=TITLE_FONT_SIZE, fontweight='bold')
            
            # Remove top and right spines (borders), keep left and bottom
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.set_ylim(0.475, 1.025)
            
            # Add custom grid lines - both x and y axis custom
            # Custom y-axis grid lines (exclude the topmost one at y=1.0)
            ytick_positions = [0.5, 0.6, 0.7, 0.8, 0.9]  # Explicit positions to avoid sharey issues
            for ytick in ytick_positions:
                ax.axhline(y=ytick, color='gray', alpha=0.3, linestyle='--', linewidth=0.4, zorder=0)
            
            # Custom x-axis grid lines that stop at y=0.95
            major_xtick_positions = [0, 25, 50, 75, 100]  # Major x-ticks
            minor_xtick_positions = [12.5, 37.5, 62.5, 87.5]  # Minor x-ticks at midpoints
            
            for xtick in major_xtick_positions:
                ax.axvline(x=xtick, ymin=(0.475-0.475)/(1.025-0.475), ymax=(0.95-0.475)/(1.025-0.475), 
                          color='gray', alpha=0.3, linestyle='--', linewidth=0.4, zorder=0)
            
            for xtick in minor_xtick_positions:
                ax.axvline(x=xtick, ymin=(0.475-0.475)/(1.025-0.475), ymax=(0.95-0.475)/(1.025-0.475), 
                          color='gray', alpha=0.3, linestyle=':', linewidth=0.3, zorder=0)
            
            # Set standardized x-axis with percentage labels
            ax.set_xlim(-5, 105)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], rotation=0, ha="center", fontsize=TICK_LABEL_FONT_SIZE)
            # Add minor ticks
            ax.set_xticks([12.5, 37.5, 62.5, 87.5], minor=True)
            
            # Move tick marks inside the plot
            ax.tick_params(axis='both', direction='in', labelsize=TICK_LABEL_FONT_SIZE)
            ax.tick_params(axis='both', which='minor', direction='in')
            
            # Add labels only to appropriate subplots
            if row_idx == n_rows - 1:  # Bottom row
                ax.set_xlabel("Layer Position", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=XLABEL_PAD)
            if col_idx == 0:  # Left column
                ax.set_ylabel("Test Accuracy", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=YLABEL_PAD)
    
    # Create grouped legend (only for datasets that were actually plotted)
    if plotted_datasets:
        # Create custom grouped legend
        from matplotlib.patches import Rectangle
        from matplotlib.text import Text
        
        # Separate blocks and fruit datasets that were plotted
        plotted_blocks = [ds for ds in plotted_datasets if ds.startswith('blocks')]
        plotted_fruit = [ds for ds in plotted_datasets if ds.startswith('fruit')]
        
        # Sort them by their percentage values
        def extract_percentage(dataset_name):
            if 'free' in dataset_name:
                return 999  # Put 'free' at the end
            # Extract number from dataset name
            import re
            numbers = re.findall(r'\d+', dataset_name)
            return int(numbers[-1]) if numbers else 0
        
        plotted_blocks = sorted(plotted_blocks, key=extract_percentage)
        plotted_fruit = sorted(plotted_fruit, key=extract_percentage)
        
        # Create the legend manually using text and line elements
        fig_width, fig_height = fig.get_size_inches()
        
        # Position for the legend (below the plots) - moved further down with more space
        legend_y = 0.145
        legend_x_start = 0.0725
        
        # Calculate dimensions for the entire legend box
        legend_box_width = 0.8  # Reduced width to fit legend entries better
        legend_box_height = 0.1  # Increased height to properly encompass both legend lines
        legend_box_x = legend_x_start - 0.00525  # Small padding on left
        legend_box_y = legend_y - 0.075 # Adjusted position to center both lines
        
        # Add single white rectangle with light gray border and rounded corners behind the entire legend
        from matplotlib.patches import FancyBboxPatch
        legend_rect = FancyBboxPatch((legend_box_x, legend_box_y), 
                                    legend_box_width, legend_box_height,
                                    boxstyle="round,pad=0.01",
                                    facecolor='white', edgecolor='lightgray', linewidth=1.0,
                                    transform=fig.transFigure, zorder=1)
        fig.patches.append(legend_rect)
        
        # Shepard-Metzler line
        shepard_text = fig.text(legend_x_start, legend_y, 'Shepard-Metzler:', 
                               fontsize=LEGEND_FONT_SIZE, fontweight='normal', 
                               ha='left', va='center', zorder=2)
        
        # Add Shepard-Metzler symbols and angle labels
        x_offset = legend_x_start + 0.25  # Reduced space after label
        for ds in plotted_blocks:
            # Get angle label
            if 'free' in ds:
                angle_label = 'Free'
            else:
                import re
                numbers = re.findall(r'\d+', ds)
                angle_label = f"±{numbers[-1]}°" if numbers else "±0°"
            
            # Create marker
            marker_line = plt.Line2D([0], [0], color=dataset_info[ds]['color'], 
                                   marker=dataset_info[ds]['symbol'], linestyle='None', 
                                   markersize=LEGEND_MARKER_SIZE)
            
            # Add marker to figure - larger axes to prevent clipping
            marker_ax = fig.add_axes([x_offset, legend_y - 0.02, 0.04, 0.04])
            marker_ax.add_line(marker_line)
            marker_ax.set_xlim(-0.5, 0.5)
            marker_ax.set_ylim(-0.5, 0.5)
            marker_ax.axis('off')
            marker_ax.set_zorder(3)  # Set higher z-order to appear on top of legend box
            
            # Add angle text with more space from marker
            fig.text(x_offset + 0.035, legend_y, angle_label, 
                    fontsize=LEGEND_FONT_SIZE, ha='left', va='center', zorder=2)
            
            x_offset += 0.11  # More space between entries
        
        # Photo-Realistic line - much more vertical separation to prevent clipping
        photo_y = legend_y - 0.05
        photo_text = fig.text(legend_x_start, photo_y, 'Photo-Realistic:', 
                             fontsize=LEGEND_FONT_SIZE, fontweight='normal', 
                             ha='left', va='center', zorder=2)
        
        # Add Photo-Realistic symbols and viewing angles
        x_offset = legend_x_start + 0.25  # Reduced space after label
        for ds in plotted_fruit:
            # Get viewing angle label - convert from rotation angle to viewing angle
            import re
            numbers = re.findall(r'\d+', ds)
            if numbers:
                rotation_angle = int(numbers[-1])
                viewing_angle = 90 - rotation_angle  # Convert rotation to viewing angle
                angle_label = f"{viewing_angle}°"
            else:
                angle_label = "90°"
            
            # Create marker
            marker_line = plt.Line2D([0], [0], color=dataset_info[ds]['color'], 
                                   marker=dataset_info[ds]['symbol'], linestyle='None', 
                                   markersize=LEGEND_MARKER_SIZE)
            
            # Add marker to figure - larger axes to prevent clipping
            marker_ax = fig.add_axes([x_offset, photo_y - 0.02, 0.04, 0.04])
            marker_ax.add_line(marker_line)
            marker_ax.set_xlim(-0.5, 0.5)
            marker_ax.set_ylim(-0.5, 0.5)
            marker_ax.axis('off')
            marker_ax.set_zorder(3)  # Set higher z-order to appear on top of legend box
            
            # Add angle text with more space from marker
            fig.text(x_offset + 0.035, photo_y, angle_label, 
                    fontsize=LEGEND_FONT_SIZE, ha='left', va='center', zorder=2)
            
            x_offset += 0.11  # More space between entries
    
    # Adjust subplot positions manually to avoid tight_layout warnings with custom legend axes
    # Much more bottom space for the expanded legend with proper symbol spacing
    plt.subplots_adjust(left=0.0, bottom=0.3, right=1.0, top=1.0, wspace=0.05, hspace=0.1)
    # Save plot with high quality as PDF
    plt.savefig("figure_4.pdf", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.show()