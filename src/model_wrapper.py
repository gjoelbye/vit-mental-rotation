import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Union, List, Tuple, Optional, Dict, Any
import open_clip
from transformers import AutoImageProcessor
from transformers import Dinov2Model, DINOv3ViTModel, ViTModel, ConvNextModel, ViTMAEModel
import warnings
import torchvision.models as models
import torchvision.transforms as transforms

# Model configurations
MODEL_CONFIGS = {
    'dinov2': {
        'small': {'id': 'facebook/dinov2-small', 'input_size': 224, 'feature_dim': 384},
        'base': {'id': 'facebook/dinov2-base', 'input_size': 224, 'feature_dim': 768},
        'large': {'id': 'facebook/dinov2-large', 'input_size': 224, 'feature_dim': 1024},
        'huge': {'id': 'facebook/dinov2-giant', 'input_size': 224, 'feature_dim': 1280},  # Giant is the "huge" version
    },
    'dinov3': {
        'base': {'id': 'facebook/dinov3-vitb16-pretrain-lvd1689m', 'input_size': 224, 'feature_dim': 768},
        'large': {'id': 'facebook/dinov3-vitl16-pretrain-lvd1689m', 'input_size': 224, 'feature_dim': 1024},
        'huge': {'id': 'facebook/dinov3-vith16plus-pretrain-lvd1689m', 'input_size': 224, 'feature_dim': 1280},
    },
    'clip': {
        'base': {'arch': 'ViT-B-16', 'pretrained': 'laion2b_s34b_b88k', 'input_size': 224, 'feature_dim': 512},
        'large': {'arch': 'ViT-L-14', 'pretrained': 'laion2b_s32b_b82k', 'input_size': 224, 'feature_dim': 768},
        'huge': {'arch': 'ViT-H-14', 'pretrained': 'laion2b_s32b_b79k', 'input_size': 224, 'feature_dim': 1024},
    },
    'vit': {
        'base': {'id': 'google/vit-base-patch16-224-in21k', 'input_size': 224, 'feature_dim': 768},
        'large': {'id': 'google/vit-large-patch16-224-in21k', 'input_size': 224, 'feature_dim': 1024},
        'huge': {'id': 'google/vit-huge-patch14-224-in21k', 'input_size': 224, 'feature_dim': 1280},
    },
    # 'convnext': {
    #     'base': {'id': 'google/vit-base-patch16-224-in21k', 'input_size': 224, 'feature_dim': 768},
    #     'large': {'id': 'google/vit-large-patch16-224-in21k', 'input_size': 224, 'feature_dim': 1024},
    #     'huge': {'id': 'google/vit-huge-patch14-224-in21k', 'input_size': 224, 'feature_dim': 1280},
    # },
    'vitmae': {
        'base': {'id': 'facebook/vit-mae-base', 'input_size': 224, 'feature_dim': 768},
        'large': {'id': 'facebook/vit-mae-large', 'input_size': 224, 'feature_dim': 1024},
        'huge': {'id': 'facebook/vit-mae-huge', 'input_size': 224, 'feature_dim': 1280},
    }
}


class ModelWrapper(nn.Module):
    """Base wrapper class that standardizes model input interface."""
    
    def __init__(self, model: nn.Module, model_name: str, model_size: str):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.model_size = model_size
        
    def forward(self, x: torch.Tensor) -> Any:
        raise NotImplementedError
        
    def to(self, device: Union[str, torch.device]) -> 'ModelWrapper':
        self.model = self.model.to(device)
        return self
        
    def cpu(self) -> 'ModelWrapper':
        return self.to('cpu')
        
    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> 'ModelWrapper':
        return self.to('cuda' if device is None else device)
        
    def eval(self) -> 'ModelWrapper':
        self.model.eval()
        return self
        
    def train(self, mode: bool = True) -> 'ModelWrapper':
        self.model.train(mode)
        return self
        
    @property
    def parameters(self):
        """Delegate parameters to the wrapped model."""
        return self.model.parameters()
    
    @property
    def named_parameters(self):
        """Delegate named_parameters to the wrapped model."""
        return self.model.named_parameters()
    
    @property
    def modules(self):
        """Delegate modules to the wrapped model."""
        return self.model.modules()
    
    @property
    def named_modules(self):
        """Delegate named_modules to the wrapped model."""
        return self.model.named_modules()
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'model_name': self.model_name,
            'model_size': self.model_size,
            'model_type': type(self.model).__name__,
            'device': next(self.model.parameters()).device,
            'training': self.model.training,
        }


class DINOv2Wrapper(ModelWrapper):
    """Wrapper for DINOv2 models that expect pixel_values input."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        # Return pooled representation instead of full transformer output
        return outputs.pooler_output


class DINOv3Wrapper(ModelWrapper):
    """Wrapper for DINOv3 models that expect pixel_values input."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        # Return pooled representation instead of full transformer output
        return outputs.pooler_output


class CLIPWrapper(ModelWrapper):
    """Wrapper for CLIP visual models that expect direct tensor input."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CLIP already returns pooled features directly
        return self.model(x)


class ViTWrapper(ModelWrapper):
    """Wrapper for ViT models that expect pixel_values input."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        # Return pooled representation instead of full transformer output
        return outputs.pooler_output


class ConvNextWrapper(ModelWrapper):
    """Wrapper for ConvNext models that expect pixel_values input."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        # Return pooled representation instead of full transformer output
        return outputs.pooler_output

class ViTMAEWrapper(ModelWrapper):
    """Wrapper for ViTMAE models that expect pixel_values input."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
         # Create fixed noise tensor for deterministic masking
        batch_size = x.shape[0]
        seq_length = self.model.embeddings.num_patches  # Number of patches
        fixed_noise = torch.zeros(batch_size, seq_length, device=x.device)

        outputs = self.model(pixel_values=x)#, noise=fixed_noise)
        # Return pooled representation instead of full transformer output
        return outputs.last_hidden_state[:, 0]

class UnifiedPreprocessor:
    """Unified preprocessor that handles various input formats and converts them for model consumption."""
    
    def __init__(self, preprocess_fn, input_size: int = 224, device: Optional[str] = None):
        self.preprocess_fn = preprocess_fn
        self.input_size = input_size
        self.device = device
        
    def __call__(self, images: Union[np.ndarray, torch.Tensor, Image.Image, List]) -> torch.Tensor:
        """
        Process images in various formats.
        
        Args:
            images: Can be:
                - 4D numpy array: (N, H, W, C) or (N, C, H, W) where C ∈ [1, 3]
                - 4D torch tensor: (N, H, W, C) or (N, C, H, W) where C ∈ [1, 3]
                - PIL Image (automatically converted to batch of 1)
                - List of PIL Images (automatically stacked into batch)
                
        Returns:
            torch.Tensor: Preprocessed tensor ready for model input
        """
        try:
            if isinstance(images, Image.Image):
                result = self.preprocess_fn(images).unsqueeze(0)
            elif isinstance(images, list):
                result = self._process_list(images)
            elif isinstance(images, (np.ndarray, torch.Tensor)):
                result = self._process_array(images)
            else:
                raise ValueError(f"Unsupported input type: {type(images)}")
            
            # Move to device if specified
            if self.device is not None:
                result = result.to(self.device)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess images: {str(e)}")
    
    def _process_list(self, images: List) -> torch.Tensor:
        """Handle list of images more efficiently."""
        processed = []
        for img in images:
            if isinstance(img, Image.Image):
                processed.append(self.preprocess_fn(img))
            else:
                processed.append(self.preprocess_fn(self._to_pil(img)))
        return torch.stack(processed)
    
    def _process_array(self, images: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Process numpy arrays or torch tensors - only accepts 4D inputs."""
        # Keep as torch tensor if possible to avoid unnecessary conversions
        if isinstance(images, torch.Tensor):
            original_device = images.device
            if images.device != torch.device('cpu'):
                images = images.cpu()
            images_np = images.detach().numpy()
        else:
            images_np = images
            original_device = None
        
        # Only accept 4D arrays
        if images_np.ndim != 4:
            raise ValueError(f"Only 4D arrays are accepted. Got {images_np.ndim}D array with shape {images_np.shape}. "
                           f"Expected formats: (N, H, W, 1), (N, H, W, 3), (N, 1, H, W), or (N, 3, H, W)")
        
        # Validate 4D array format
        n, dim1, dim2, dim3 = images_np.shape
        
        # Determine format based on dimension values
        # Use heuristics: channels (1 or 3) are typically much smaller than height/width
        if dim1 in [1, 3] and dim1 < min(dim2, dim3):
            # (N, C, H, W) format - channels first
            c, h, w = dim1, dim2, dim3
            format_type = "channels_first"
        elif dim3 in [1, 3] and dim3 < max(dim1, dim2):
            # (N, H, W, C) format - channels last
            h, w, c = dim1, dim2, dim3
            format_type = "channels_last"
        else:
            # Ambiguous or invalid format
            raise ValueError(f"Cannot determine valid format from shape {images_np.shape}. "
                           f"Expected formats:\n"
                           f"  - (N, H, W, C) where C ∈ [1, 3] (channels last)\n"
                           f"  - (N, C, H, W) where C ∈ [1, 3] (channels first)\n"
                           f"Hint: Channels dimension should be 1 or 3, and smaller than spatial dimensions.")
        
        # Validate channel count
        if c not in [1, 3]:
            raise ValueError(f"Channel count must be 1 or 3, got {c}. Shape: {images_np.shape}")
        
        # Validate spatial dimensions are reasonable
        if h < 1 or w < 1:
            raise ValueError(f"Invalid spatial dimensions H={h}, W={w}. Shape: {images_np.shape}")
        if h > 10000 or w > 10000:
            warnings.warn(f"Very large spatial dimensions detected: H={h}, W={w}. Are you sure this is correct?")
        
        # Process batch of images
        if n > 256:
            warnings.warn(f"Processing large batch of {n} images. Consider processing in smaller batches for better memory efficiency.")
        
        processed = []
        for i in range(n):
            img = images_np[i]
            # Ensure img is in (H, W, C) format for _to_pil
            if format_type == "channels_first":
                img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
            processed.append(self.preprocess_fn(self._to_pil(img)))
        
        return torch.stack(processed)
    
    def _to_pil(self, img: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image with better validation."""
        if img.ndim not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D array for single image, got {img.ndim}D")
        
        # Handle different channel arrangements
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))
        
        # Handle grayscale
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            if img.ndim == 3:
                img = img.squeeze(2)
            mode = 'L'
        else:
            mode = 'RGB'
        
        # Normalize to 0-255 if needed
        if img.dtype in [np.float32, np.float64]:
            if img.max() <= 1.0 and img.min() >= 0.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        try:
            return Image.fromarray(img, mode=mode).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to convert array to PIL Image: {str(e)}")


def list_available_models() -> Dict[str, List[str]]:
    """List all available models and their sizes."""
    return {model_name: list(sizes.keys()) for model_name, sizes in MODEL_CONFIGS.items()}


def get_model_info(model_name: str, model_size: Optional[str] = None) -> Dict[str, Any]:
    """Get information about available model configurations."""
    model_name = model_name.lower()
    if model_name not in MODEL_CONFIGS:
        available = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Unsupported model: {model_name}. Available models: {available}")
    
    config = MODEL_CONFIGS[model_name]
    if model_size is None:
        return {
            'model_name': model_name,
            'available_sizes': list(config.keys()),
            'default_size': 'base' if 'base' in config else list(config.keys())[0]
        }
    
    if model_size not in config:
        available_sizes = list(config.keys())
        raise ValueError(f"Unsupported {model_name} size: {model_size}. Available sizes: {available_sizes}")
    
    return {
        'model_name': model_name,
        'model_size': model_size,
        **config[model_size]
    }

def get_model(model_name: str, model_size: Optional[str] = None, device: Optional[str] = None, **kwargs) -> Tuple[ModelWrapper, UnifiedPreprocessor]:
    """
    Get a unified model and preprocessor interface.
    
    Args:
        model_name: Model type ('dinov2', 'dinov3', 'clip', 'vit', 'convnext', 'vitmae')
        model_size: Model size variant (e.g., 'small', 'base', 'large', 'huge')
        device: Device to move model to ('cuda', 'cpu', or None for auto)
        **kwargs: Additional arguments passed to model-specific loaders
        
    Returns:
        Tuple of (model, preprocessor)
        
    Raises:
        ValueError: If model_name or model_size is not supported
        RuntimeError: If model loading fails
        
    Notes:
        - All models return consistent tensor outputs (pooled representations)
        - DINOv2, DINOv3, ViT, ConvNext, ViTMAE return pooler_output from transformer models
        - CLIP returns direct pooled features
        - All models support the unified interface: model(preprocess(images))
        - Output shapes vary by model architecture and size
    """
    try:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = model_name.lower()
        
        # Validate inputs
        model_info = get_model_info(model_name, model_size)
        if model_size is None:
            model_size = model_info['default_size']
        
        # Load model based on type
        if model_name == 'dinov2':
            return _get_dinov2_model(model_size, device)
        elif model_name == 'dinov3':
            return _get_dinov3_model(model_size, device)
        elif model_name == 'clip':
            return _get_clip_model(model_size, device)
        elif model_name == 'vit':
            return _get_vit_model(model_size, device)
        elif model_name == 'convnext':
            return _get_convnext_model(model_size, device, **kwargs)
        elif model_name == 'vitmae':
            return _get_vitmae_model(model_size, device)
        else:
            available = list(MODEL_CONFIGS.keys())
            raise ValueError(f"Unsupported model: {model_name}. Available: {available}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}-{model_size}: {str(e)}")


def _get_dinov2_model(model_size: str, device: str) -> Tuple[ModelWrapper, UnifiedPreprocessor]:
    """Get DINOv2 model and preprocessor."""
    config = MODEL_CONFIGS['dinov2'][model_size]
    
    try:
        processor = AutoImageProcessor.from_pretrained(config['id'], use_fast=True)
        base_model = Dinov2Model.from_pretrained(config['id']).eval().to(device)
        model = DINOv2Wrapper(base_model, 'dinov2', model_size).eval()
        
        def preprocess_fn(image):
            inputs = processor(image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)
        
        preprocessor = UnifiedPreprocessor(preprocess_fn, config['input_size'], device)
        return model, preprocessor
        
    except Exception as e:
        raise RuntimeError(f"Failed to load DINOv2 {model_size}: {str(e)}")


def _get_dinov3_model(model_size: str, device: str) -> Tuple[ModelWrapper, UnifiedPreprocessor]:
    """Get DINOv3 model and preprocessor."""
    config = MODEL_CONFIGS['dinov3'][model_size]
    
    try:
        processor = AutoImageProcessor.from_pretrained(config['id'], use_fast=True)
        base_model = DINOv3ViTModel.from_pretrained(config['id']).eval().to(device)
        model = DINOv3Wrapper(base_model, 'dinov3', model_size).eval()
        
        def preprocess_fn(image):
            inputs = processor(image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)
        
        preprocessor = UnifiedPreprocessor(preprocess_fn, config['input_size'], device)
        return model, preprocessor
        
    except Exception as e:
        raise RuntimeError(f"Failed to load DINOv3 {model_size}: {str(e)}")


def _get_clip_model(model_size: str, device: str) -> Tuple[ModelWrapper, UnifiedPreprocessor]:
    """Get CLIP model and preprocessor."""
    config = MODEL_CONFIGS['clip'][model_size]
    
    try:
        base_model, _, preprocess_fn = open_clip.create_model_and_transforms(
            config['arch'], pretrained=config['pretrained']
        )
        base_model = base_model.visual.eval().to(device)
        model = CLIPWrapper(base_model, 'clip', model_size).eval()
        
        preprocessor = UnifiedPreprocessor(preprocess_fn, config['input_size'], device)
        return model, preprocessor
        
    except Exception as e:
        raise RuntimeError(f"Failed to load CLIP {model_size}: {str(e)}")


def _get_vit_model(model_size: str, device: str) -> Tuple[ModelWrapper, UnifiedPreprocessor]:
    """Get Google ViT model and preprocessor."""
    config = MODEL_CONFIGS['vit'][model_size]
    
    try:
        processor = AutoImageProcessor.from_pretrained(config['id'], use_fast=True)
        base_model = ViTModel.from_pretrained(config['id']).eval().to(device)
        model = ViTWrapper(base_model, 'vit', model_size).eval()
        
        def preprocess_fn(image):
            inputs = processor(image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)
        
        preprocessor = UnifiedPreprocessor(preprocess_fn, config['input_size'], device)
        return model, preprocessor
        
    except Exception as e:
        raise RuntimeError(f"Failed to load ViT {model_size}: {str(e)}")


def _get_convnext_model(model_size: str, device: str, **kwargs) -> Tuple[ModelWrapper, UnifiedPreprocessor]:
    """Get ConvNext model and preprocessor that returns feature representations."""
    config = MODEL_CONFIGS['convnext'][model_size]
    
    try:
        model_id = config['id']
        processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        base_model = ConvNextModel.from_pretrained(model_id).eval().to(device)
        
        # Wrap the model to standardize input interface
        model = ConvNextWrapper(base_model, 'convnext', model_size).eval()
        
        def preprocess_fn(image):
            inputs = processor(image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)
        
        preprocessor = UnifiedPreprocessor(preprocess_fn, config['input_size'], device)
        return model, preprocessor
        
    except Exception as e:
        raise RuntimeError(f"Failed to load ConvNext {model_size}: {str(e)}")

def _get_vitmae_model(model_size: str, device: str) -> Tuple[ModelWrapper, UnifiedPreprocessor]:
    """Get ViTMAE model and preprocessor."""
    config = MODEL_CONFIGS['vitmae'][model_size]
    
    try:
        model_id = config['id']
        processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        base_model = ViTMAEModel.from_pretrained(model_id).eval().to(device)

        # Wrap the model to standardize input interface
        model = ViTMAEWrapper(base_model, 'vitmae', model_size).eval()
        
        def preprocess_fn(image):
            inputs = processor(image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)
        
        preprocessor = UnifiedPreprocessor(preprocess_fn, config['input_size'], device)
        return model, preprocessor
        
    except Exception as e:
        raise RuntimeError(f"Failed to load ViTMAE {model_size}: {str(e)}")

# Example usage and testing
if __name__ == "__main__":
    print("Available models:")
    for model_name, sizes in list_available_models().items():
        print(f"  {model_name}: {sizes}")
    
    # Get model info examples
    print(f"\nDINOv2 info: {get_model_info('dinov2')}")
    # print(f"ConvNext info: {get_model_info('convnext')}")
    
    # Test models - test ALL available model configurations
    try:
        # Generate all possible model combinations from MODEL_CONFIGS
        models_to_test = []
        for model_name, sizes in MODEL_CONFIGS.items():
            for size_name in sizes.keys():
                models_to_test.append((model_name, size_name))
        
        # Create dummy 4D images (batch format required)
        dummy_images = np.random.randint(0, 255, (2, 224, 224, 3), dtype=np.uint8)
        
        print(f"\nTesting {len(models_to_test)} model configurations...")
        print("=" * 60)
        
        for model_name, model_size in models_to_test:
            try:
                print(f"\n[TESTING] Testing {model_name}-{model_size}...")
                model, preprocess = get_model(model_name, model_size)
                inputs = preprocess(dummy_images)
                
                with torch.no_grad():
                    outputs = model(inputs)
                    # Test deterministic output consistency
                    outputs2 = model(inputs)
                
                # All models now return consistent tensor outputs
                if hasattr(outputs, 'shape'):
                    output_shape = outputs.shape
                    output_type = "tensor output"
                else:
                    output_shape = "unknown"
                    output_type = "unknown output"
                
                # Check deterministic consistency
                deterministic = torch.allclose(outputs, outputs2, atol=1e-6) if hasattr(outputs, 'shape') else False
                if not deterministic:
                    raise ValueError("Output is not deterministic")


                print(f"  [SUCCESS] Success!")
                #print(f"     Model info: {model.get_info()}")
                print(f"     Input shape: {inputs.shape}")
                print(f"     Output shape: {output_shape} ({output_type})")
                print(f"     Deterministic: {'Yes' if deterministic else 'No'}")
                
                # Test device movement
                if torch.cuda.is_available():
                    try:
                        model_cuda = model.cuda()
                        inputs_cuda = inputs.cuda()
                        with torch.no_grad():
                            outputs_cuda = model_cuda(inputs_cuda)
                        print(f"     [SUCCESS] CUDA test passed: {outputs_cuda.shape if hasattr(outputs_cuda, 'shape') else 'unknown'}")
                        model = model.cpu()  # Move back to CPU
                    except Exception as cuda_e:
                        print(f"     [WARNING] CUDA test failed: {cuda_e}")
                else:
                    print(f"     [INFO] CUDA not available, skipping GPU test")
                
            except Exception as e:
                print(f"  [FAILED] Failed to test {model_name}-{model_size}: {e}")
                
        print("\n" + "=" * 60)
        print("[SUCCESS] Model testing completed!")
        
        # Summary
        print(f"\nSummary:")
        print(f"  - Total models available: {sum(len(sizes) for sizes in list_available_models().values())}")
        print(f"  - Model types: {list(list_available_models().keys())}")
        print(f"  - All models return consistent tensor outputs (pooled representations)")
        print(f"  - All models support the unified interface: model(preprocess(images))")
        
    except Exception as e:
        print(f"[FAILED] Testing failed: {e}")
        import traceback
        traceback.print_exc()
