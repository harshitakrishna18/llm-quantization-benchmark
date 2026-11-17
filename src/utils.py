import torch
import platform
import transformers
import accelerate
import bitsandbytes

def get_model_size_mb(model):
    total_params = sum(p.numel() * p.element_size() for p in model.parameters())
    total_buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return (total_params + total_buffers) / (1024 ** 2)

def get_runtime_info(model):
        print("DEBUG MPS Backend Availability: ", torch.backends.mps.is_available())
        print("DEBUG CUDA Backend Availability: ", torch.cuda.is_available())
        print("DEBUG model.device: ", model.device)
        raw_device_map = getattr(model, "hf_device_map", None)
        device_map = (
              {name: str(device) for name, device in raw_device_map.items()}
              if raw_device_map is not None
              else None
        )
        print("DEBUG model device map:", device_map)

        quantization_config = getattr(model.config, "quantization_config", None)

        if hasattr(quantization_config, "to_dict"):
              quantization_config = quantization_config.to_dict()

        return {"mps_available" : torch.backends.mps.is_available(), 
                "mps_built": torch.backends.mps.is_built(), 
                "cuda_available":torch.cuda.is_available(), 
                "cuda_version": torch.version.cuda,
                "model_device": str(model.device), 
                "model_device_map": device_map, 
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "accelerate_version": accelerate.__version__,
                "bitsandbytes_version": bitsandbytes.__version__,
                "requested_quantization_config": quantization_config}