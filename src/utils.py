def get_model_size_mb(model):
    total_params = sum(p.numel() * p.element_size() for p in model.parameters())
    total_buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return (total_params + total_buffers) / (1024 ** 2)