from jittor import nn


def get_param_groups(module) -> tuple:
    """
    Separates module parameters into three groups for the optimizer:
    - Group 0: weights with decay (e.g., from Conv, Linear layers)
    - Group 1: weights without decay (from normalization layers and Sobel filters)
    - Group 2: all bias parameters
    """
    groups = ([], [], [])  # (weights_decay, weights_no_decay, biases)
    bn_types = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k and isinstance(v, type))
    
    processed_params = set()

    # Use named_modules to identify Sobel filters by name
    for name, m in module.named_modules():
        # Check for Sobel modules based on their names in the model hierarchy
        # e.g., 'dmrm.grad_ir'
        is_sobel_module = '.grad_ir' in name or '.grad_vi' in name

        # Add bias to the bias group (no decay)
        if hasattr(m, 'bias') and m.bias is not None:
            # use id to avoid adding the same parameter tensor multiple times
            if id(m.bias) not in processed_params:
                groups[2].append(m.bias)
                processed_params.add(id(m.bias))

        # Add weights to decay or no-decay group
        if hasattr(m, 'weight') and m.weight is not None:
            if id(m.weight) not in processed_params:
                # Add weights from Norm layers and our custom Sobel layers to the no-decay group
                if isinstance(m, bn_types) or is_sobel_module:
                    groups[1].append(m.weight)
                # Add all other weights to the decay group
                else:
                    groups[0].append(m.weight)
                processed_params.add(id(m.weight))
                
    return groups
