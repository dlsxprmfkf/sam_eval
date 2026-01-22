from .wrappers import MLPBlockWrapper, LinearWrapperUnstructured

def apply_sam_pruning(sam_model, args):
    """
    Applies the selected pruning wrapper to the SAM Image Encoder.
    Dispatches between structured ('sp') and unstructured pruning.
    
    Args:
        sam_model: SAM model instance
        args: Arguments containing:
            - sparsity_type: 'sp' (structured) or 'unstructured'
            - prune_mlp_ratio: MLP pruning ratio (0.0 - 1.0)
            - prune_attn_ratio: Attention pruning ratio (0.0 - 1.0)
            - rebuild_freq: How often to rebuild pruning masks
            - act_mode: Activation mode for importance calculation ('rms', etc.)
    
    Returns:
        Modified SAM model with pruning wrappers applied
    """
    sparsity_type = getattr(args, 'sparsity_type', 'sp')
    print(f"Applying '{sparsity_type}' pruning to SAM Image Encoder...")

    try:
        encoder_blocks = sam_model.image_encoder.blocks
    except AttributeError:
        print("Error: Could not find 'image_encoder.blocks' in the provided model.")
        return sam_model

    if sparsity_type == 'sp':
        for i, block in enumerate(encoder_blocks):
            block.mlp = MLPBlockWrapper(
                mlp_mod=block.mlp,
                layer_id=i,
                prune_mlp_ratio=getattr(args, 'prune_mlp_ratio', 0.0),
                rebuild_freq=getattr(args, 'rebuild_freq', 1),
                act_mode=getattr(args, 'act_mode', 'rms'),
            )
    
    elif sparsity_type == 'unstructured':
        prune_mlp_ratio = getattr(args, 'prune_mlp_ratio', 0.0)
        prune_attn_ratio = getattr(args, 'prune_attn_ratio', 0.0)
        rebuild_freq = getattr(args, 'rebuild_freq', 1)
        act_mode = getattr(args, 'act_mode', 'rms')

        for i, block in enumerate(encoder_blocks):
            if prune_mlp_ratio > 0:
                block.mlp.lin1 = LinearWrapperUnstructured(
                    block.mlp.lin1, 
                    prune_ratio=prune_mlp_ratio,
                    rebuild_freq=rebuild_freq,
                    act_mode=act_mode
                )
                block.mlp.lin2 = LinearWrapperUnstructured(
                    block.mlp.lin2,
                    prune_ratio=prune_mlp_ratio,
                    rebuild_freq=rebuild_freq,
                    act_mode=act_mode
                )
            if prune_attn_ratio > 0:
                block.attn.qkv = LinearWrapperUnstructured(
                    block.attn.qkv,
                    prune_ratio=prune_attn_ratio,
                    rebuild_freq=rebuild_freq,
                    act_mode=act_mode
                )
                block.attn.proj = LinearWrapperUnstructured(
                    block.attn.proj,
                    prune_ratio=prune_attn_ratio,
                    rebuild_freq=rebuild_freq,
                    act_mode=act_mode
                )
    else:
        raise ValueError(f"Unsupported sparsity_type: '{sparsity_type}'")

    print(f"Successfully applied '{sparsity_type}' wrapper to {len(encoder_blocks)} blocks in SAM.")
    return sam_model
