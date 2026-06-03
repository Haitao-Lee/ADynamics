"""
Multi-GPU training utilities for ADynamics.

Provides MultiModalDataParallel wrapper that properly handles dict inputs
(MultiModalVAE3D takes a dict of {modality_name: tensor}).
"""

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel


class MultiModalDataParallel(DataParallel):
    """
    DataParallel wrapper that properly handles dict inputs for MultiModalVAE3D.

    Standard DataParallel only works with tensor/list inputs. Our multi-modal
    VAE takes a dict like {"t1": tensor, "fmri": tensor, ...}, so we need
    custom scatter/gather logic.
    """

    def scatter(self, inputs: Any, kwargs: Dict[str, Any], device_ids: List[int]) -> tuple:
        """Scatter dict inputs to multiple GPUs. Replicates short batches.

        Returns (input_shards, kwargs_shards) where each is a list of length
        num_gpus, matching what torch.nn.parallel.parallel_apply expects.

        Handles three call patterns:
        - inputs is a dict (e.g. self.model(x=x_dict, ...))
        - inputs is a tuple with dict as first element (e.g. self.model(x_dict, return_components=True))
        - inputs is something else (fall back to default scatter)
        """
        # If the model was called with the dict as a positional arg, DataParallel
        # wraps the inputs as a tuple where the first element is our dict.
        if isinstance(inputs, tuple) and len(inputs) >= 1 and isinstance(inputs[0], dict):
            inputs = inputs[0]

        if not isinstance(inputs, dict):
            return super().scatter(inputs, kwargs, device_ids)

        batch_size = -1
        for v in inputs.values():
            if isinstance(v, torch.Tensor):
                batch_size = v.size(0)
                break
        if batch_size == -1:
            inputs_shards = [(inputs,) for _ in device_ids]
            kwargs_shards = [kwargs for _ in device_ids]
            return inputs_shards, kwargs_shards

        num_gpus = len(device_ids)
        if batch_size < num_gpus:
            # Replicate samples to fill all GPUs
            replicates = num_gpus // batch_size + (1 if num_gpus % batch_size else 0)
            replicated = {k: (v.repeat(replicates, *([1] * (v.dim() - 1))) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            inputs = replicated
            batch_size = num_gpus

        inputs_shards = []
        kwargs_shards = []
        chunk_size = batch_size // num_gpus
        remainder = batch_size % num_gpus
        start = 0
        for i, device_id in enumerate(device_ids):
            end = start + chunk_size + (1 if i < remainder else 0)
            sub_dict = {k: (v[start:end].to(device_ids[i]) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            inputs_shards.append((sub_dict,))
            kwargs_shards.append(kwargs)
            start = end
        return inputs_shards, kwargs_shards

    def gather(self, outputs: List[Any], output_device: int) -> Any:
        """Gather outputs from multiple GPUs back to the output device.

        Each output is on a different GPU, so we must move them to the same
        device before concatenation.
        """
        def _move_and_cat(tensors: List[torch.Tensor]) -> torch.Tensor:
            """Move all tensors to output_device then concatenate."""
            moved = [t.to(output_device, non_blocking=True) for t in tensors]
            return torch.cat(moved, dim=0)

        if isinstance(outputs[0], dict):
            gathered = {}
            for key in outputs[0].keys():
                values = [o[key] for o in outputs if key in o]
                if values and isinstance(values[0], torch.Tensor):
                    gathered[key] = _move_and_cat(values)
                else:
                    gathered[key] = values[0]
            return gathered
        elif isinstance(outputs[0], (tuple, list)):
            num_components = len(outputs[0])
            gathered = []
            for i in range(num_components):
                component_outputs = [o[i] for o in outputs]
                if isinstance(component_outputs[0], torch.Tensor):
                    gathered.append(_move_and_cat(component_outputs))
                else:
                    gathered.append(component_outputs[0])
            return tuple(gathered)
        elif isinstance(outputs[0], torch.Tensor):
            return _move_and_cat(outputs)
        else:
            return outputs[0]


def setup_data_parallel(model: nn.Module, num_gpus: int) -> nn.Module:
    """
    Wrap model with MultiModalDataParallel if num_gpus > 1.

    Args:
        model: The model to wrap (e.g. MultiModalVAE3D)
        num_gpus: Number of GPUs (1 = no wrap, >1 = DataParallel)

    Returns:
        Wrapped model (or original if num_gpus <= 1)
    """
    if num_gpus > 1 and torch.cuda.is_available() and torch.cuda.device_count() >= num_gpus:
        print(f"Setting up DataParallel with {num_gpus} GPUs")
        return MultiModalDataParallel(model, device_ids=list(range(num_gpus)))
    else:
        if num_gpus > 1:
            print(f"Warning: requested {num_gpus} GPUs but only {torch.cuda.device_count() if torch.cuda.is_available() else 0} available, falling back to single GPU")
        return model
