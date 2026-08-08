from typing import Sequence, Optional
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys.llaisys_types import DataType
from ..libllaisys.tensor import llaisysTensor_t
from ..libllaisys.qwen2 import llaisysQwen2Model_t

from pathlib import Path
import safetensors.torch
import torch
import numpy as np
import ctypes
import math


# Qwen2 model configuration for DeepSeek-R1-Distill-Qwen-1.5B
QWEN2_CONFIG = {
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "num_hidden_layers": 28,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 131072,
    "vocab_size": 151936,
    "tie_word_embeddings": False,
}


def _create_tensor(shape, dtype, device_type=DeviceType.CPU, device_id=0):
    """Create a tensor via the C API."""
    ndim = len(shape)
    shape_arr = (ctypes.c_size_t * ndim)(*shape)
    tensor = LIB_LLAISYS.tensorCreate(
        shape_arr,
        ctypes.c_size_t(ndim),
        ctypes.c_int(dtype.value),
        ctypes.c_int(device_type.value),
        ctypes.c_int(device_id),
    )
    return tensor


def _destroy_tensor(tensor):
    """Destroy a tensor via the C API."""
    LIB_LLAISYS.tensorDestroy(tensor)


def _tensor_load(tensor, data):
    """Load numpy data into a tensor via the C API."""
    data_ptr = data.ctypes.data_as(ctypes.c_void_p)
    LIB_LLAISYS.tensorLoad(tensor, data_ptr)


def _tensor_get_data(tensor):
    """Get raw data pointer from a tensor."""
    return LIB_LLAISYS.tensorGetData(tensor)


def _tensor_get_shape(tensor):
    """Get shape of a tensor."""
    ndim = LIB_LLAISYS.tensorGetNdim(tensor)
    shape_arr = (ctypes.c_size_t * ndim)()
    LIB_LLAISYS.tensorGetShape(tensor, shape_arr)
    return [shape_arr[i] for i in range(ndim)]


def _tensor_view(tensor, shape):
    """Create a view of a tensor with new shape."""
    ndim = len(shape)
    shape_arr = (ctypes.c_size_t * ndim)(*shape)
    return LIB_LLAISYS.tensorView(tensor, shape_arr, ctypes.c_size_t(ndim))


def _tensor_permute(tensor, order):
    """Permute dimensions of a tensor."""
    ndim = len(order)
    order_arr = (ctypes.c_size_t * ndim)(*order)
    return LIB_LLAISYS.tensorPermute(tensor, order_arr)


def _tensor_slice(tensor, dim, start, end):
    """Slice a tensor along a dimension."""
    return LIB_LLAISYS.tensorSlice(
        tensor,
        ctypes.c_size_t(dim),
        ctypes.c_size_t(start),
        ctypes.c_size_t(end),
    )


def _tensor_is_contiguous(tensor):
    """Check if tensor is contiguous."""
    return bool(LIB_LLAISYS.tensorIsContiguous(tensor))


class Qwen2:
    """Qwen2 model wrapper using LLAISYS C/C++ backend."""

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.device = device
        self.device_id = 0
        self.cfg = dict(QWEN2_CONFIG)

        # Create the C model handle
        self._model = LIB_LLAISYS.qwen2Create(
            ctypes.c_int(device.value),
            ctypes.c_int(self.device_id),
        )

        # Load weights from safetensors
        model_path = Path(model_path)
        self._load_weights(model_path)

    def _load_weights(self, model_path: Path):
        """Load all weights from safetensors files into the C model."""
        safetensor_files = sorted(model_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(
                f"No .safetensors files found in {model_path}"
            )

        for file in safetensor_files:
            print(f"Loading {file.name} ...")
            # Use safetensors.torch to load bfloat16 weights properly
            state_dict = safetensors.torch.load_file(str(file), device="cpu")
            for name_, tensor_torch in state_dict.items():
                # Convert torch tensor to numpy float32
                if tensor_torch.dtype == torch.bfloat16:
                    tensor_np = tensor_torch.float().numpy()
                elif tensor_torch.dtype == torch.float16:
                    tensor_np = tensor_torch.float().numpy()
                elif tensor_torch.dtype == torch.float64:
                    tensor_np = tensor_torch.float().numpy()
                else:
                    tensor_np = tensor_torch.numpy()

                # Create LLAISYS tensor and load data
                shape = list(tensor_np.shape)
                tensor = _create_tensor(shape, DataType.F32, self.device, self.device_id)
                _tensor_load(tensor, tensor_np)

                # Register weight in the C model
                LIB_LLAISYS.qwen2LoadWeight(
                    self._model,
                    name_.encode("utf-8"),
                    tensor,
                )

                # We don't destroy the tensor here - the model takes ownership

    def _sample_top_k(self, logits: np.ndarray, k: int) -> int:
        """Sample from top-k logits."""
        if k <= 0:
            k = 1
        k = min(k, len(logits))
        indices = np.argpartition(logits, -k)[-k:]
        top_k_logits = logits[indices]
        # Convert to probabilities
        top_k_logits = top_k_logits - np.max(top_k_logits)
        probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
        # Sample
        idx = np.random.choice(len(indices), p=probs)
        return int(indices[idx])

    def _sample_top_p(self, logits: np.ndarray, p: float) -> int:
        """Nucleus (top-p) sampling."""
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        cumulative_probs = np.cumsum(np.exp(sorted_logits - np.max(sorted_logits)) /
                                      np.sum(np.exp(sorted_logits - np.max(sorted_logits))))
        # Find cutoff
        cutoff_idx = np.searchsorted(cumulative_probs, p) + 1
        top_p_indices = sorted_indices[:cutoff_idx]
        top_p_logits = logits[top_p_indices]
        # Softmax
        top_p_logits = top_p_logits - np.max(top_p_logits)
        probs = np.exp(top_p_logits) / np.sum(np.exp(top_p_logits))
        # Sample
        idx = np.random.choice(len(top_p_indices), p=probs)
        return int(top_p_indices[idx])

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> Sequence[int]:
        """Generate tokens given input token IDs.

        Args:
            inputs: Input token IDs (pre-tokenized).
            max_new_tokens: Maximum number of tokens to generate.
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter.
            temperature: Temperature for sampling.

        Returns:
            List of generated token IDs (including input).
        """
        vocab_size = self.cfg["vocab_size"]

        if max_new_tokens is None:
            max_new_tokens = self.cfg["max_position_embeddings"] - len(inputs)

        # Reset KV cache for new generation
        LIB_LLAISYS.qwen2ResetKV(self._model)

        generated = list(inputs)

        # Prefill: process all input tokens at once
        input_ids_np = np.array(inputs, dtype=np.int64)
        input_tensor = _create_tensor(
            [len(inputs)], DataType.I64, DeviceType.CPU, self.device_id
        )
        _tensor_load(input_tensor, input_ids_np)

        # Create output logits tensor
        logits_tensor = _create_tensor(
            [vocab_size], DataType.F32, DeviceType.CPU, self.device_id
        )

        # Forward pass for prefill
        LIB_LLAISYS.qwen2Forward(self._model, input_tensor, logits_tensor)

        # Get logits data
        logits_ptr = _tensor_get_data(logits_tensor)
        logits_np = np.ctypeslib.as_array(
            ctypes.cast(logits_ptr, ctypes.POINTER(ctypes.c_float)),
            shape=(vocab_size,),
        ).copy()

        # Sample next token
        if temperature > 0:
            logits_np = logits_np / temperature

        if top_k > 1:
            next_token = self._sample_top_k(logits_np, top_k)
        elif top_p < 1.0:
            next_token = self._sample_top_p(logits_np, top_p)
        else:
            next_token = int(np.argmax(logits_np))

        generated.append(next_token)

        _destroy_tensor(input_tensor)
        _destroy_tensor(logits_tensor)

        # Autoregressive generation: generate one token at a time
        for step in range(1, max_new_tokens):
            # Single token input
            input_ids_np = np.array([next_token], dtype=np.int64)
            input_tensor = _create_tensor(
                [1], DataType.I64, DeviceType.CPU, self.device_id
            )
            _tensor_load(input_tensor, input_ids_np)

            logits_tensor = _create_tensor(
                [vocab_size], DataType.F32, DeviceType.CPU, self.device_id
            )

            # Forward pass (uses KV cache internally)
            LIB_LLAISYS.qwen2Forward(self._model, input_tensor, logits_tensor)

            # Get logits
            logits_ptr = _tensor_get_data(logits_tensor)
            logits_np = np.ctypeslib.as_array(
                ctypes.cast(logits_ptr, ctypes.POINTER(ctypes.c_float)),
                shape=(vocab_size,),
            ).copy()

            # Sample
            if temperature > 0:
                logits_np = logits_np / temperature

            if top_k > 1:
                next_token = self._sample_top_k(logits_np, top_k)
            elif top_p < 1.0:
                next_token = self._sample_top_p(logits_np, top_p)
            else:
                next_token = int(np.argmax(logits_np))

            generated.append(next_token)

            _destroy_tensor(input_tensor)
            _destroy_tensor(logits_tensor)

        return generated

    def __del__(self):
        """Clean up the C model handle."""
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.qwen2Destroy(self._model)
