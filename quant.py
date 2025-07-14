#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import List

import torch
import torch_npu

from .quant_utils import (SRC_DTYPE_TO_ACL_DTYPE, TYPE_QUANT_QKV_ONLINE,
                          quant_per_tensor)


class AscendFAQuantAttentionMethod:
    """Linear method for Ascend FAQuant
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_quant_param() -> List[str]:
        return [
            "fa_q.scale", "fa_q.offset", "fa_k.scale", "fa_k.offset",
            "fa_v.scale", "fa_v.offset"
        ]

    @staticmethod
    def get_extra_module_names() -> List[str]:
        return ["fa_q", "fa_k", "fa_v"]

    @staticmethod
    def process_weights_after_loading(layer):
        fa_qscale = layer.fa_q.scale
        fa_kscale = layer.fa_k.scale
        fa_vscale = layer.fa_v.scale
        repeated_query_scale = layer.fa_q.scale.repeat(1, 64)
        layer.fa_qscale = torch.nn.Parameter(repeated_query_scale,
                                             requires_grad=False)
        repeated_query_offset = layer.fa_q.offset.repeat(1, 64)
        layer.fa_qoffset = torch.nn.Parameter(repeated_query_offset,
                                              requires_grad=False)
        repeated_fa_kscale = layer.fa_k.scale.repeat(1, 64)
        layer.fa_kscale = torch.nn.Parameter(repeated_fa_kscale,
                                             requires_grad=False)
        repeated_fa_koffset = layer.fa_k.offset.repeat(1, 64)
        layer.fa_koffset = torch.nn.Parameter(repeated_fa_koffset,
                                              requires_grad=False)
        repeated_fa_vscale = layer.fa_v.scale.repeat(1, 64)
        layer.fa_vscale = torch.nn.Parameter(repeated_fa_vscale,
                                             requires_grad=False)
        repeated_fa_voffset = layer.fa_v.offset.repeat(1, 64)
        layer.fa_voffset = torch.nn.Parameter(repeated_fa_voffset,
                                              requires_grad=False)

        if fa_kscale.shape[0] <= 0:
            raise ValueError(
                "Expected size of fa_kscale in dimension 0 should be greater than 0"
                f"but got {fa_kscale.shape[0]}.")
        gqa_size = fa_qscale.shape[0] // fa_kscale.shape[0]
        fa3_k_scale, fa3_v_scale = fa_kscale.repeat(1, gqa_size).view(
            -1, 1), fa_vscale.repeat(1, gqa_size).view(-1, 1)
        qk_scale = torch.nn.Parameter(torch.squeeze(
            fa_qscale * fa3_k_scale).to(torch.float),
                                      requires_grad=False)
        layer.register_parameter("qk_scale", qk_scale)
        fa3_v_scale = torch.nn.Parameter(
            torch.squeeze(fa3_v_scale).contiguous().to(torch.float),
            requires_grad=False)
        layer.register_parameter("fa3_v_scale", fa3_v_scale)

    @classmethod
    def apply(cls, layer: torch.nn.Module, query: torch.Tensor,
              key: torch.Tensor, value: torch.Tensor, *extra_args,
              **optional_args) -> torch.Tensor:
        key_cache, value_cache, scale, block_tables, \
            is_prefill, mask, slots, output = extra_args
        seq_lens_tensor_cpu = optional_args.get("seq_lens_tensor_cpu", None)

        query_shape = query.shape
        key_shape = key.shape
        value_shape = value.shape

        query = query.view(query.shape[0], -1)
        key = key.view(key.shape[0], -1)
        value = value.view(value.shape[0], -1)

        if is_prefill:
            if key_cache is not None:

                key_int8 = quant_per_tensor(key, layer.fa_kscale,
                                            layer.fa_koffset, True)
                value_int8 = quant_per_tensor(value, layer.fa_vscale,
                                              layer.fa_voffset, True)
                key_int8 = key_int8.view(key_shape)
                value_int8 = key_int8.view(value_shape)
                query = query.view(query_shape)
                torch_npu._npu_reshape_and_cache(key_int8, value_int8,
                                                 key_cache, value_cache, slots)
            if mask is None:
                raise ValueError(
                    "attn_metadata.attn_mask is Null. Please check.")
            if output is not None:
                key = key.view(key_shape)
                value = key.view(value_shape)
                query = query.view(query_shape)
                output = output.view(query.shape)
                torch_npu._npu_flash_attention(query,
                                               key,
                                               value,
                                               mask,
                                               torch.tensor(
                                                   seq_lens_tensor_cpu,
                                                   dtype=torch.int32),
                                               scale,
                                               layer.num_heads,
                                               layer.num_kv_heads,
                                               out=output)
            else:
                key = key.view(key_shape)
                value = key.view(value_shape)
                query = query.view(query_shape)
                output = output.view(query.shape)
                output = torch.empty_like(query,
                                          dtype=query.dtype).to(query.device)
                torch_npu._npu_flash_attention(query,
                                               key,
                                               value,
                                               mask,
                                               torch.tensor(
                                                   seq_lens_tensor_cpu,
                                                   dtype=torch.int32),
                                               scale,
                                               layer.num_heads,
                                               layer.num_kv_heads,
                                               out=output)

        else:
            if key_cache is None:
                raise ValueError(
                    "KV Cache can't be None in decoding phase. Got None. Please check."
                )
            query_int8 = quant_per_tensor(query, layer.fa_qscale,
                                          layer.fa_qoffset, True)
            key_int8 = quant_per_tensor(key, layer.fa_kscale, layer.fa_koffset,
                                        True)
            value_int8 = quant_per_tensor(value, layer.fa_vscale,
                                          layer.fa_voffset, True)
            key_int8 = key_int8.view(key_shape)
            value_int8 = value_int8.view(value_shape)
            query = query.view(query_shape)
            query_int8 = query_int8.view(query_shape)
            output = output.view(query.shape)
            torch_npu._npu_reshape_and_cache(key_int8, value_int8, key_cache,
                                             value_cache, slots)
            if output is not None:
                output = output.view(query.shape)
                torch_npu._npu_paged_attention_quant(
                    query_int8, key_cache, value_cache, layer.num_kv_heads,
                    layer.num_heads, scale, block_tables,
                    torch.tensor(seq_lens_tensor_cpu, dtype=torch.int32),
                    TYPE_QUANT_QKV_ONLINE, SRC_DTYPE_TO_ACL_DTYPE[query.dtype],
                    layer.qk_scale, layer.fa3_v_scale, output)
            else:
                output = torch.empty_like(query,
                                          dtype=query.dtype).to(query.device)
                torch_npu._npu_paged_attention_quant(
                    query_int8, key_cache, value_cache, layer.num_kv_heads,
                    layer.num_heads, scale, block_tables,
                    torch.tensor(seq_lens_tensor_cpu, dtype=torch.int32),
                    TYPE_QUANT_QKV_ONLINE, SRC_DTYPE_TO_ACL_DTYPE[query.dtype],
                    layer.qk_scale, layer.fa3_v_scale, output)

        output = torch.flatten(output, start_dim=-2)
        return output

    @classmethod
    def create_weights(cls, layer: torch.nn.Module) -> None:
        extra_module_names = cls.get_extra_module_names()
        for name in extra_module_names:
            setattr(layer, name, torch.nn.Module())

        params_dtype = torch.get_default_dtype()

        params_dict = {}

        params_dict["fa_q.scale"] = torch.empty((layer.num_heads, 1),
                                                dtype=params_dtype)
        params_dict["fa_q.offset"] = torch.empty((layer.num_heads, 1),
                                                 dtype=torch.int8)
        params_dict["fa_k.scale"] = torch.empty((layer.num_kv_heads, 1),
                                                dtype=params_dtype)
        params_dict["fa_k.offset"] = torch.empty((layer.num_kv_heads, 1),
                                                 dtype=torch.int8)
        params_dict["fa_v.scale"] = torch.empty((layer.num_kv_heads, 1),
                                                dtype=params_dtype)
        params_dict["fa_v.offset"] = torch.empty((layer.num_kv_heads, 1),
                                                 dtype=torch.int8)

        for name, weight in params_dict.items():
            module_name, weight_name = name.split('.')
            module = getattr(layer, module_name)
            module.register_parameter(
                weight_name, torch.nn.Parameter(weight, requires_grad=False))


from .quant_utils import quant_per_tensor


class FAQuantizer(VLLMAscendQuantizer):

    @staticmethod
    def build_attention_method():
        return AscendFAQuantAttentionMethod()
            if "fa_quant_type" in name:
                VLLMAscendQuantizer.apply_patch(
                    "vllm.model_executor.model_loader.weight_utils",
                    "safetensors_weights_iterator",
                    [wrapper_weights_iterator(safetensors_weights_iterator)],
                )
                VLLMAscendQuantizer.apply_patch(
                    "vllm.worker.cache_engine.CacheEngine",
                    "__init__",
                    [cache_engine_init],
                )
                VLLMAscendQuantizer.apply_patch(
                    "vllm.attention.layer.Attention",
                    "__init__",
                    [attention_init],
                )


from vllm.model_executor.layers.quantization.base_config import \
    QuantizationConfig
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.platforms import current_platform
from vllm.utils import LayerBlockType

TYPE_QUANT_QKV_ONLINE = 3

SRC_DTYPE_TO_ACL_DTYPE = {
    torch.float16: 1,
    torch.bfloat16: 27,
}


def quant_per_tensor(in_tensor: torch.Tensor,
                     input_scale: torch.Tensor,
                     input_offset: torch.Tensor,
                     function=False):
    input_scale = input_scale.view(-1)
    input_offset = input_offset.view(-1)
    return torch_npu.npu_quantize(in_tensor, input_scale, input_offset,
                                  torch.qint8, -1, function)


def wrapper_weights_iterator(func):

    def _safetensors_weights_iterator(
        hf_weights_files: List[str]
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        current_rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        for name, weight in func(hf_weights_files):
            # The name of attention weights generated by msmodelslim
            # must be modified so that these weights can be loaded
            # into Attention module rather than LlamaAttention module.
            if "fa_" in name and '.attn.' not in name:
                name = name.split('.')
                name.insert(name.index('self_attn') + 1, 'attn')
                name = '.'.join(name)
                # vLLM originally does not support splitting attention
                # weights with respect to TP ranks. We need split
                # weights manually here.
                split_size = weight.size(0) // world_size
                weight = weight[current_rank * split_size:(current_rank + 1) *
                                split_size]

            # msmodelslim add these two extra weights for pd-mix cases.
            # Currently we have to ignore these weights, and will load
            #  these weights once dynamic quantization is supported
            if "weight_scale" in name or "weight_offset" in name:
                continue
            yield name, weight

    return _safetensors_weights_iterator


# Replace CacheEngine.__init__
# vLLM does not include int8 cache dtype.
# We should set it here.
def cache_engine_init(
    self,
    cache_config: CacheConfig,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    device_config: DeviceConfig,
) -> None:
    self.cache_config = cache_config
    self.model_config = model_config
    self.parallel_config = parallel_config
    self.device_config = device_config

    self.head_size = model_config.get_head_size()
    # Models like Jamba, have mixed typed layers, E.g Mamba
    self.num_attention_layers = model_config.get_num_layers_by_block_type(
        parallel_config, LayerBlockType.attention)
    self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

    self.block_size = cache_config.block_size
    self.num_gpu_blocks = cache_config.num_gpu_blocks
    if self.num_gpu_blocks:
        self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
    self.num_cpu_blocks = cache_config.num_cpu_blocks
    if self.num_cpu_blocks:
        self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

    # modified here. vLLM does not include int8 cache dtype.
    # We should set it here.
    self.dtype = torch.int8

    # Get attention backend.
    self.attn_backend = get_attn_backend(self.head_size,
                                         model_config.dtype,
                                         cache_config.cache_dtype,
                                         self.block_size,
                                         model_config.is_attention_free,
                                         use_mla=model_config.use_mla)

    # Initialize the cache.
    self.gpu_cache = self._allocate_kv_cache(self.num_gpu_blocks,
                                             self.device_config.device_type)
    self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")


# Replace Attention.__init__
# Need set attributes including num_heads, head_size and num_kv_heads
# before quant_method is initialized.
def attention_init(
    self,
    num_heads: int,
    head_size: int,
    scale: float,
    num_kv_heads: Optional[int] = None,
    alibi_slopes: Optional[List[float]] = None,
    cache_config: Optional[CacheConfig] = None,
    quant_config: Optional[QuantizationConfig] = None,
    blocksparse_params: Optional[Dict[str, Any]] = None,
    logits_soft_cap: Optional[float] = None,
    per_layer_sliding_window: Optional[int] = None,
    use_mla: bool = False,
    prefix: str = "",
    attn_type: str = AttentionType.DECODER,
    **extra_impl_args,
) -> None:
    super(Attention, self).__init__()
    if per_layer_sliding_window is not None:
        # per-layer sliding window
        sliding_window = per_layer_sliding_window
    elif cache_config is not None:
        # model-level sliding window
        sliding_window = cache_config.sliding_window
    else:
        sliding_window = None

    if cache_config is not None:
        kv_cache_dtype = cache_config.cache_dtype
        block_size = cache_config.block_size
        is_attention_free = cache_config.is_attention_free
        calculate_kv_scales = cache_config.calculate_kv_scales
    else:
        kv_cache_dtype = "auto"
        block_size = 16
        is_attention_free = False
        calculate_kv_scales = False
    if num_kv_heads is None:
        num_kv_heads = num_heads

    # The default k/v_scale is set to 1.0. This is ignored
    # when kv-cache is not fp8, and should be used with
    # kv-cache in fp8_e5m2. For kv-cache in fp8_e4m3, we
    # expect the pre-quantized k/v_scale to be loaded along
    # with the model weights.
    self.kv_cache_dtype = kv_cache_dtype
    self.calculate_kv_scales = calculate_kv_scales
    self._k_scale = torch.tensor(1.0, dtype=torch.float32)
    self._v_scale = torch.tensor(1.0, dtype=torch.float32)

    # We also keep the float32 versions of k/v_scale for attention
    # backends that don't support tensors (Flashinfer)
    self._k_scale_float = 1.0
    self._v_scale_float = 1.0

    # Modified here.
    self.num_heads = num_heads
    self.head_size = head_size
    self.num_kv_heads = num_kv_heads

    quant_method = quant_config.get_quant_method(
        self, prefix=prefix) if quant_config else None
    if quant_method is not None:
        assert isinstance(quant_method, BaseKVCacheMethod)
        # TODO (mgoin): kv cache dtype should be specified in the FP8
        # checkpoint config and become the "auto" behavior
        if self.kv_cache_dtype == "fp8_e5m2":
            raise ValueError("fp8_e5m2 kv-cache is not supported with "
                             "fp8 checkpoints.")
        # If quantization is enabled, we make "k_scale" and "v_scale"
        # parameters so that it can be loaded from the model checkpoint.
        # The k/v_scale will then be converted back to native float32
        # values after weight loading.
        self.quant_method = quant_method
        self.quant_method.create_weights(self)

    # During model initialization, the default dtype is set as the model
    # weight and activation dtype.
    dtype = torch.get_default_dtype()
    attn_backend = get_attn_backend(head_size,
                                    dtype,
                                    kv_cache_dtype,
                                    block_size,
                                    is_attention_free,
                                    blocksparse_params is not None,
                                    use_mla=use_mla)
    impl_cls = attn_backend.get_impl_cls()
    self.impl = impl_cls(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         **extra_impl_args)

    self.sliding_window = sliding_window
    self.backend = backend_name_to_enum(attn_backend.get_name())
    self.dtype = dtype

    # For cuda-alike (CUDA and ROCM) and cpu platforms, we control how
    # torch.compile works by registering the attention as one giant
    # opaque custom op. For other platforms, we directly call them
    # and let torch.compile handle them.
    self.use_direct_call = not current_platform.is_cuda_alike(
    ) and not current_platform.is_cpu()

    self.use_output = attn_backend.accept_output_buffer
    compilation_config = get_current_vllm_config().compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError(f"Duplicate layer name: {prefix}")
    compilation_config.static_forward_context[prefix] = self
    self.layer_name = prefix
    self.attn_type = attn_type
    # use a placeholder kv cache tensor during init, which will be replaced
    # by bind_kv_cache
    # this variable will not be accessed if use_direct_call is True
    self.kv_cache = [
        torch.tensor([]) for _ in range(
            get_current_vllm_config().parallel_config.pipeline_parallel_size)
    ]

    self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
    self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)

