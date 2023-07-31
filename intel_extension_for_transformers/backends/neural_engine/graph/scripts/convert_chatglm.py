"""
Convert Hugging Face ChatGLM/ChatGLM2 models to NE format
"""
import argparse
import platform
import struct
import sys
from enum import Enum
from pathlib import Path

import torch
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

NE_QK8_0 = 32
NE_QK4_0 = 32
NE_QK4_1 = 32
NE_QK5_0 = 32
NE_QK5_1 = 32

NE_MEM_ALIGN = 16

if platform.system() == "Darwin":
    # cpm_kernels doesn't support macOS but transformers will check missing packages, so mock it
    sys.modules["cpm_kernels"] = object()


class NEType(Enum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8


class ModelType(Enum):
    CHATGLM = 1
    CHATGLM2 = 2


def quantize_q8_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ne_quantize_q8_0 in ne.c
    assert tensor.shape[1] % NE_QK8_0 == 0
    tensor = tensor.view(-1, NE_QK8_0)
    scale = tensor.abs().max(dim=-1, keepdim=True).values / ((1 << 7) - 1)
    tensor = (tensor / scale).round().clamp(min=-128, max=127).char()
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor


def quantize_q4_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ne_quantize_q4_0 in ne.c
    assert tensor.shape[1] % NE_QK4_0 == 0
    tensor = tensor.view(-1, NE_QK4_0)
    abs_max_indices = tensor.abs().max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    scale = max_values / -8
    tensor = (tensor / scale + 8).round().clamp(min=0, max=15).char()
    # compress two int4 weights into an int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor


def quantize_q4_1(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ne_quantize_q4_1 in ne.c
    assert tensor.shape[1] % NE_QK4_1 == 0
    tensor = tensor.view(-1, NE_QK4_1)
    min_vals = tensor.min(dim=-1, keepdim=True).values
    max_vals = tensor.max(dim=-1, keepdim=True).values
    scale = (max_vals - min_vals) / ((1 << 4) - 1)
    tensor = ((tensor - min_vals) / scale).round().clamp(min=0, max=15).char()
    # compress two int4 weights into an int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    # add scale & min into each block
    tensor = torch.cat((scale.half().view(torch.int8), min_vals.half().view(torch.int8), tensor), dim=-1)
    return tensor


def quantize_q5_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ne_quantize_q5_0 in ne.c
    assert tensor.shape[1] % NE_QK5_0 == 0
    tensor = tensor.view(-1, NE_QK5_0)
    abs_max_indices = tensor.abs().max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    scale = max_values / -16
    tensor = (tensor / scale + 16).round().clamp(min=0, max=31).char()
    qs = (tensor[:, :16] & 0x0F) | (tensor[:, 16:] << 4)
    qh = torch.zeros(tensor.shape[:-1], dtype=torch.int32)
    for i in range(32):
        qh |= ((tensor[:, i] & 0x10) >> 4).int() << i

    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), qh[..., None].view(torch.int8), qs), dim=-1)
    return tensor


def quantize_q5_1(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ne_quantize_q5_1 in ne.c
    assert tensor.shape[1] % NE_QK5_1 == 0
    tensor = tensor.view(-1, NE_QK5_1)
    min_vals = tensor.min(dim=-1, keepdim=True).values
    max_vals = tensor.max(dim=-1, keepdim=True).values
    scale = (max_vals - min_vals) / ((1 << 5) - 1)
    tensor = ((tensor - min_vals) / scale).round().clamp(min=0, max=31).char()
    qs = (tensor[:, :16] & 0x0F) | (tensor[:, 16:] << 4)
    qh = torch.zeros(tensor.shape[:-1], dtype=torch.int32)
    for i in range(32):
        qh |= ((tensor[:, i] & 0x10) >> 4).int() << i

    # add scale & min into each block
    tensor = torch.cat(
        (scale.half().view(torch.int8), min_vals.half().view(torch.int8), qh[..., None].view(torch.int8), qs), dim=-1
    )
    return tensor


def dump_tensor(f, name: str, tensor: torch.Tensor, ne_type: NEType):
    assert tensor.dtype == torch.float32

    # tensor name
    f.write(struct.pack("i", len(name.encode())))
    f.write(name.encode())

    # tensor shape & dtype
    f.write(struct.pack("i" * (2 + tensor.ndim), tensor.ndim, *tensor.shape, ne_type.value))

    # tensor data
    if ne_type == NEType.F32:
        tensor = tensor.float()
    elif ne_type == NEType.F16:
        tensor = tensor.half()
    elif ne_type == NEType.Q8_0:
        tensor = quantize_q8_0(tensor)
    elif ne_type == NEType.Q4_0:
        tensor = quantize_q4_0(tensor)
    elif ne_type == NEType.Q4_1:
        tensor = quantize_q4_1(tensor)
    elif ne_type == NEType.Q5_0:
        tensor = quantize_q5_0(tensor)
    elif ne_type == NEType.Q5_1:
        tensor = quantize_q5_1(tensor)
    else:
        raise NotImplementedError(f"Cannot dump tensor of dtype {tensor.dtype}")

    # align address
    aligned_pos = (f.tell() + (NE_MEM_ALIGN - 1)) // NE_MEM_ALIGN * NE_MEM_ALIGN
    f.seek(aligned_pos)
    tensor.numpy().tofile(f)


def dump_state_dict(f, weight_names, state_dict, quantization_bit, ne_type):
    tensor_info = []
    for name in tqdm(weight_names, desc="Dumping model state"):
        tensor = state_dict[name]
        if tensor.ndim == 2:
            # 2d weight: should quantize it if needed

            # step 1: de-quantize it back to float32
            if tensor.dtype == torch.int8:
                assert quantization_bit in [4, 8]
                scale = state_dict[f"{name}_scale"].float()  # channel-wise scale

                if quantization_bit == 4:
                    # convert int4 weight to int8
                    low_bits = ((tensor << 4) & 0xF0) >> 4
                    high_bits = (tensor & 0xF0) >> 4
                    tensor = torch.stack((high_bits, low_bits), dim=-1).view(tensor.shape[0], -1)
                tensor = tensor * scale[:, None]
            else:
                tensor = tensor.float()

            # step 2: quantize it into ne format
            tensor_ne_type = ne_type
        else:
            # 1d weight: convert it to float32
            assert tensor.ndim == 1
            tensor = tensor.float()
            tensor_ne_type = NEType.F32

        dump_tensor(f, name, tensor, tensor_ne_type)
        tensor_info.append((name, tensor.shape, tensor_ne_type.name))

    print(tabulate(tensor_info, headers=["name", "shape", "dtype"], tablefmt="psql"))


class BaseConverter:
    @classmethod
    def convert(cls, model, tokenizer, ne_type, save_path):
        # convert all weights to fp16
        with open(save_path, "wb") as f:
            f.write(b"ne")  # magic
            f.write(struct.pack("ii", cls.MODEL_TYPE.value, 1))  # model type & version
            cls.dump_config(f, model.config, ne_type)
            cls.dump_tokenizer(f, tokenizer)
            cls.dump_model(f, model, ne_type)

        print(f"{cls.MODEL_TYPE.name} NE model saved to {save_path}")


class ChatGLMConverter(BaseConverter):
    MODEL_TYPE = ModelType.CHATGLM

    @staticmethod
    def dump_config(f, config, ne_type):
        assert config.position_encoding_2d, "unimplemented: position_encoding_2d should be True"
        assert (
            config.inner_hidden_size == 4 * config.hidden_size
        ), "unimplemented: inner_hidden_size should be 4 times hidden_size"
        config_values = [
            ne_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_layers,
            config.inner_hidden_size,
            config.max_sequence_length,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def dump_tokenizer(f, tokenizer):
        serialized_model_proto = tokenizer.sp_tokenizer.text_tokenizer.sp.serialized_model_proto()
        f.write(struct.pack("i", len(serialized_model_proto)))
        f.write(serialized_model_proto)

    @staticmethod
    def dump_model(f, model, ne_type):
        assert torch.allclose(
            model.state_dict()["transformer.word_embeddings.weight"], model.state_dict()["lm_head.weight"]
        ), "unimplemented: lm_head weight must be tied to input embedding"

        weight_names = ["transformer.word_embeddings.weight"]
        for i in range(model.config.num_layers):
            weight_names += [
                f"transformer.layers.{i}.input_layernorm.weight",
                f"transformer.layers.{i}.input_layernorm.bias",
                f"transformer.layers.{i}.attention.query_key_value.weight",
                f"transformer.layers.{i}.attention.query_key_value.bias",
                f"transformer.layers.{i}.attention.dense.weight",
                f"transformer.layers.{i}.attention.dense.bias",
                f"transformer.layers.{i}.post_attention_layernorm.weight",
                f"transformer.layers.{i}.post_attention_layernorm.bias",
                f"transformer.layers.{i}.mlp.dense_h_to_4h.weight",
                f"transformer.layers.{i}.mlp.dense_h_to_4h.bias",
                f"transformer.layers.{i}.mlp.dense_4h_to_h.weight",
                f"transformer.layers.{i}.mlp.dense_4h_to_h.bias",
            ]
        weight_names += [
            "transformer.final_layernorm.weight",
            "transformer.final_layernorm.bias",
        ]
        dump_state_dict(f, weight_names, model.state_dict(), model.config.quantization_bit, ne_type)


class ChatGLM2Converter(BaseConverter):
    MODEL_TYPE = ModelType.CHATGLM2

    @staticmethod
    def dump_config(f, config, ne_type):
        assert config.add_bias_linear is False, "unimplemented: add_bias_linear must be false"
        assert config.add_qkv_bias is True, "unimplemented: add_qkv_bias must be true"
        assert (
            config.apply_residual_connection_post_layernorm is False
        ), "unimplemented: apply_residual_connection_post_layernorm must be false"
        assert (
            config.kv_channels * config.num_attention_heads == config.hidden_size
        ), "unimplemented: invalid kv_channels"
        assert config.multi_query_attention is True, "unimplemented: multi_query_attention must be true"
        assert config.original_rope is True, "unimplemented: original_rope must be true"
        assert config.post_layer_norm is True, "unimplemented: post_layer_norm must be true"
        assert config.rmsnorm is True, "unimplemented: rmsnorm must be true"

        config_values = [
            ne_type.value,
            config.padded_vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_layers,
            config.ffn_hidden_size,
            config.seq_length,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.multi_query_group_num,
        ]

        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def dump_tokenizer(f, tokenizer):
        serialized_model_proto = tokenizer.tokenizer.sp_model.serialized_model_proto()
        f.write(struct.pack("i", len(serialized_model_proto)))
        f.write(serialized_model_proto)

    @staticmethod
    def dump_model(f, model, ne_type):
        weight_names = ["transformer.embedding.word_embeddings.weight"]
        for i in range(model.config.num_layers):
            weight_names += [
                f"transformer.encoder.layers.{i}.input_layernorm.weight",
                f"transformer.encoder.layers.{i}.self_attention.query_key_value.weight",
                f"transformer.encoder.layers.{i}.self_attention.query_key_value.bias",
                f"transformer.encoder.layers.{i}.self_attention.dense.weight",
                f"transformer.encoder.layers.{i}.post_attention_layernorm.weight",
                f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight",
                f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight",
            ]
        weight_names += [
            "transformer.encoder.final_layernorm.weight",
            "transformer.output_layer.weight",
        ]
        dump_state_dict(f, weight_names, model.state_dict(), model.config.quantization_bit, ne_type)


def main():
    parser = argparse.ArgumentParser("chatglm-convert")
    parser.add_argument(
        "-i",
        "--model_name_or_path",
        default="THUDM/chatglm-6b",
        type=str,
        help="Model name or path used in AutoModel.from_pretrained",
    )
    parser.add_argument(
        "-l",
        "--lora_model_name_or_path",
        default=None,
        type=str,
        help="Lora model name or path used in PeftModel.from_pretrained",
    )
    parser.add_argument(
        "-o", "--save_path", default="chatglm-ne.bin", type=Path, help="Path to save the generated NE model"
    )
    parser.add_argument(
        "-t",
        "--type",
        default="q4_0",
        type=str,
        choices=["f32", "f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"],
        help="NE model quantization type",
    )
    args = parser.parse_args()

    ne_type = NEType[args.type.upper()]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if args.lora_model_name_or_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora_model_name_or_path)
        model = model.merge_and_unload()

    if hasattr(model.config, "multi_query_attention"):
        ChatGLM2Converter.convert(model, tokenizer, ne_type, args.save_path)
    else:
        ChatGLMConverter.convert(model, tokenizer, ne_type, args.save_path)


if __name__ == "__main__":
    main()