# coding=utf-8
# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from transformers.utils.generic import check_model_inputs

from ...cache_utils import Cache, EncoderDecoderCache
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqModelOutput,
)
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple
from ..bert.modeling_bert import BertSelfAttention
from ..llama.modeling_llama import LlamaMLP
from ..moonshine.modeling_moonshine import (
    MoonshineDecoder,
    MoonshineEncoderLayer,
    MoonshineEncoderMLP,
    MoonshineForConditionalGeneration,
    MoonshineModel,
    MoonshinePreTrainedModel,
)
from .configuration_moonshine_streaming import MoonshineStreamingConfig


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    Extends [~modeling_outputs.BaseModelOutput] to include the output attention mask since sequence length is not preserved in the model's forward.
    """
)
class MoonshineStreamingEncoderModelOutput(BaseModelOutput):
    attention_mask: Optional[torch.Tensor] = None


class MoonshineStreamingFrameCMVN(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        rms = (centered.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return centered / rms


class MoonshineStreamingAsinhCompression(nn.Module):
    def __init__(self, k_init: float = 0.75):
        super().__init__()
        self.log_k = nn.Parameter(torch.log(torch.tensor(k_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.asinh(torch.exp(self.log_k) * x)


class MoonshineStreamingCausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = nn.functional.pad(x, (self.left_pad, 0))
        x = super().forward(x)

        if mask is not None:
            mask = nn.functional.pad(mask, (self.left_pad, 0))
            weight = torch.ones(1, 1, self.kernel_size, device=mask.device)
            mask = nn.functional.conv1d(mask, weight, stride=self.stride)
            mask = (mask > 0).squeeze(1)
            x *= mask

        return x, mask


class MoonshineStreamingLayerNorm(nn.Module):
    def __init__(self, dim: int, unit_offset: bool = True, device=None, dtype=None):
        super().__init__()
        self.unit_offset = float(unit_offset)
        self.ln = nn.LayerNorm(dim, elementwise_affine=False, device=device, dtype=dtype)
        self.gamma = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        nn.init.constant_(self.gamma, 1.0 - self.unit_offset)

    def forward(self, x: Tensor) -> Tensor:
        normed = self.ln(x)
        gamma = self.gamma + self.unit_offset
        return normed * gamma


class MoonshineStreamingEncoderMLP(MoonshineEncoderMLP): ...


class MoonshineStreamingEncoderAttention(BertSelfAttention): ...


class MoonshineStreamingEncoderLayer(MoonshineEncoderLayer):
    def __init__(self, config: MoonshineStreamingConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MoonshineStreamingEncoderAttention(config, layer_idx)
        self.mlp = MoonshineStreamingEncoderMLP(config)
        self.input_layernorm = MoonshineStreamingLayerNorm(config.encoder_hidden_size)
        self.post_attention_layernorm = MoonshineStreamingLayerNorm(config.encoder_hidden_size, layer_idx=layer_idx)


class MoonshineStreamingEncoderEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cmvn = MoonshineStreamingFrameCMVN()
        self.comp = MoonshineStreamingAsinhCompression()
        self.conv1 = MoonshineStreamingCausalConv1d(
            config.encoder_hidden_size, config.encoder_hidden_size * 2, kernel=5, stride=2
        )
        self.conv2 = MoonshineStreamingCausalConv1d(
            config.encoder_hidden_size * 2, config.encoder_hidden_size, kernel=5, stride=2
        )
        self.frame_len = int(round(config.sample_rate * config.frame_ms / 1000.0))
        self.linear = nn.Linear(self.frame_len, config.encoder_hidden_size, bias=False)

    def forward(self, input_values, padding_mask=None):
        hidden_states = self.cmvn(input_values.reshape(input_values.shape[0], -1, self.frame_len))
        hidden_states = self.comp(hidden_states)
        hidden_states = nn.functional.silu(self.linear(hidden_states))

        if padding_mask is not None:
            num_frames = padding_mask.sum(-1) // self.frame_len
            padding_mask = (
                torch.arange(padding_mask.shape[-1], device=padding_mask.device)[None, :] < num_frames[:, None]
            )
            hidden_states *= padding_mask

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states, padding_mask = self.conv1(hidden_states, padding_mask)
        hidden_states = nn.functional.silu(hidden_states)

        hidden_states, padding_mask = self.conv2(hidden_states, padding_mask)
        hidden_states = nn.functional.silu(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states, padding_mask


class MoonshineStreamingPreTrainedModel(MoonshinePreTrainedModel):
    supports_gradient_checkpointing = False  # TODO: check


def sliding_window_mask_function(sliding_window: tuple[int, int], is_causal=True) -> Callable:
    """
    This creates uni/bidirectional attention mask with sliding window.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        left_window_size, right_window_size = sliding_window

        dist = q_idx - kv_idx
        left_mask = (dist >= 0) & (dist < left_window_size)
        right_mask = (dist < 0) & (-dist < right_window_size)
        return left_mask | right_mask

    return inner_mask


class MoonshineStreamingEncoder(MoonshineStreamingPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embedder = MoonshineStreamingEncoderEmbedder(config)
        self.layers = nn.ModuleList(
            [MoonshineStreamingEncoderLayer(config, idx) for idx in range(config.encoder_num_hidden_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.encoder_hidden_size, bias=False)
        self.gradient_checkpointing = False

        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_values: torch.FloatTensor,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, audio_length)`):
                Float values of the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a
                `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or
                the soundfile library (`pip install soundfile`). To prepare the array into
                `input_values`, the [`AutoFeatureExtractor`] should be used for padding
                and conversion into a tensor of type `torch.FloatTensor`.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding indices in `input_values`. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
        """
        inputs_embeds, attention_mask = self.embedder(input_values, padding_mask=padding_mask)

        if attention_mask is not None:
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }
            per_layer_attention_mask = [
                create_bidirectional_mask(
                    and_mask_function=sliding_window_mask_function(self.config.sliding_windows[layer_idx]),
                    **mask_kwargs,
                )
                for layer_idx in range(self.config.encoder_num_hidden_layers)
            ]

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=per_layer_attention_mask[encoder_layer.layer_idx],
                **kwargs,
            )

        hidden_states = self.layer_norm(hidden_states)

        return MoonshineStreamingEncoderModelOutput(last_hidden_state=hidden_states, attention_mask=attention_mask)


class MoonshinMoonshineStreamingDecoderMLP(LlamaMLP): ...


class MoonshineStreamingDecoder(MoonshineDecoder):
    def __init__(self, config):
        super().__init__(config)
        self.pos_emb = nn.Embedding(config.adapter_max_positions, config.encoder_dim)

        if config.encoder_dim != config.decoder_dim:
            self.proj = nn.Linear(config.encoder_dim, config.decoder_dim, bias=False)
        else:
            self.proj = nn.Identity()

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        position_embeddings = self.pos_emb(torch.arange(encoder_hidden_states.shape[1], device=input_ids.device))
        encoder_hidden_states += position_embeddings
        encoder_hidden_states = self.proj(encoder_hidden_states)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs,
        )


class MoonshineStreamingModel(MoonshineModel):
    def _mask_input_features(self):
        raise AttributeError("Not needed for MoonshineStreaming")

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        padding_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        decoder_inputs_embeds: Optional[tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[tuple[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Seq2SeqModelOutput:
        if encoder_outputs is None:
            encoder_outputs: BaseModelOutput = self.encoder(input_values, padding_mask=padding_mask, **kwargs)

        decoder_outputs: BaseModelOutputWithPastAndCrossAttentions = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_outputs.attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class MoonshineStreamingForConditionalGeneration(MoonshineForConditionalGeneration):
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        """
        Computes the output length of the convolutional layers for MoonshineStreaming.
        Different from Moonshine due to frame-based preprocessing with causal convolutions.
        """
        frame_len = int(round(self.config.sample_rate * self.config.frame_ms / 1000.0))
        output_lengths = input_lengths // frame_len
        output_lengths = (output_lengths - 1) // 2 + 1
        output_lengths = (output_lengths - 1) // 2 + 1
        return output_lengths

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str],
        generation_config,
    ):
        del model_input_name
        attention_mask = model_kwargs.get("attention_mask", None)

        # Pass raw audio directly to encoder (preprocessing is now handled internally)
        encoder_outputs = self.model.encoder(
            input_values=inputs_tensor,
            attention_mask=attention_mask,
            output_attentions=generation_config.output_attentions,
            output_hidden_states=generation_config.output_hidden_states,
            return_dict=True,
        )

        # Compute encoder attention mask from input lengths
        if attention_mask is not None:
            lengths = attention_mask.sum(-1).to(dtype=torch.long)
            encoder_lengths = self._get_feat_extract_output_lengths(lengths)
            seq_len = encoder_outputs.last_hidden_state.shape[1]
            encoder_attention_mask = torch.arange(
                seq_len, device=encoder_outputs.last_hidden_state.device
            ) < encoder_lengths.unsqueeze(1)
        else:
            encoder_attention_mask = None

        model_kwargs["encoder_outputs"] = encoder_outputs
        model_kwargs["encoder_attention_mask"] = encoder_attention_mask
        model_kwargs["attention_mask"] = encoder_attention_mask
        return model_kwargs


__all__ = [
    "MoonshineStreamingPreTrainedModel",
    "MoonshineStreamingModel",
    "MoonshineStreamingForConditionalGeneration",
]
