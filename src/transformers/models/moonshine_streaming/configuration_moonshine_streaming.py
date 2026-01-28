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

from typing import Optional

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ..auto import CONFIG_MAPPING


class MoonshineStreamingEncoderConfig(PreTrainedConfig):
    model_type = "moonshine_streaming_encoder"

    def __init__(
        self,
        hidden_size: Optional[int] = 320,
        intermediate_size: Optional[int] = 1280,
        hidden_act: Optional[str] = "gelu",
        num_hidden_layers: Optional[int] = 6,
        num_attention_heads: Optional[int] = 8,
        num_key_value_heads: Optional[int] = 8,
        max_position_embeddings: Optional[int] = 4096,
        attention_dropout: Optional[float] = 0.0,
        attention_bias: Optional[bool] = False,
        sample_rate: int = 16000,
        frame_ms: float = 5.0,
        sliding_windows: list[tuple[int, int]] = [(16, 4), (16, 4), (16, 0), (16, 0), (16, 4), (16, 4)],
        head_dim: Optional[int] = None,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.sliding_windows = [list(window) for window in sliding_windows]

        super().__init__(**kwargs)


class MoonshineStreamingConfig(PreTrainedConfig):
    model_type = "moonshine_streaming"
    sub_configs = {"encoder_config": MoonshineStreamingEncoderConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        encoder_config: MoonshineStreamingEncoderConfig = None,
        vocab_size: int = 32768,
        hidden_size: Optional[int] = 320,
        encoder_hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = 1280,
        num_hidden_layers: Optional[int] = 6,
        num_attention_heads: Optional[int] = 8,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: int = 4096,
        use_cache: Optional[bool] = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.8,
        },
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        decoder_start_token_id: Optional[int] = None,
        head_dim: Optional[int] = None,
        pad_head_dim_to_multiple_of: Optional[int] = None,
        **kwargs,
    ):
        encoder_config_is_none = encoder_config is None
        if isinstance(encoder_config, dict):
            encoder_config["model_type"] = encoder_config.get("model_type", "moonshine_streaming_encoder")
            encoder_config = CONFIG_MAPPING[encoder_config["model_type"]](**encoder_config)
        elif encoder_config is None:
            encoder_config = CONFIG_MAPPING["moonshine_streaming_encoder"]()

        self.encoder_config = encoder_config
        if encoder_hidden_size is None:
            encoder_hidden_size = hidden_size if encoder_config_is_none else self.encoder_config.hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_config.hidden_size = encoder_hidden_size

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.rope_parameters = rope_parameters
        self.pad_head_dim_to_multiple_of = pad_head_dim_to_multiple_of

        kwargs.update(tie_word_embeddings=False, is_encoder_decoder=True)
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )


__all__ = ["MoonshineStreamingConfig", "MoonshineStreamingEncoderConfig"]
