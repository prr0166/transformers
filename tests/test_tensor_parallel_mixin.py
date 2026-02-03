# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""Tensor parallel tester mixin for model tests."""

import os
import tempfile
from abc import ABC, abstractmethod

from transformers import set_seed
from transformers.testing_utils import (
    backend_device_count,
    get_torch_dist_unique_port,
    is_torch_available,
    torch_device,
)
from transformers.utils import is_torch_greater_or_equal


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp


def get_packed_grad_shard(grad, world_size, rank, dim):
    """Get the correct shard of a packed gradient (matching get_packed_weights interleaved logic).

    Packed weights like gate_up_proj are sharded with interleaving:
    Original: [G0 G1 G2 G3 | U0 U1 U2 U3]  (gate | up)
    Rank 0:   [G0 G1 | U0 U1]
    Rank 1:   [G2 G3 | U2 U3]
    """
    total_size = grad.shape[dim]
    # Packed weights have 2 blocks (gate and up)
    block_size = total_size // 2
    shard_block_size = block_size // world_size

    # Build interleaved indices
    indices = []
    for block_idx in range(2):  # gate block, then up block
        block_offset = block_idx * block_size
        start = block_offset + rank * shard_block_size
        stop = block_offset + (rank + 1) * shard_block_size
        indices.extend(range(start, stop))

    # Select along the sharded dimension
    return grad.index_select(dim, torch.tensor(indices, device=grad.device))


def _global_wrapper(rank, func, tp, port, func_args, func_kwargs):
    """Wrapper to set up distributed environment and run the test function."""

    def setup_dist_env(rank, world_size, port):
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

    world_size = tp
    setup_dist_env(rank, world_size, port)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    func(rank, *func_args, **func_kwargs)

    dist.barrier()
    dist.destroy_process_group()


def _init_distributed(tp: int):
    """Decorator to initialize distributed environment and spawn processes."""

    def _init_distributed_inner(func):
        def wrapper(*args, **kwargs):
            world_size = tp
            port = get_torch_dist_unique_port()
            spawn_args = (func, tp, port, args, kwargs)
            mp.spawn(_global_wrapper, args=spawn_args, nprocs=world_size)

        return wrapper

    return _init_distributed_inner


class TensorParallelTesterMixin(ABC):
    """
    Mixin for tensor parallel tests. Add to model test classes alongside ModelTesterMixin.

    The model_tester (e.g., CausalLMModelTester) already provides:
      - get_config() -> tiny model config
      - causal_lm_class, base_model_class, etc.

    This mixin adds tensor parallel-specific tests using that infrastructure.
    """

    # ============================================================
    # Configuration (can be overridden per model)
    # ============================================================
    tensor_parallel_size: int = 2
    tensor_parallel_atol: float = 1e-5
    tensor_parallel_rtol: float = 1e-5

    @property
    @abstractmethod
    def model_tester(self):
        """The model tester instance (e.g., CausalLMModelTester)."""
        ...

    # ============================================================
    # Helper methods
    # ============================================================
    def _has_tp_plan(self) -> bool:
        """Check if model has a tensor parallel plan defined."""
        config = self.model_tester.get_config()
        return hasattr(config, "base_model_tp_plan") and config.base_model_tp_plan is not None

    def _get_tp_model_class(self):
        """Get the model class to use for TP tests (prefers *ForCausalLM)."""
        # Prefer model classes with a head (for computing loss)
        if hasattr(self.model_tester, "causal_lm_class") and self.model_tester.causal_lm_class is not None:
            return self.model_tester.causal_lm_class
        # Fall back to first model class
        return self.all_model_classes[0]

    def _skip_if_not_supported(self):
        """Check and skip test if TP is not supported for this model/environment."""
        # Check PyTorch version
        if not is_torch_greater_or_equal("2.9"):
            self.skipTest("Tensor parallel tests require torch >= 2.9")

        # Check if model has TP plan
        if not self._has_tp_plan():
            self.skipTest("Model does not have a tensor parallel plan (base_model_tp_plan)")

        # Check device availability
        if backend_device_count(torch_device) < self.tensor_parallel_size:
            self.skipTest(
                f"Need at least {self.tensor_parallel_size} devices, "
                f"have {backend_device_count(torch_device)}"
            )

    # ============================================================
    # Test implementations (run inside distributed processes)
    # ============================================================
    def _test_tp_forward_impl(self, _rank, model_path, model_class, atol, rtol):
        """Implementation for comparing TP and non-TP model outputs."""
        set_seed(0)

        # Load TP model first to determine device
        model_tp = model_class.from_pretrained(model_path, tp_plan="auto")
        dist.barrier()
        model_tp.eval()

        # Load non-TP model and move to same device as TP model
        device = model_tp.device
        model = model_class.from_pretrained(model_path)
        model = model.to(device)
        model.eval()

        # Create deterministic inputs
        batch_size, seq_length = 2, 64
        vocab_size = model.config.vocab_size
        set_seed(42)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

            outputs_tp = model_tp(input_ids)
            logits_tp = outputs_tp.logits

        diff = (logits - logits_tp).abs()
        assert torch.allclose(logits, logits_tp, atol=atol, rtol=rtol), (
            f"TP and non-TP model outputs differ. "
            f"Max diff: {diff.max().item()} | Min diff: {diff.min().item()}"
        )

        dist.barrier()

    def _test_tp_backward_impl(self, rank, model_path, model_class, atol, rtol):
        """Implementation for comparing TP and non-TP model backward passes."""
        set_seed(0)

        # Load TP model first to determine device
        model_tp = model_class.from_pretrained(model_path, tp_plan="auto")
        dist.barrier()
        model_tp.train()

        # Load non-TP model and move to same device as TP model
        device = model_tp.device
        model = model_class.from_pretrained(model_path)
        model = model.to(device)
        model.train()

        # Create deterministic inputs
        batch_size, seq_length = 2, 64
        vocab_size = model.config.vocab_size
        set_seed(42)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)

        # Forward and backward for non-TP model
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        # Forward and backward for TP model
        outputs_tp = model_tp(input_ids, labels=labels)
        loss_tp = outputs_tp.loss
        loss_tp.backward()

        # Compare losses
        assert torch.allclose(loss, loss_tp, atol=atol, rtol=rtol), (
            f"TP and non-TP model losses differ. "
            f"Non-TP loss: {loss.item()}, TP loss: {loss_tp.item()}, "
            f"Diff: {(loss - loss_tp).abs().item()}"
        )

        # Compare gradients for matching parameters
        world_size = dist.get_world_size()
        for (name, param), (_, param_tp) in zip(model.named_parameters(), model_tp.named_parameters()):
            if param.grad is not None and param_tp.grad is not None:
                grad = param.grad
                grad_tp = param_tp.grad

                # Slice reference gradient to match local shard if parameter is sharded
                if grad.shape != grad_tp.shape:
                    for dim in range(grad.ndim):
                        if grad.size(dim) != grad_tp.size(dim):
                            # Packed weights (gate_up_proj) use interleaved sharding
                            if "gate_up_proj" in name:
                                grad = get_packed_grad_shard(grad, world_size, rank, dim)
                            else:
                                # Regular weights use simple chunking
                                shard_size = grad_tp.size(dim)
                                start = rank * shard_size
                                grad = grad.narrow(dim, start, shard_size)
                            break

                assert torch.allclose(grad.cpu(), grad_tp.cpu(), atol=atol, rtol=rtol), (
                    f"Gradients differ for parameter {name}. "
                    f"Max diff: {(grad.cpu() - grad_tp.cpu()).abs().max().item()}"
                )

        dist.barrier()

    def _test_tp_generation_impl(self, _rank, model_path, model_class, atol, rtol, max_new_tokens):
        """Implementation for comparing TP and non-TP model generation outputs."""
        set_seed(0)

        # Load TP model first to determine device
        model_tp = model_class.from_pretrained(model_path, tp_plan="auto")
        dist.barrier()
        model_tp.eval()

        # Load non-TP model and move to same device as TP model
        device = model_tp.device
        model = model_class.from_pretrained(model_path)
        model = model.to(device)
        model.eval()

        # Create deterministic inputs (short prompt for generation)
        batch_size, seq_length = 1, 10
        vocab_size = model.config.vocab_size
        set_seed(42)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)

        # Generation kwargs for greedy decoding with logit output
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "output_scores": True,
            "return_dict_in_generate": True,
            "use_cache": True,
        }

        with torch.no_grad():
            # Generate with non-TP model
            output = model.generate(input_ids, **generation_kwargs)

            # Generate with TP model
            output_tp = model_tp.generate(input_ids, **generation_kwargs)

        # Compare generated sequences
        sequences_match = torch.equal(output.sequences, output_tp.sequences)

        # Compare logits/scores at each generation step
        scores = torch.stack(output.scores)  # (max_new_tokens, batch, vocab)
        scores_tp = torch.stack(output_tp.scores)

        diff = (scores - scores_tp).abs()
        logits_match = torch.allclose(scores, scores_tp, atol=atol, rtol=rtol)

        assert logits_match, (
            f"TP and non-TP model generation logits differ. "
            f"Max diff: {diff.max().item()} | Mean diff: {diff.mean().item()}"
        )

        # If logits match but sequences don't, that's unexpected
        if not sequences_match and logits_match:
            # This shouldn't happen with greedy decoding if logits match
            pass  # Log warning but don't fail since logits match

        dist.barrier()

    # ============================================================
    # Public test methods
    # ============================================================
    def test_tensor_parallel_forward(self):
        """Test that TP and non-TP models produce the same outputs."""
        self._skip_if_not_supported()

        config = self.model_tester.get_config()
        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        # Save model to temp directory so we can load it with from_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save a model with the test config
            model = model_class(config)
            model.save_pretrained(tmp_dir)

            _init_distributed(tp=self.tensor_parallel_size)(self._test_tp_forward_impl)(
                tmp_dir, model_class, atol, rtol
            )

    def test_tensor_parallel_backward(self):
        """Test that TP and non-TP models produce the same gradients."""
        self._skip_if_not_supported()

        config = self.model_tester.get_config()
        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        # Save model to temp directory so we can load it with from_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save a model with the test config
            model = model_class(config)
            model.save_pretrained(tmp_dir)

            _init_distributed(tp=self.tensor_parallel_size)(self._test_tp_backward_impl)(
                tmp_dir, model_class, atol, rtol
            )

    def test_tensor_parallel_generation(self):
        """Test that TP and non-TP models produce the same generation logits."""
        self._skip_if_not_supported()

        config = self.model_tester.get_config()
        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol
        max_new_tokens = 10  # Keep short for test speed

        # Save model to temp directory so we can load it with from_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save a model with the test config
            model = model_class(config)
            model.save_pretrained(tmp_dir)

            _init_distributed(tp=self.tensor_parallel_size)(self._test_tp_generation_impl)(
                tmp_dir, model_class, atol, rtol, max_new_tokens
            )
