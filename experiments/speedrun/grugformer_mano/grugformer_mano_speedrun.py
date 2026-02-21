# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Grugformer + MANO starter speedrun.

This starts from the grugformer starter and swaps in a custom `ManoConfig`
optimizer so you can iterate on Grugformer + MANO ideas in one file.

By default, this runs the 130m preset. Set `SR_RUN_ALL_SIZES=1` to run all
scale points.
"""

# nodryrun

import os
from dataclasses import dataclass, replace

import optax
from fray.cluster import ResourceConfig
from haliax import Axis
from jaxtyping import PRNGKeyArray

from levanter.grug.model import GrugModelConfig
from levanter.models.grug_wrapper import GrugWrapper
from levanter.models.lm_model import LmConfig
from levanter.optim import OptimizerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

from experiments.llama import llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.prebuilt_caches import fineweb_edu_subcache_10M


def _get_num_train_steps(param_count: int, batch_size: int, max_seq_len: int, tpp: int = 20) -> int:
    total_tokens = param_count * tpp
    return max(1, total_tokens // (batch_size * max_seq_len))


@OptimizerConfig.register_subclass("mano")
@dataclass(frozen=True)
class ManoConfig(OptimizerConfig):
    """
    Initial MANO optimizer config for submission bootstrapping.

    This implementation uses Adam-style adaptive moments plus RMS update
    clipping as a practical MANO baseline you can tune from.
    """

    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
    update_rms_clip: float | None = 1.0

    def build(self, num_train_steps):
        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))

            if self.update_rms_clip:
                components.append(optax.clip_by_block_rms(self.update_rms_clip))

            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))

            components.append(optax.scale(-learning_rate))
            return optax.chain(*components)

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))


def _size_presets() -> dict[str, "GrugformerConfig"]:
    base = dict(max_seq_len=int(os.environ.get("SR_SEQ_LEN", "2048")), head_dim=None)
    return {
        "30m": GrugformerConfig(
            hidden_dim=128, intermediate_dim=448, num_layers=4, num_heads=2, num_kv_heads=2, **base
        ),
        "130m": GrugformerConfig(
            hidden_dim=512, intermediate_dim=1792, num_layers=6, num_heads=8, num_kv_heads=8, **base
        ),
        "300m": GrugformerConfig(
            hidden_dim=768, intermediate_dim=2688, num_layers=12, num_heads=12, num_kv_heads=12, **base
        ),
        "520m": GrugformerConfig(
            hidden_dim=1024, intermediate_dim=3584, num_layers=24, num_heads=16, num_kv_heads=16, **base
        ),
        "1_2b": GrugformerConfig(
            hidden_dim=2048, intermediate_dim=7168, num_layers=16, num_heads=16, num_kv_heads=16, **base
        ),
    }


def _mano_presets() -> dict[str, ManoConfig]:
    return {
        "30m": ManoConfig(learning_rate=3e-3, weight_decay=0.1, beta1=0.9, beta2=0.95, max_grad_norm=1.0),
        "130m": ManoConfig(learning_rate=3e-3, weight_decay=0.1, beta1=0.9, beta2=0.95, max_grad_norm=1.0),
        "300m": ManoConfig(learning_rate=2e-3, weight_decay=0.1, beta1=0.9, beta2=0.95, max_grad_norm=1.0),
        "520m": ManoConfig(learning_rate=1.5e-3, weight_decay=0.1, beta1=0.9, beta2=0.95, max_grad_norm=1.0),
        "1_2b": ManoConfig(learning_rate=1e-3, weight_decay=0.1, beta1=0.9, beta2=0.95, max_grad_norm=1.0),
    }


def _resource_presets(use_tpu: bool = False):
    if use_tpu:
        tpu_type = os.environ.get("SR_TPU_TYPE", "v5p-8")
        return {
            "30m": ResourceConfig.with_tpu(tpu_type),
            "130m": ResourceConfig.with_tpu(tpu_type),
            "300m": ResourceConfig.with_tpu(tpu_type),
            "520m": ResourceConfig.with_tpu(tpu_type),
            "1_2b": ResourceConfig.with_tpu(tpu_type),
        }

    gpu_count = int(os.environ.get("SR_GPU_COUNT", "1"))
    return {
        "30m": ResourceConfig.with_gpu("auto", count=gpu_count),
        "130m": ResourceConfig.with_gpu("auto", count=gpu_count),
        "300m": ResourceConfig.with_gpu("auto", count=gpu_count),
        "520m": ResourceConfig.with_gpu("auto", count=max(2, gpu_count)),
        "1_2b": ResourceConfig.with_gpu("auto", count=max(4, gpu_count)),
    }


def _batch_sizes() -> dict[str, int]:
    default_batch = int(os.environ.get("SR_BATCH_SIZE", "128"))
    return {
        "30m": int(os.environ.get("SR_BATCH_SIZE_30M", str(default_batch))),
        "130m": int(os.environ.get("SR_BATCH_SIZE_130M", str(default_batch))),
        "300m": int(os.environ.get("SR_BATCH_SIZE_300M", str(default_batch))),
        "520m": int(os.environ.get("SR_BATCH_SIZE_520M", str(default_batch))),
        "1_2b": int(os.environ.get("SR_BATCH_SIZE_1_2B", str(max(256, default_batch)))),
    }


@LmConfig.register_subclass("grugformer_mano")
@dataclass(frozen=True)
class GrugformerConfig(LmConfig[GrugWrapper]):
    max_seq_len: int = 2048
    hidden_dim: int = 1024
    intermediate_dim: int = 2752
    num_layers: int = 12
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int | None = None

    @property
    def model_type(self) -> type[GrugWrapper]:
        return GrugWrapper

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    def build(self, Vocab: Axis, *, key: PRNGKeyArray) -> GrugWrapper:
        grug_cfg = GrugModelConfig(
            vocab_size=Vocab.size,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
        )
        return GrugWrapper.init(Vocab, grug_cfg, key=key)

    def flops_per_token(self, vocab_size: int, context_length: int) -> float | None:
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=context_length,
            vocab_size=vocab_size,
            glu=True,
        )

    def total_trainable_params(self, vocab_size: int) -> int:
        head_dim = self.head_dim or (self.hidden_dim // self.num_heads)
        token_embedding = vocab_size * self.hidden_dim
        attn = (
            self.hidden_dim * head_dim * self.num_heads
            + 2 * self.hidden_dim * head_dim * self.num_kv_heads
            + head_dim * self.num_heads * self.hidden_dim
        )
        mlp = 3 * self.hidden_dim * self.intermediate_dim
        transformer = self.num_layers * (attn + mlp + 2 * self.hidden_dim) + self.hidden_dim
        return int(transformer + 2 * token_embedding)


def build_run(size: str, *, use_tpu: bool = False) -> tuple[str, SpeedrunConfig]:
    sizes = _size_presets()
    if size not in sizes:
        raise ValueError(f"Unknown size: {size}")

    model_cfg = sizes[size]
    batch = _batch_sizes()[size]
    max_seq_len = model_cfg.max_seq_len
    params = int(model_cfg.total_trainable_params(llama3_tokenizer_vocab_size))
    steps = int(os.environ.get("SR_NUM_STEPS", _get_num_train_steps(params, batch, max_seq_len, tpp=20)))

    mano_cfg = _mano_presets()[size]
    resources = _resource_presets(use_tpu=use_tpu)[size]
    max_eval_batches_env = os.environ.get("SR_MAX_EVAL_BATCHES")
    max_eval_batches = None if max_eval_batches_env is None else int(max_eval_batches_env)

    train = SimpleTrainConfig(
        resources=resources,
        train_seq_len=max_seq_len,
        train_batch_size=batch,
        num_train_steps=steps,
        learning_rate=mano_cfg.learning_rate,
        weight_decay=mano_cfg.weight_decay,
        steps_per_eval=500,
        max_eval_batches=max_eval_batches,
        steps_per_hf_export=-1,
        # Grug uses explicit PartitionSpec("data", "model") sharding in init.
        explicit_mesh_axes=True,
        optimizer_config=mano_cfg,
    )

    run_suffix = os.environ.get("SR_RUN_NAME_SUFFIX", "").strip()
    run_name = f"grugformer_mano_{size}"
    if run_suffix:
        run_name = f"{run_name}_{run_suffix}"
    desc = f"Grugformer + MANO starter ({size})."
    use_10m_data = bool(int(os.environ.get("SR_USE_10M_DATASET", "0")))
    cfg_kwargs = {}
    if use_10m_data:
        cfg_kwargs["tokenized_dataset"] = fineweb_edu_subcache_10M

    cfg = SpeedrunConfig(
        author=Author(
            name="Suraj Ranganath",
            affiliation="UC San Diego",
            url="https://www.linkedin.com/in/suraj-ranganath/",
        ),
        description=desc,
        model_config=model_cfg,
        train_config=train,
        **cfg_kwargs,
    )
    return run_name, cfg


def _patch_gpu_mesh_batch_axis(step):
    """Align batch sharding with Grug's explicit data/model sharding on GPU."""
    cfg = getattr(step, "config", None)
    train_cfg = getattr(cfg, "train_config", None)
    trainer_cfg = getattr(train_cfg, "trainer", None)
    mesh_cfg = getattr(trainer_cfg, "mesh", None)
    if mesh_cfg is None:
        return step

    compute_mapping = dict(mesh_cfg.compute_mapping)
    if compute_mapping.get("batch") == "data":
        return step

    compute_mapping["batch"] = "data"
    patched_mesh = replace(mesh_cfg, compute_mapping=compute_mapping)
    patched_trainer = replace(trainer_cfg, mesh=patched_mesh)
    patched_train_cfg = replace(train_cfg, trainer=patched_trainer)
    patched_cfg = replace(cfg, train_config=patched_train_cfg)
    return replace(step, config=patched_cfg)


def main() -> None:
    use_tpu = bool(int(os.environ.get("SR_USE_TPU", "0")))
    run_all = bool(int(os.environ.get("SR_RUN_ALL_SIZES", "0")))
    sizes_env = os.environ.get("SR_SIZES") or os.environ.get("SR_SIZE")
    if sizes_env:
        sizes = [s.strip() for s in sizes_env.split(",") if s.strip()]
    else:
        sizes = ["30m", "130m", "300m", "520m", "1_2b"] if run_all else ["130m"]

    steps = []
    for s in sizes:
        name, cfg = build_run(s, use_tpu=use_tpu)
        if cfg.vocab_size != llama3_tokenizer_vocab_size:
            raise AssertionError("Speedrun vocab_size mismatch; expected llama3_tokenizer_vocab_size")
        cfg.print_run_info()
        generated = list(default_speedrun(name, cfg))
        if not use_tpu and bool(int(os.environ.get("SR_FORCE_BATCH_AXIS_DATA", "1"))):
            generated = [_patch_gpu_mesh_batch_axis(step) for step in generated]
        steps.extend(generated)

    executor_main(steps=steps, description="Grugformer + MANO starter")


if __name__ == "__main__":
    main()
