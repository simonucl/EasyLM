import dataclasses
import pprint
from functools import partial
import re

from tqdm import tqdm, trange
import numpy as np
import wandb

import absl.app
import absl.flags
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit, with_sharding_constraint
from jax.experimental import PartitionSpec as PS
import flax
from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from flax.training.train_state import TrainState
import optax


from ..data import PretrainDataset
from ..jax_utils import (
    JaxRNG, ShardingHelper, get_jax_mp_mesh, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, named_tree_map, global_norm,
    optax_add_scheduled_weight_decay
)
from ..utils import (
    WandBLogger, define_flags_with_default, get_user_flags, set_random_seed,
    load_pickle, load_checkpoint, user_flags_to_config_dict
)
from ..models.gptj import GPTJConfig, FlaxGPTJForCausalLMModule


FLAGS_DEF = define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    mp_mesh_dim=1,
    total_steps=10000,
    accumulate_gradient_step=1,
    lr=0.01,
    lr_warmup_steps=10000,
    opt_b1=0.9,
    opt_b2=0.99,
    clip_gradient=1.0,
    weight_decay=1e-4,
    load_hf_pretrained='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    tokenizer=GPTJConfig.get_tokenizer_config(),
    dataset=PretrainDataset.get_default_config(),
    gptj=GPTJConfig.get_default_config(),
    logger=WandBLogger.get_default_config(),
    log_all_worker=False,
)
FLAGS = absl.flags.FLAGS


def main(argv):
    FLAGS = absl.flags.FLAGS
    if FLAGS.initialize_jax_distributed:
        jax.distributed.initialize()

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    if FLAGS.load_dataset_state != '':
        dataset = load_pickle(FLAGS.load_dataset_state)['dataset']
    else:
        tokenizer = GPTJConfig.get_tokenizer(FLAGS.tokenizer)
        dataset = PretrainDataset.load_dataset(FLAGS.dataset, tokenizer)

    seq_length = dataset.seq_length
    gptj_config = GPTJConfig(**FLAGS.gptj)
    gptj_config.update(dict(
        bos_token_id=dataset.tokenizer.bos_token_id,
        eos_token_id=dataset.tokenizer.eos_token_id,
        vocab_size=dataset.vocab_size,
    ))
    model = FlaxGPTJForCausalLMModule(gptj_config)

    def weight_decay_mask(params):
        def decay(name, _):
            for rule in GPTJConfig.get_weight_decay_exclusions():
                if re.search(rule, name) is not None:
                    return False
            return True
        return named_tree_map(decay, params, sep='/')

    def learning_rate_schedule(step):
        multiplier = FLAGS.lr / 0.01
        return multiplier / jnp.sqrt(jnp.maximum(step, FLAGS.lr_warmup_steps))

    def weight_decay_schedule(step):
        multiplier = FLAGS.weight_decay / 1e-4
        return -multiplier * jnp.square(learning_rate_schedule(step))

    optimizer = optax.chain(
        optax.clip_by_global_norm(FLAGS.clip_gradient),
        optax.adafactor(
            learning_rate=learning_rate_schedule,
            multiply_by_parameter_scale=True,
            momentum=FLAGS.opt_b1,
            decay_rate=FLAGS.opt_b2,
            factored=False,
            clipping_threshold=None,
            dtype_momentum=jnp.bfloat16,
        ),
        optax_add_scheduled_weight_decay(
            weight_decay_schedule, weight_decay_mask
        )
    )

    if FLAGS.accumulate_gradient_step > 1:
        optimizer = optax.MultiSteps(optimizer, FLAGS.accumulate_gradient_step)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(gptj_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        tokens = with_sharding_constraint(batch['tokens'], PS('dp'))
        def loss_and_accuracy(params):
            bos_tokens = jnp.full(
                (tokens.shape[0], 1), gptj_config.bos_token_id, dtype=jnp.int32
            )
            inputs = jnp.concatenate([bos_tokens, tokens[:, :-1]], axis=1)
            logits = model.apply(
                params, inputs, deterministic=False,
                rngs=rng_generator(gptj_config.rng_keys()),
            ).logits
            return cross_entropy_loss_and_accuracy(logits, tokens)
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=learning_rate_schedule(train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        return train_state, rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        GPTJConfig.get_partition_rules(), train_state_shapes
    )

    sharding_helper = ShardingHelper(train_state_partition)

    sharded_init_fn = pjit(
        init_fn,
        in_axis_resources=PS(),
        out_axis_resources=train_state_partition
    )

    sharded_train_step = pjit(
        train_step,
        in_axis_resources=(train_state_partition, PS(), PS()),
        out_axis_resources=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    def save_checkpoint(train_state):
        metadata = dict(
            step=train_state.step,
            variant=variant,
            flags=flags_config_dict,
            dataset=dataset,
            gptj_config=gptj_config,
        )
        logger.save_checkpoint(train_state, metadata)
        logger.save_pickle(metadata, 'metadata.pkl')

    if FLAGS.load_checkpoint != '':
        with jax.default_device(jax.devices("cpu")[0]):
            restored_checkpoint_state, _ = load_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes
            )
            start_step = restored_checkpoint_state.step
    else:
        start_step = 0
        if FLAGS.load_hf_pretrained != '':
            hf_pretrained_params = gptj_config.load_pretrained(
                FLAGS.load_hf_pretrained
            )

    mesh = get_jax_mp_mesh(FLAGS.mp_mesh_dim)
    with mesh:
        if FLAGS.load_checkpoint != '':
            train_state = sharding_helper.put(restored_checkpoint_state)
            del restored_checkpoint_state
        else:
            train_state = sharded_init_fn(next_rng())
            if FLAGS.load_hf_pretrained != '':
                train_state = sharding_helper.get(train_state)
                train_state = train_state.replace(params=hf_pretrained_params)
                del hf_pretrained_params
                train_state = sharding_helper.put(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        for step, batch in zip(step_counter, dataset):
            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, sharded_rng, batch
            )

            if step % FLAGS.log_freq == 0:
                log_metrics = {"step": step}
                log_metrics.update(metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    absl.app.run(main)
