# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import logging
import math
import os
import sys
from argparse import Namespace
import numpy as np
from itertools import chain
import torch

from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.data import encoders
from omegaconf import DictConfig

def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger('fairseq_cli.predict')

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)
    task.load_dataset(cfg.dataset.gen_subset)

    # Set dictionaries
    src_dict = getattr(task, 'source_dictionary', None)
    tag_dict = task.tag_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Optimize ensemble for generation
    for model in models:
        model.prepare_for_inference_(cfg)
        if cfg.common.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(cfg.tokenizer)
    bpe = encoders.build_bpe(cfg.bpe)

    def decode_fn(x):  # decode tag
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    num_sentences = 0
    wps_meter = TimeMeter()
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue

        gen_timer.start()
        hypos = task.inference_step(models, sample, cfg.task.tagging_head_name)
        num_generated_tokens = sample['ntokens']
        gen_timer.stop(num_generated_tokens)

        for i, sample_id in enumerate(sample['id'].tolist()):
            hypo = hypos[i]

            hypo_tokens = np.array(
                hypo) + tag_dict.nspecial  # can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first
            hypo_str = tag_dict.string(hypo_tokens)
            has_target = sample['target'] is not None

            # Remove padding
            if 'src_tokens' in sample['net_input']:
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], src_dict.pad())
                src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                assert len(hypo) == src_tokens.numel()

            if has_target:
                tag_offset = 1
                tag_tokens = utils.strip_pad(sample['target'][i, :],
                                             src_dict.pad()).int().cpu() - tag_offset + tag_dict.nspecial
                tag_str = tag_dict.string(tag_tokens)

            src_str = decode_fn(src_str)
            tag_str = decode_fn(tag_str)

            if not cfg.common_eval.quiet:
                if src_dict is not None:
                    print('S-{}\t{}'.format(sample_id, src_str), file=output_file)
                if has_target:
                    print('T-{}\t{}'.format(sample_id, tag_str), file=output_file)
                print('H-{}\t{}'.format(sample_id, hypo_str), file=output_file)

        wps_meter.update(num_generated_tokens)
        progress.log({'wps': round(wps_meter.avg)})
        num_sentences += sample["nsentences"] if "nsentences" in sample else sample['id'].numel()

    logger.info('NOTE: hypothesis and token scores are output in base 2')
    logger.info('Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
