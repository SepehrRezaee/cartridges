#!/usr/bin/env python3
"""Benchmark composed HuggingFace cartridges on LongHealth MC questions."""

from __future__ import annotations

import argparse
from typing import List, Optional

from cartridges.clients.base import CartridgeConfig
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.evaluate import GenerationEvalConfig, GenerationEvalRunConfig, ICLBaseline

from examples.hf_cartridge_utils import resolve_cartridge_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate concatenated HuggingFace cartridges on LongHealth."
    )
    parser.add_argument(
        "--collection-slug",
        default="hazyresearch/cartridges",
        help="Slug from https://huggingface.co/collections/<slug>",
    )
    parser.add_argument(
        "--cartridge-ids",
        nargs="+",
        default=None,
        help="Explicit cartridge repo IDs.",
    )
    parser.add_argument(
        "--num-cartridges",
        type=int,
        default=2,
        help="Number of cartridges to pull from the collection when not provided explicitly.",
    )
    parser.add_argument(
        "--tokasaurus-url",
        default="http://localhost:10210",
        help="Tokasaurus base URL.",
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model name hosted by the Tokasaurus server.",
    )
    parser.add_argument(
        "--tokenizer",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Tokenizer used for counting tokens during evaluation.",
    )
    parser.add_argument(
        "--patient-ids",
        nargs="+",
        default=None,
        help="Optional subset of LongHealth patient IDs.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=128,
        help="Ceiling on the number of eval questions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Total batch size (per eval step).",
    )
    parser.add_argument(
        "--max-parallel-batches",
        type=int,
        default=2,
        help="Number of concurrent async batches.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Samples per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=256,
        help="Max assistant tokens per completion.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force Tokasaurus to redownload cartridges.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking tokens for thinking-capable templates.",
    )
    parser.add_argument(
        "--run-name",
        default="hf_cartridge_longhealth",
        help="Human-readable name for the evaluation run.",
    )
    parser.add_argument(
        "--eval-name",
        default="longhealth_mc",
        help="Name passed to GenerationEvalConfig.",
    )
    return parser.parse_args()


def _build_cartridge_configs(ids: List[str], force_redownload: bool) -> List[CartridgeConfig]:
    return [
        CartridgeConfig(id=repo_id, source="huggingface", force_redownload=force_redownload)
        for repo_id in ids
    ]


def main() -> None:
    args = _parse_args()
    cartridge_ids = resolve_cartridge_ids(
        args.collection_slug,
        args.num_cartridges,
        args.cartridge_ids,
    )
    cartridge_cfgs = _build_cartridge_configs(cartridge_ids, args.force_redownload)

    generator_cfg = ICLBaseline.Config(
        client=TokasaurusClient.Config(
            model_name=args.model_name,
            url=args.tokasaurus_url,
            cartridges=cartridge_cfgs,
            show_progress_bar=True,
        ),
        tokenizer=args.tokenizer,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        context="Cartridge-augmented inference",
        enable_thinking=args.enable_thinking,
    )

    eval_cfg = GenerationEvalConfig(
        dataset=LongHealthMultipleChoiceGenerateDataset.Config(
            patient_ids=args.patient_ids,
            max_questions=args.max_questions,
        ),
        num_samples=args.num_samples,
        temperature=args.temperature,
        name_for_wandb=args.eval_name,
    )

    run_cfg = GenerationEvalRunConfig(
        eval=eval_cfg,
        generator=generator_cfg,
        batch_size=args.batch_size,
        max_num_batches_in_parallel=args.max_parallel_batches,
        tokenizer=args.tokenizer,
        name=args.run_name,
    )

    print(f"Evaluating cartridges: {', '.join(cartridge_ids)}")
    run_cfg.run()


if __name__ == "__main__":
    main()

