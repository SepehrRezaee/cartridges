#!/usr/bin/env python3
"""Compose multiple HuggingFace cartridges in a single Tokasaurus request."""

from __future__ import annotations

import argparse
import asyncio

from cartridges.clients.tokasaurus import TokasaurusClient

from examples.hf_cartridge_utils import build_cartridge_payload, resolve_cartridge_ids


async def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate two or more cartridges (Tokasaurus handles the actual fusion)."
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
        help="Number of cartridges to pull from the collection when --cartridge-ids is omitted.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Answer using every cartridge I attached. "
            "Highlight which cartridge supplied each fact."
        ),
        help="User prompt sent to Tokasaurus.",
    )
    parser.add_argument(
        "--tokasaurus-url",
        default="http://localhost:10210",
        help="Tokasaurus base URL.",
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model name the Tokasaurus server hosts.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=384,
        help="Maximum generation length.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force the server to redownload cartridges.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode for supported chat templates.",
    )
    parser.add_argument(
        "--show-progress-bar",
        action="store_true",
        help="Render tqdm progress for the batch request.",
    )
    args = parser.parse_args()

    cartridge_ids = resolve_cartridge_ids(
        args.collection_slug,
        args.num_cartridges,
        args.cartridge_ids,
    )
    payloads = build_cartridge_payload(
        cartridge_ids,
        force_redownload=args.force_redownload,
    )

    client = TokasaurusClient.Config(
        model_name=args.model_name,
        url=args.tokasaurus_url,
        show_progress_bar=args.show_progress_bar,
    ).instantiate()

    response = await client.chat(
        chats=[[{"role": "user", "content": args.prompt}]],
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        cartridges=payloads,
        enable_thinking=args.enable_thinking,
    )

    print(f"Attached cartridges: {', '.join(cartridge_ids)}\n")
    print(response.samples[0].text.strip() or "[empty response]")


if __name__ == "__main__":
    asyncio.run(_main())

