#!/usr/bin/env python3
"""Smoke-test individual HuggingFace cartridges via Tokasaurus."""

from __future__ import annotations

import argparse
import asyncio
from typing import List

from cartridges.clients.tokasaurus import TokasaurusClient

from examples.hf_cartridge_utils import build_cartridge_payload, resolve_cartridge_ids


async def _ping_cartridge(
    client: TokasaurusClient,
    prompt: str,
    cartridge: dict,
    *,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> str:
    """Send a single prompt with the provided cartridge."""
    response = await client.chat(
        chats=[[{"role": "user", "content": prompt.format(cartridge=cartridge["id"])}]],
        max_completion_tokens=max_tokens,
        temperature=temperature,
        cartridges=[cartridge],
        enable_thinking=enable_thinking,
    )
    return response.samples[0].text.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download two cartridges from a HuggingFace collection and query each individually."
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
        help="Explicit cartridge repo IDs to use instead of inferring from the collection.",
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
        "--prompt",
        default="Summarize the most critical context stored inside {cartridge}.",
        help="Prompt template (can reference {cartridge}).",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=256,
        help="Maximum new tokens per request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--num-cartridges",
        type=int,
        default=2,
        help="How many cartridge IDs to pull from the collection.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force the server to redownload cartridges even if cached.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Request thinking-enabled prompts for models that support it.",
    )
    parser.add_argument(
        "--show-progress-bar",
        action="store_true",
        help="Ask the Tokasaurus client to render tqdm progress.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    cartridge_ids: List[str] = resolve_cartridge_ids(
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

    print(f"Resolved cartridges: {', '.join(cartridge_ids)}\n")

    for idx, cartridge in enumerate(payloads, start=1):
        text = await _ping_cartridge(
            client,
            args.prompt,
            cartridge,
            max_tokens=args.max_completion_tokens,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
        )
        print(f"=== Cartridge {idx}: {cartridge['id']} ===")
        print(text or "[empty response]")
        print()


if __name__ == "__main__":
    asyncio.run(_main())
