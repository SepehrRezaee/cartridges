#!/usr/bin/env python3
"""Utilities for discovering and packaging cartridge IDs from HuggingFace."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Iterable, List, Sequence

import requests


logger = logging.getLogger(__name__)

HF_COLLECTION_ENDPOINT = "https://huggingface.co/api/collections"


@dataclass(frozen=True)
class CollectionItem:
    """Lightweight view on the HuggingFace collection payload."""

    repo_id: str
    repo_type: str

    @classmethod
    def from_dict(cls, payload: dict) -> "CollectionItem":
        return cls(
            repo_id=payload["id"],
            repo_type=payload.get("repoType", payload.get("type", "")),
        )


def _fetch_collection_payload(slug: str) -> dict:
    """Download the collection JSON from HuggingFace."""
    url = f"{HF_COLLECTION_ENDPOINT}/{slug}"
    response = requests.get(url, timeout=30)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Failed to fetch collection '{slug}': {exc}") from exc
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Collection response for '{slug}' is not valid JSON") from exc


def _candidates_from_items(items: Iterable[dict]) -> List[CollectionItem]:
    """Return collection items that look like cartridge checkpoints."""
    candidates: List[CollectionItem] = []
    for item in items:
        collection_item = CollectionItem.from_dict(item)
        if collection_item.repo_type == "model" and "cartridge" in collection_item.repo_id:
            candidates.append(collection_item)
    return candidates


def resolve_cartridge_ids(
    collection_slug: str,
    desired_count: int,
    override_ids: Sequence[str] | None = None,
) -> List[str]:
    """Return at least ``desired_count`` cartridge repo IDs."""
    if desired_count <= 0:
        raise ValueError("desired_count must be positive")

    if override_ids:
        unique_ids = list(dict.fromkeys(override_ids))
        if len(unique_ids) < desired_count:
            raise ValueError(
                f"Only {len(unique_ids)} cartridge ids provided; require {desired_count}"
            )
        return unique_ids[:desired_count]

    payload = _fetch_collection_payload(collection_slug)
    candidates = _candidates_from_items(payload.get("items", []))
    ids = [item.repo_id for item in candidates]
    if len(ids) < desired_count:
        raise RuntimeError(
            f"Collection '{collection_slug}' exposes {len(ids)} cartridge repos; "
            f"{desired_count} required. Provide --cartridge-ids to override."
        )
    return ids[:desired_count]


def build_cartridge_payload(
    cartridge_ids: Sequence[str],
    *,
    source: str = "huggingface",
    force_redownload: bool = False,
) -> List[dict]:
    """Return the JSON-friendly payload expected by Tokasaurus."""
    if not cartridge_ids:
        raise ValueError("cartridge_ids must not be empty")
    return [
        {
            "id": cartridge_id,
            "source": source,
            "force_redownload": force_redownload,
        }
        for cartridge_id in cartridge_ids
    ]

