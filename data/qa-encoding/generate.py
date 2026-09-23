"""Rebuild the fixed synthetic questionnaire responses.

The 50 public-domain IPIP item statements and keys live in items.csv. This
generator creates response fixtures only; it does not score personalities.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
SEED = "hyp84-questionnaire-v1"
SENSITIVITY_PROFILES = 30
FEATURED_ANCHORS = (4, 3, 5, 2, 4)
FACTOR_NAMES = (
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Emotional Stability",
    "Intellect/Imagination",
)


def stable_byte(*parts: object) -> int:
    payload = "|".join(str(part) for part in (SEED, *parts)).encode("utf-8")
    return hashlib.sha256(payload).digest()[0]


def load_items() -> list[dict[str, str]]:
    with (HERE / "items.csv").open(newline="", encoding="utf-8") as file:
        items = list(csv.DictReader(file))

    if len(items) != 50:
        raise ValueError(f"Expected 50 items, found {len(items)}")
    if len({item["item_id"] for item in items}) != 50:
        raise ValueError("Question IDs must be unique")

    for position, item in enumerate(items, start=1):
        factor_id = (position - 1) % 5 + 1
        expected = {
            "item_id": f"Q{position:02d}",
            "position": str(position),
            "factor_id": str(factor_id),
            "factor_name": FACTOR_NAMES[factor_id - 1],
        }
        for field, value in expected.items():
            if item[field] != value:
                raise ValueError(f"Unexpected {field} for {expected['item_id']}")
        if item["key_direction"] not in {"+", "-"}:
            raise ValueError(f"Invalid key direction for {item['item_id']}")
        if not item["statement"].strip():
            raise ValueError(f"Empty statement for {item['item_id']}")
    return items


def profile_anchors(profile_id: str) -> tuple[int, ...]:
    if profile_id == "featured":
        return FEATURED_ANCHORS
    return tuple(1 + stable_byte(profile_id, factor_id, "anchor") % 5 for factor_id in range(1, 6))


def raw_answer(profile_id: str, item: dict[str, str], anchors: tuple[int, ...]) -> int:
    factor_id = int(item["factor_id"])
    anchor = anchors[factor_id - 1]
    keyed_answer = anchor if item["key_direction"] == "+" else 6 - anchor
    bucket = stable_byte(profile_id, item["item_id"], "item-variation") % 6
    variation = -1 if bucket == 0 else (1 if bucket == 5 else 0)
    return max(1, min(5, keyed_answer + variation))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    items = load_items()
    profiles: list[dict[str, object]] = []
    responses: list[dict[str, object]] = []
    ids = ["featured", *(f"sensitivity_{index:02d}" for index in range(1, SENSITIVITY_PROFILES + 1))]

    for profile_id in ids:
        anchors = profile_anchors(profile_id)
        profiles.append(
            {
                "profile_id": profile_id,
                "profile_kind": "featured" if profile_id == "featured" else "sensitivity",
                **{f"factor_{index}_anchor": anchor for index, anchor in enumerate(anchors, start=1)},
            }
        )
        for item in items:
            responses.append(
                {
                    "profile_id": profile_id,
                    "item_id": item["item_id"],
                    "raw_answer": raw_answer(profile_id, item, anchors),
                }
            )

    write_csv(
        HERE / "profiles.csv",
        ["profile_id", "profile_kind", *(f"factor_{index}_anchor" for index in range(1, 6))],
        profiles,
    )
    write_csv(HERE / "responses.csv", ["profile_id", "item_id", "raw_answer"], responses)


if __name__ == "__main__":
    main()
