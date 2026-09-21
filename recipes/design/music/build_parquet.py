#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Build the music arm's train/val parquets from a query file.

Input is a json list of ``{src_id, lang, tag, length, nvoice_want, bpm, meter,
text}``; ``text`` is the brief shown to the policy verbatim, in Chinese or
English, and it is the only field the rollout sees.

    MUSIC_DATA=/path/to/music_data python3 -m recipes.design.music.build_parquet

Reads ``$MUSIC_DATA/queries.json`` and writes ``music_train.parquet`` /
``music_val.parquet`` beside it. No path defaults into this repository: the
queries are data, the training launcher reads ``$MUSIC_DATA`` too, so the two
cannot drift.

Row schema, which is what ``verl.utils.dataset.rl_dataset.RLHFDataset`` expects:

    prompt        list[dict]  chat messages; one user turn
    data_source   "music"     metric grouping key; routes nothing (see below)
    ability       "music"     carried by verl convention; read by nothing
    reward_model  {style, ground_truth}   ground_truth is empty -- there is no gold
    extra_info    {index, src_id, lang, tag, length, nvoice_want, bpm, meter}

``data_source`` does not select the scorer. The launcher names the scorer
explicitly through ``reward.custom_reward_function``, which short-circuits
verl's ``data_source`` dispatch entirely, so this value only groups metrics.

Nothing reads ``extra_info`` at scoring time either: the scorer looks only at the
ABC the policy emitted. The constraint fields are build-time provenance -- and
they are also the material for grading instruction-following, which this scorer
does not do.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pandas as pd

N_VAL = 182
SEED = 20260827


def to_verl_row(item: dict, index: int) -> dict:
    return {
        "prompt": [{"role": "user", "content": item["text"]}],
        "data_source": "music",
        "ability": "music",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "index": index,
            "src_id": item.get("src_id", ""),
            "lang": item.get("lang", ""),
            "tag": item.get("tag", ""),
            "length": item.get("length", ""),
            "nvoice_want": int(item.get("nvoice_want") or 0),
            "bpm": int(item.get("bpm") or 0),
            "meter": item.get("meter", ""),
        },
    }


def main() -> None:
    data_dir = Path(os.environ["MUSIC_DATA"])
    src = Path(os.environ.get("MUSIC_QUERIES", str(data_dir / "queries.json")))
    items = json.loads(src.read_text())
    print(f"loaded {len(items)} queries from {src}")
    if len(items) <= N_VAL:
        raise ValueError(f"{src}: need more than {N_VAL} queries, found {len(items)}")

    random.Random(SEED).shuffle(items)
    splits = {
        "music_val.parquet": items[:N_VAL],
        "music_train.parquet": items[N_VAL:],
    }

    for name, split_items in splits.items():
        rows = [to_verl_row(item, i) for i, item in enumerate(split_items)]
        path = data_dir / name
        pd.DataFrame(rows).to_parquet(path)
        langs: dict[str, int] = {}
        for row in rows:
            lang = row["extra_info"]["lang"]
            langs[lang] = langs.get(lang, 0) + 1
        lengths = sorted(len(row["prompt"][0]["content"]) for row in rows)
        mid = lengths[len(lengths) // 2]
        p95 = lengths[int(len(lengths) * 0.95)]
        print(f"wrote {path} ({len(rows)} rows) lang={langs} prompt_chars p50={mid} p95={p95}")


if __name__ == "__main__":
    main()
