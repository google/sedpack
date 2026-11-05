# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the bundled sedpack Agent Skill and skills.sh.json."""

from importlib import resources
import itertools
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from sedpack.io import (
    Attribute,
    Dataset,
    DatasetFiller,
    DatasetStructure,
    Metadata,
)
from sedpack.io.iteration import RustBatchedGenerator
from sedpack.io.shard_file_metadata import ShardInfo
from sedpack.io.shard_info_iterator.balanced_iterator import (
    BalancedShardInfoIterator,)
from sedpack.io.types import ExampleT, SplitT, TEST_SPLIT, TRAIN_SPLIT


def _feed_worker(
    dataset_filler: DatasetFiller,
    worker_items: list[int],
    *,
    split: SplitT,
) -> int:
    """Top-level multiprocessing worker matching Pattern B in SKILL.md."""
    with dataset_filler:
        for item in worker_items:
            dataset_filler.write_example(
                values=_make_example(item),
                split=split,
            )
    return len(worker_items)


def _make_example(i: int) -> ExampleT:
    """Helper matching make_example in SKILL.md."""
    return {
        "trace": np.zeros((32,), dtype=np.float32),
        "plaintext": np.zeros((16,), dtype=np.uint8),
        "key_id": i,
        "label": np.asarray(i % 256, dtype=np.uint8),
        "raw_packet": b"variable_len_bytes",
        "fixed_nonce": b"0123456789ab",
        "notes": f"example_{i}",
    }


def _by_device(shard_info: ShardInfo) -> int:
    """Shard balancing key matching Section 6 in SKILL.md."""
    return int(shard_info.custom_metadata["device"])


def test_bundled_skill_location_and_content() -> None:
    """Verify SKILL.md is bundled at sedpack/.agents/skills/sedpack/SKILL.md."""
    skill_dir = resources.files("sedpack") / ".agents" / "skills" / "sedpack"
    skill_md = skill_dir / "SKILL.md"
    assert skill_dir.is_dir()
    assert skill_md.is_file()

    text = skill_md.read_text(encoding="utf-8")
    raw_bytes = skill_md.read_bytes()
    line_count = len(text.splitlines())
    assert 50 <= line_count <= 500
    assert len(raw_bytes) <= 35_000

    assert text.startswith("---\n")
    end_idx = text.find("\n---\n", 4)
    assert end_idx != -1, "Missing closing YAML frontmatter delimiter"
    frontmatter = text[4:end_idx]

    name_match = re.search(r"^name:\s*(\S+)\s*$", frontmatter, re.MULTILINE)
    assert name_match is not None
    skill_name = name_match.group(1)
    assert skill_name == "sedpack"
    assert re.fullmatch(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$", skill_name)

    assert "description: >-\n" in frontmatter
    desc_lines: list[str] = []
    in_desc = False
    for line in frontmatter.splitlines():
        if line.startswith("description:"):
            in_desc = True
            continue
        if in_desc:
            if line.startswith("  "):
                desc_lines.append(line.strip())
            else:
                break
    description = " ".join(desc_lines)
    assert 0 < len(description) <= 1024
    assert "<" not in description and ">" not in description
    assert "Use when" in description
    assert "Don't use for" in description
    assert "/tmp/" not in text

    for expected_symbol in (
            "Dataset",
            "Metadata",
            "DatasetStructure",
            "Attribute",
            "as_numpy_iterator",
            "as_tfdataset",
            "write_multiprocessing",
            "BalancedShardInfoIterator",
            "allow_pickle=False",
    ):
        assert expected_symbol in text


def test_skill_documented_workflows_execute(tmp_path: Path) -> None:
    """Execute the core workflows documented in SKILL.md end-to-end."""
    dataset_path = tmp_path / "aes_traces"
    metadata = Metadata(
        description="AES power traces dataset",
        dataset_license="Apache-2.0",
        dataset_version="1.0.0",
        custom_metadata={"capture_scope": "ChipWhisperer"},
    )
    dataset_structure = DatasetStructure(
        saved_data_description=[
            Attribute(name="trace", dtype="float32", shape=(32,)),
            Attribute(name="plaintext", dtype="uint8", shape=(16,)),
            Attribute(name="key_id", dtype="int", shape=()),
            Attribute(name="label", dtype="uint8", shape=()),
            Attribute(name="raw_packet", dtype="bytes", shape=()),
            Attribute(name="fixed_nonce", dtype="bytes", shape=(12,)),
            Attribute(name="notes", dtype="str", shape=()),
        ],
        compression="LZ4",
        examples_per_shard=4,
        shard_file_type="fb",
        hash_checksum_algorithms=("sha256",),
    )

    dataset = Dataset.create(
        path=dataset_path,
        metadata=metadata,
        dataset_structure=dataset_structure,
    )

    with dataset.filler(concurrency=2) as dataset_filler:
        for i in range(8):
            dataset_filler.write_example(
                values=_make_example(i),
                split=TRAIN_SPLIT,
                custom_metadata={"device": i % 2},
            )

    counts = dataset.write_multiprocessing(
        feed_writer=_feed_worker,
        custom_arguments=[([0, 1],), ([2, 3],)],
        custom_kwarguments=[{
            "split": TEST_SPLIT
        }, {
            "split": TEST_SPLIT
        }],
        consistency_check=True,
        single_process=True,
    )
    assert counts == [2, 2]

    reloaded = Dataset(dataset_path)
    reloaded.check(show_progressbar=False)
    assert reloaded.dataset_info.splits[TRAIN_SPLIT].number_of_examples == 8
    assert reloaded.dataset_info.splits[TEST_SPLIT].number_of_examples == 4

    batches = list(
        reloaded.as_numpy_iterator_rust_batched(
            split=TRAIN_SPLIT,
            batch_size=4,
            repeat=False,
            shuffle=0,
        ))
    assert len(batches) == 2
    assert isinstance(batches[0]["trace"], np.ndarray)
    assert batches[0]["trace"].shape == (4, 32)
    raw_packets: Any = batches[0]["raw_packet"]
    assert raw_packets == [b"variable_len_bytes"] * 4
    notes: Any = batches[0]["notes"]
    assert len(notes) == 4

    filtered = list(
        reloaded.as_numpy_iterator_rust(
            split=TRAIN_SPLIT,
            repeat=False,
            shuffle=0,
            shard_filter=lambda s: s.custom_metadata.get("device") == 0,
        ))
    assert len(filtered) == 4

    balanced_shards = BalancedShardInfoIterator(
        dataset_path=reloaded.path,
        dataset_info=reloaded.dataset_info,
        split=TRAIN_SPLIT,
        repeat=True,
        shuffle=10,
        balance_by=(_by_device,),
        shard_filter=None,
    )
    with RustBatchedGenerator(
            dataset_path=reloaded.path,
            dataset_structure=reloaded.dataset_structure,
            shard_iterator=balanced_shards,
            batch_size=4,
            process_batch=None,
            shuffle_buffer_size=16,
    ) as rust_generator:
        sampled_batches = list(itertools.islice(rust_generator(), 3))
    assert len(sampled_batches) == 3


def test_skills_sh_json_config() -> None:
    """Verify skills.sh.json follows the skills.sh schema and resolves."""
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "skills.sh.json"
    assert config_path.is_file()

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert (config.get("$schema") ==
            "https://skills.sh/schemas/skills.sh.schema.json")
    paths = config.get("paths")
    assert isinstance(paths, list) and paths == ["./src/sedpack/.agents/skills"]

    for rel_path in paths:
        assert isinstance(rel_path, str)
        assert rel_path.startswith("./")
        assert "\\" not in rel_path
        container = repo_root / rel_path
        assert container.is_dir()
        assert (container / "sedpack" / "SKILL.md").is_file()
