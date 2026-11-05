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

import ast
from importlib import resources
import json
from pathlib import Path
import re


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

    python_blocks = re.findall(
        r"```python\n(.*?)```",
        text,
        flags=re.DOTALL,
    )
    assert len(python_blocks) >= 10
    for block in python_blocks:
        ast.parse(block)


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
