#!/usr/bin/env python3
"""Generate the conservative KV capability matrix."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_streaming.capability_state import (  # noqa: E402
    CAPABILITY_STATES,
    dense_kv_capability_matrix,
)


def _git_revision():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_report():
    entries = [item.as_dict() for item in dense_kv_capability_matrix()]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "vocabulary": list(CAPABILITY_STATES),
        "qualification_ready_is_qualified": False,
        "entries": entries,
    }


def render_markdown(report):
    lines = [
        "# KV Capability Matrix",
        "",
        "`QUALIFICATION_READY` means the implementation and harness are ready; it does not mean `QUALIFIED`.",
        "`BLOCKED` is reported separately by qualification plan bundles as an execution disposition, not as a capability maturity state.",
        "",
        "| Capability | State | Evidence | Blockers |",
        "|---|---|---|---|",
    ]
    for item in report["entries"]:
        lines.append(
            "| {} | `{}` | {} | {} |".format(
                item["capability"],
                item["state"],
                "<br>".join(item["evidence"]),
                "<br>".join(item["blockers"]),
            )
        )
    lines.extend(
        [
            "",
            "Legacy Provider ABI labels remain serialized for compatibility and are conservatively mapped in `layer_streaming/capability_state.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = output_dir / "matrix.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default=str(ROOT / "reports" / "kv_capability_matrix")
    )
    args = parser.parse_args(argv)
    paths = write_report(args.output_dir)
    print("\n".join(str(item) for item in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
