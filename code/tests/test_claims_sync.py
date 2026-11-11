"""findings.md Section 1 and claims.md Table 1 are the same short list.

findings.md says so in prose: "Every row here appears there, with the same
value and the same run id, and neither file may gain a row the other lacks."
A rule that lives in a document gets skipped, see L-013, so it lives here
instead. Both files are separately checked against runs/ by check_findings,
which stops either drifting from the data; this stops them drifting from
each other, which no run can catch because both would still be individually
correct while disagreeing about what the short list is.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"


def _rows(path: Path, value_header: str, start: str, end: str) -> dict[str, str]:
    """Rows naming a run, as run_id to value, within one section.

    Scoped to a section rather than to the whole file, because both documents
    carry other tables with a Value column and those are not the short list.
    """
    text = path.read_text(encoding="utf-8")
    text = text[text.index(start):]
    if end in text:
        text = text[:text.index(end)]
    out: dict[str, str] = {}
    header: list[str] | None = None
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = [c.replace("*", "").strip().lower() for c in cells]
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        if value_header not in header or "run_id" not in header:
            continue
        run = cells[header.index("run_id")].replace("`", "").strip()
        val = cells[header.index(value_header)].replace("*", "").strip()
        if run and re.fullmatch(r"\d+\.\d+", val):
            out[run] = val
    return out


@pytest.mark.skipif(not (DOCS / "claims.md").is_file(), reason="claims.md absent")
def test_the_short_list_is_the_same_in_both_files() -> None:
    findings = _rows(DOCS / "findings.md", "value",
                     "## 1. Headline claims", "## 2. Gate outcomes")
    claims = _rows(DOCS / "claims.md", "value",
                   "## Table 1.", "## Table 2.")

    assert findings, "findings.md Section 1 has no run-attributed rows"
    assert claims, "claims.md Table 1 has no run-attributed rows"

    only_findings = sorted(set(findings) - set(claims))
    only_claims = sorted(set(claims) - set(findings))
    assert not only_findings, f"in findings.md and not claims.md: {only_findings}"
    assert not only_claims, f"in claims.md and not findings.md: {only_claims}"

    disagree = {r: (findings[r], claims[r]) for r in findings
               if findings[r] != claims[r]}
    assert not disagree, f"same run, different value: {disagree}"


@pytest.mark.skipif(not (DOCS / "claims.md").is_file(), reason="claims.md absent")
def test_no_row_claims_a_confirmation_run_before_milestone_9() -> None:
    """Milestone 9 is what turns a run id into a confirming run id, and it has
    not started. A row that quietly starts claiming confirmation is the way
    this file stops being worth reading."""
    for name in ("claims.md", "findings.md"):
        # Whitespace collapsed, because both files are hard wrapped and the
        # phrase lands across a line break in one of them.
        text = " ".join((DOCS / name).read_text(encoding="utf-8").lower().split())
        assert "milestone 9 has not started" in text, name
        # The word to be afraid of is a row quietly asserting it is done.
        assert "confirmed by run" not in text, name
