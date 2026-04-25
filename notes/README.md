# notes/ — paper-writing scratchpad

Purpose: capture *narrative* and *paper-relevant* content as the project unfolds, so that when we sit down to write the manuscript six months from now we don't have to reverse-engineer findings from `git log`. **Different audience from `methods/`**: those docs are reproducibility-focused (someone re-running our pipeline; supplementary text). The files here are for us, writing the manuscript.

## Files

| File | Purpose | Update cadence |
|---|---|---|
| `research_journal.md` | Chronological log of *what* we discovered *when*, in 2-3 sentences each. The history of the project from the paper writer's perspective. | Add an entry whenever a non-trivial finding lands (post-commit). |
| `surprises_and_caveats.md` | Findings that surprised us (positive or negative), and methodological caveats reviewers will care about. Feeds the paper's Discussion + Limitations sections. | Add an entry whenever a finding subverts our prior expectation, or whenever a caveat is identified. |
| `open_questions.md` | Things we noticed but haven't resolved. Could feed `FUTURE_WORK.md` later, or be addressed in a later phase. Keeps "things we should think about" from getting lost. | Add when a question arises; remove or move to `FUTURE_WORK.md` when resolved. |
| `paper_outline.md` | Target paper structure (Intro / Results / Methods / Discussion) with placeholders for findings to slot in. Helps us see the shape of the manuscript. | Update when paper structure decisions are made. |
| `paper_figures_inventory.md` | Every figure we might want, with: data source, current status, paper section it serves. Prevents figure ideas from getting forgotten. | Add when a figure is conceived; mark status changes (planned → drafted → committed → polished). |
| `paper_tables_inventory.md` | Same as figures, for tables. | Same cadence. |

## Maintenance protocol

- **Append-only** for `research_journal.md` (history doesn't get rewritten).
- **Edit-in-place** for the others (state evolves).
- Keep entries short (2-5 sentences). Long-form goes in `methods/*.md`.
- Cross-reference: every entry points to the relevant commit hash, methods doc, or roadmap section.
- When a finding is described in `methods/`, the journal entry just summarizes + links; don't duplicate.
