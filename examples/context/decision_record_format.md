# Decision Record Format

Use `log_decision` whenever the team makes a choice that is non-obvious,
controversial, or likely to be revisited.  Good decision records prevent
circular debates and help new turns re-orient without replaying the whole
transcript.

## When to write a decision record

- An architectural or design choice was made among two or more alternatives.
- A trade-off was accepted (speed vs. correctness, simplicity vs. flexibility).
- A requirement was interpreted in a specific way.
- The team agreed to defer something or explicitly decided *not* to do it.

## Template

```
## <Short imperative title — what was decided>

**Status:** <Proposed | Accepted | Superseded by #N | Deprecated>

**Context**
One paragraph: what situation prompted this decision?  What constraints,
requirements, or prior discussions are relevant?

**Decision**
What was decided, in plain language.  Be specific enough that a reader
who was not present can understand the choice without reading the transcript.

**Alternatives considered**
- Option A — why it was rejected or deprioritised.
- Option B — why it was rejected or deprioritised.

**Consequences**
- What becomes easier or better as a result?
- What becomes harder or worse?  What follow-on work does this create?
```

## Example

```
## Use SQLite for the local run cache

**Status:** Accepted

**Context**
The team needs a lightweight store for caching LLM responses between runs.
The store must work without a network connection and without a running server.

**Decision**
Use SQLite via the standard-library `sqlite3` module.  A single `.db` file
is written to the workspace directory.

**Alternatives considered**
- Redis — requires a running server; overkill for local use.
- Plain JSON files — poor query performance; concurrent writes are error-prone.

**Consequences**
- Zero extra dependencies; ships in the Python standard library.
- Concurrent writes from parallel members need a write lock (acceptable:
  cache hits are reads; writes are rare).
```

## Rules

- Titles must be short imperatives: "Use X for Y", "Defer Z until W", "Do not Y".
- Mark superseded records explicitly — do **not** delete old entries.
- Link forward: if a new decision overrides an old one, write "Superseded by:
  <new title>" in the old record's status line.
