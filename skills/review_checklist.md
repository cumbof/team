# Peer Review Checklist

Use this checklist whenever you are reviewing another member's contribution —
code, prose, design, data, or any other artifact.  Work through each section
systematically and cite specific lines or passages when raising a concern.

## 1. Correctness
- [ ] Does the artifact do what it claims to do?  Trace the logic manually for
      at least one representative input.
- [ ] Are edge cases handled (empty input, boundary values, null / None,
      very large or very small values)?
- [ ] Are error paths explicit?  Does the code fail loudly rather than
      silently swallowing exceptions?
- [ ] For code: does it compile / parse without errors?  Are there obvious
      runtime hazards (off-by-one, unguarded index, division by zero)?

## 2. Completeness
- [ ] Are all deliverables listed in the goal present?
- [ ] Are there any obvious gaps — missing sections, unimplemented stubs,
      `TODO` / `...` placeholders left in place?
- [ ] Is the artifact self-contained, or does it depend on something that has
      not yet been created?

## 3. Consistency with prior decisions
- [ ] Does this artifact agree with decisions already logged in `decisions.md`?
- [ ] Does it follow the naming, style, and structural conventions established
      in earlier turns?
- [ ] If it contradicts an earlier decision, is that contradiction explicit and
      justified?

## 4. Clarity and quality
- [ ] Is the intent clear to a reader who was not involved in producing it?
- [ ] Are variable / function / section names descriptive?
- [ ] Is the level of detail appropriate — neither over-engineered nor
      underspecified?

## 5. Testability and verifiability
- [ ] Can the artifact be tested or verified independently?
- [ ] Are the acceptance criteria clear enough to write a passing test?
- [ ] For code: are unit tests present or explicitly deferred with a reason?

## Verdict
End your review with one of:

- **APPROVED** — ready to ship as-is.
- **APPROVED WITH NOTES** — minor issues that can be fixed without another
  review cycle; list them explicitly.
- **CHANGES REQUESTED** — one or more blocking issues; list each one with a
  suggested fix or a concrete question.
