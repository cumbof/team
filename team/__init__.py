"""team — orchestrate a cluster of containerized local LLMs.

A `team` is a set of `Member`s (each backed by an Ollama model running in its
own Docker container) that collaborate to accomplish a `goal` according to a
chosen `workflow` (round-robin, manager-driven, review-loop, ...).

Public entry points:

* :func:`team.config.load_team` — parse a YAML team spec.
* :class:`team.orchestrator.Orchestrator` — bring members up, run a workflow,
  tear them down.
* :mod:`team.cli` — the ``team`` command-line interface.
"""

from team._version import __version__

__all__ = ["__version__"]
