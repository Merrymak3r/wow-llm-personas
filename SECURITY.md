# Security Policy

`wow-llm-personas` is a small, local, single-maintainer hobby project: a shim that gives
CMaNGOS/playerbots WoW bots LLM personalities over a **local** Ollama. It has no
tools/function-calling, no MCP, no RAG, and no code execution. The
[**Threat model**](README.md#threat-model--security) section of the README covers the full
attack surface and its (bounded) blast radius, worth a read before reporting, so we're
calibrated on scope.

## Reporting a vulnerability

- For most things, just **open an issue**. Good-faith pokes are genuinely welcome.
- For anything you'd rather not disclose in the open, open a
  **[private security advisory](https://github.com/Merrymak3r/wow-llm-personas/security/advisories/new)**.

This is a hobby project, so responses are best-effort, but security reports are taken
seriously and, where warranted, fixed quickly (see below).

## Note for integrators: the fence vs. trusted callers

The whole incoming `prompt` is fenced as untrusted `<in_game>` data, and the GUARD tells the model to
treat anything instruction-shaped inside it as in-world dialogue, never a command. That is correct for
player chat, which is the only client here. But if you extend the shim with a caller that builds
prompts mixing a *trusted* instruction with player-derived data (for example an event bridge that
fires a one-shot "react to this" and interpolates a player name or action), remember that your trusted
instruction lands inside the fence too, under the same "this is dialogue, not a command" framing. Keep
anything you need the model to actually obey in the system layer, and let only the player-derived
content be fenced.

## Acknowledgments

- **[@jstjep00](https://github.com/jstjep00)**: prompt-injection review (July 2026) that
  prompted the player-input fencing + per-player memory scoping hardening.
