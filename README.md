# wow-llm-personas

A tiny (~230-line, **stdlib-only**) Python shim that gives [CMaNGOS](https://github.com/cmangos) +
[playerbots](https://github.com/cmangos/playerbots) AI playerbots real **personalities** by
routing their in-game "ai chat" through your **local [Ollama](https://ollama.com)**.

Party bots roast you in character, remember the last few things you said, and banter with each
other, all on one consumer GPU, nothing leaving your LAN.

> This is the shim behind [that r/homelab post] about ~1,800 vanilla-WoW bots with AI personalities.
> The bots and the server are stock projects; **this repo is the glue that gives them a voice**:
> the persona injection, the per-bot memory, and the bot-to-bot banter tuning.

## Prerequisites

This shim only handles the *personality* layer. You need a working **CMaNGOS 1.12 + playerbots**
server first (the one with the native LLM-chat hook). The fastest path, and the one I used, is the
prebuilt **Eluna-CMaNGOS-Classic** Windows builds, which bundle the bots, Eluna, and the extractors:

- **Core (prebuilt, easiest):** [Eluna-Ports/Eluna-CMaNGOS-Classic](https://github.com/Eluna-Ports/Eluna-CMaNGOS-Classic): grab a `with-all` build
- **Playerbots module:** [cmangos/playerbots](https://github.com/cmangos/playerbots)
- **World database:** [cmangos/classic-db](https://github.com/cmangos/classic-db)
- **LLM runtime:** [Ollama](https://ollama.com), with a model pulled

You supply your own 1.12 game client. This repo distributes no game data and links to none. Once your
bots chat *at all* server-side, point `AiPlayerbot.LLMApiEndpoint` at this shim and you're set (see
[Quick start](#quick-start)).

## How it works

The playerbots module already has a native "LLM chat" hook: when a bot talks, it POSTs a
[KoboldCpp](https://github.com/LostRuins/koboldcpp)-shaped request to whatever endpoint you configure:

```
POST /api/v1/generate   {"max_length": 100, "prompt": "<the full RP prompt>"}
->                       {"results": [{"text": "<the reply>"}]}
```

So instead of running KoboldCpp, you point it at **this shim**, which speaks that shape on the front
and talks to **Ollama** on the back. In between, it does the three things that turn "an LLM answered"
into "a character answered":

### 1. Per-bot persona injection
The server's prompt always contains `Your name is <BotName>.`. The shim pulls that name out and, if
`personas/<name>.txt` exists, loads it as the **system message** (not prepended to the user prompt:
that's an A/B-proven fix; user-message personas made RP models leak `assistant  Name says:`
scaffolding). Bots with a persona file speak in character; nameless ambient bots just get the
server's default framing. Adding a character is dropping in a text file. No code, no restart.

### 2. Per-conversation memory
A small in-process ring buffer (`collections.deque`, `SHIM_MEM_TURNS` turns) so a persona stays
consistent across a back-and-forth instead of firing stateless one-shots. It's keyed per
`(bot, speaker)` pair, not globally per bot, so on a shared server one player's lines never surface in
another player's conversation with the same bot. Deliberately **not** Redis: keeping the shim
dependency-free means it runs as a bare system service on stock Python with zero `pip install`.

A request can set `"no_memory": true` to opt a single reply out of memory, both read and write. Use it
for one-shot calls, like an event-driven "react to this" that fires once: memory helps a conversation,
but for a one-shot it just feeds the model its own last reply and it copies it verbatim.

### 3. Bot-to-bot banter (tuned so it can't storm)
The playerbots fork can let bots reply to *each other*, so your party riffs among themselves, not just
at you. When the shim sees a prompt whose *speaker* is itself a known persona, it switches to **banter
mode** ("react to your companion `<name>` by name, keep it playful"). Real players (no persona file)
never trigger it, so player-directed chat stays normal.

The catch is a feedback loop: every bot that hears a line might answer it, and those answers are more
lines. Keep it **subcritical**: in a 5-bot party, `(party − 1) × chance = 4 × 0.20 = 0.8 < 1`, so on
average one line can't spawn more than one reply and the conversation dampens instead of exploding.
Push the chance past ~25% in a big group and it branches into spam (and thrashes your GPU). That's why
the recommended `LLMBotToBotChatChance` is **20**, with `LLMMaxSimultaniousGenerations` as a hard
concurrency backstop.

### Fail-quiet
If Ollama errors or times out, the shim returns an empty line. The bot just stays silent for a beat
instead of crashing the chat. The game server is never blocked on the LLM.

### Output hygiene
Uncensored RP models occasionally leak junk into a reply, so the shim scrubs each one before it hits
chat: it strips leaked HTML/markup tags, cuts third-person narration that follows a quote (`", said
the gnome rogue...`) while leaving a legit in-character line like `Gandalf said we should go`, peels
off a leading attribution wrapper (`Thorgrim yells back: "..."`) and a copied memory-block scaffold
(`You said: "..."`), drops emoji and non-BMP symbols (they break `utf8mb3` DB columns, get read aloud
by some TTS engines, and
vanilla WoW can't render them anyway), and rejects sub-stub fragments so a `.. yes.` glitch never
reaches chat. Variety sampling (`top_p` / `top_k` / `repeat_penalty`, see Config) keeps replies from
converging on the same catchphrase.

## Model choice

Default is **`hf.co/mradermacher/Fiendish_LLAMA_3B-GGUF:Q4_K_M`** (~0.5 s/reply, ~2.6 GB VRAM), an
**uncensored** community fine-tune small enough to run on almost any GPU and to co-reside with other
models on one card. It's the friendly default; the output sanitizers clean up its occasional
name-prefix leak.

Richer, saltier option if you have the VRAM: **`hf.co/TheDrummer/Tiger-Gemma-9B-v3-GGUF:Q4_K_M`**
(~0.8 s/reply, ~5.8 GB), the model behind the r/homelab post. Both are uncensored on purpose: in a
head-to-head the polite, instruction-tuned models refused to stay in character and wouldn't get salty,
which is exactly what a party of wisecracking NPCs needs.

Set either via the `OLLAMA_MODEL` env var. Two things worth knowing:
- **Reasoning models need handling.** Gemma/Qwen "thinking" variants otherwise return an empty
  `content` (the text lands in a `thinking` field). The shim sends `think: false`, which fixes Gemma;
  Qwen3 needs a `/no_think` token instead. Easiest is to just pick a clean non-thinking model.
- **If this GPU also runs other models,** heavy bot chat forces Ollama model-swaps that add latency to
  everything else sharing the card, another reason to keep banter subcritical and cap concurrency.

## Quick start

```bash
# 1. have Ollama running and the model pulled
ollama pull hf.co/mradermacher/Fiendish_LLAMA_3B-GGUF:Q4_K_M

# 2. start the shim (stdlib only, no venv, no requirements.txt)
python shim.py            # or: start-shim.bat on Windows
curl http://127.0.0.1:5005/     # -> {"ok": true, ...}
```

Then point the game server at it, in `aiplayerbot.conf`:

```ini
AiPlayerbot.LLMEnabled = 2
AiPlayerbot.LLMApiEndpoint = http://127.0.0.1:5005/api/v1/generate
AiPlayerbot.LLMApiJson = {"max_length": 100, "prompt": "[<pre prompt>]<context> <prompt> <post prompt>"}
AiPlayerbot.LLMMaxSimultaniousGenerations = 3

# optional: let the party banter with each other (see the math above)
AiPlayerbot.LLMBotToBotChatChance = 20
AiPlayerbot.LLMBlockedReplyChannels = world,general,trade,lfg,ldefence,wdefence,grecruitement
```

Restart `mangosd`. Whisper a bot or watch General. Persona bots reply in character.
Rollback is instant: `AiPlayerbot.LLMEnabled = 0` and restart.

(The shipped pre/post prompts and response-parse regexes already match what this shim returns; you
don't need to touch them. `LLMBlockedReplyChannels` keeps banter in local/social channels: say,
party, raid, guild, whisper, so it's fun to overhear, not zone-wide spam.)

## Writing personas

Drop a `personas/<botname>.txt` (lowercase, matching the bot's in-game name) with a short brief.
Three originals are included: `thorgrim` (gruff dwarf tank), `melwyn` (anxious priest), `pockets`
(scheming gnome rogue), plus `_template.txt`. ~60-100 words works best; small models drift if you
over-write it, and the `never break character / replies under 25 words` tail is doing real work.
Bring your own cast. They're just text files.

## Config (all env vars, all optional)

| Var | Default | What |
|-----|---------|------|
| `OLLAMA_MODEL` | `Fiendish_LLAMA_3B` (uncensored) | model tag; swap to `Tiger-Gemma-9B-v3` for a richer voice |
| `OLLAMA_URL` | `http://127.0.0.1:11434/api/chat` | your Ollama endpoint |
| `SHIM_HOST` / `SHIM_PORT` | `127.0.0.1` / `5005` | where the shim listens |
| `SHIM_TEMP` | `0.85` | sampling temperature |
| `SHIM_ADULT` | `1` (on) | in-character profanity / innuendo license; set `0` for a family-friendly server |
| `SHIM_TOP_P` / `SHIM_TOP_K` | `0.95` / `60` | variety sampling; higher = less repetitive |
| `SHIM_REPEAT_PENALTY` | `1.15` | penalize repeated tokens (anti-catchphrase) |
| `SHIM_MEM_TURNS` | `6` | memory depth per (bot, speaker) conversation |

(`no_memory` is a per-request JSON field, not an env var; see Per-conversation memory above.)

## Threat model & security

The only client is your game server, and **player-typed chat reaches the model as untrusted input**, so a player can attempt prompt injection ("ignore your instructions and…"). Worth being clear-eyed about what that can and can't do here.

**What the shim is *not*.** No tools/function-calling, no MCP, no RAG or external data, no code execution, no outbound network beyond your local Ollama. It reads a persona `.txt` and returns one short string. So a successful injection has a **bounded blast radius**: at worst a bot breaks character, prints arbitrary text in chat, or surfaces its own persona text / recent memory. There's nothing to exfiltrate beyond that, and nothing for an injection to *do*.

**What's hardened:**
- **Player text is fenced as data**, wrapped in an `<in_game>…</in_game>` block, with the system prompt instructing the model to treat anything instruction-shaped inside it as in-character dialogue, never a command. This *raises the bar* on injection; it is **not** a hard guarantee: prompting alone can't fully prevent it.
- **Memory is scoped per (bot, speaker) pair**, so on a shared/multi-player server, one player's lines can't surface in another player's conversation with the same bot.

**Operator guidance:**
- **Keep `SHIM_HOST=127.0.0.1`** (the default): loopback-only, so only local processes can reach it. If you must expose it on a LAN, put auth / a reverse proxy in front; the endpoint itself is unauthenticated.
- **Running an uncensored model in front of strangers** (a semi-public server)? It will say more when jailbroken than a safety-tuned model would, and `SHIM_ADULT` is **on by default** (an in-character profanity / innuendo license). Set `SHIM_ADULT=0` and moderate accordingly for a public venue. On a solo/trusted server this is moot.

The hardening in this section was prompted by a thoughtful prompt-injection review from
**[@jstjep00](https://github.com/jstjep00)** (much appreciated). Found something else? Open an
issue (or see [`SECURITY.md`](SECURITY.md)); good-faith pokes are welcome.

## Credits & license

Built on the excellent [CMaNGOS](https://github.com/cmangos) core and the
[playerbots](https://github.com/cmangos/playerbots) module, served by [Ollama](https://ollama.com).
This shim is just the persona/memory/banter layer on top. MIT. See [LICENSE](LICENSE).
