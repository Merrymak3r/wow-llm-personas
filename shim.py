#!/usr/bin/env python3
"""
WoW LLM persona shim  --  KoboldCpp-shaped facade -> local Ollama, with per-bot personas.

The cmangos/playerbots 'ai chat' feature POSTs a KoboldCpp request to
AiPlayerbot.LLMApiEndpoint:
    POST /api/v1/generate   {"max_length": 100, "prompt": "<full RP prompt>"}
and expects a KoboldCpp response:
    {"results": [{"text": "<reply>"}]}

This shim sits at that endpoint, forwards to Ollama's /api/chat, and -- the reason
it exists -- injects a per-bot personality file so named party bots have a voice,
while random ambient bots just get the server's default RP framing. Nothing here
touches the game server; it only answers HTTP.

Run:  python shim.py     (listens on 127.0.0.1:5005)
Wire: point AiPlayerbot.LLMApiEndpoint at http://127.0.0.1:5005/api/v1/generate
"""
import json
import os
import re
import sys
import threading
import urllib.request
from collections import deque, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---- config (env-overridable) -------------------------------------------------
LISTEN_HOST = os.environ.get("SHIM_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("SHIM_PORT", "5005"))
OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "hf.co/TheDrummer/Tiger-Gemma-9B-v3-GGUF:Q4_K_M")  # uncensored A/B winner: cleanest foul-mouthed character voice, ~0.8s/5.8GB. Light fallback: OLLAMA_MODEL=hf.co/mradermacher/Fiendish_LLAMA_3B-GGUF:Q4_K_M (2.2GB)
PERSONA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personas")
TEMPERATURE = float(os.environ.get("SHIM_TEMP", "0.85"))
MEM_TURNS   = int(os.environ.get("SHIM_MEM_TURNS", "6"))  # per-bot rolling conversation memory (in-process)

# ---- persona lookup -----------------------------------------------------------
# The server's pre-prompt contains "Your name is <bot name>." -- pull it out and,
# if personas/<name>.txt exists, prepend it so that bot speaks in character.
_NAME_RE = re.compile(r"[Yy]our name is (\w+)")

def bot_name(prompt: str) -> str | None:
    m = _NAME_RE.search(prompt or "")
    return m.group(1) if m else None

def persona_for(name: str | None) -> str:
    if not name:
        return ""
    path = os.path.join(PERSONA_DIR, f"{name.lower()}.txt")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return ""

# ---- bot-to-bot banter detection ----------------------------------------------
# When AiPlayerbot.LLMBotToBotChatChance fires, the server sends this bot a prompt
# whose speaker is ANOTHER bot. The pre-prompt says "<other name> is speaking to you";
# if that speaker has a persona file too, it's two persona bots bantering -- nudge
# the model to react to them BY NAME instead of firing a parallel monologue.
_SPEAKER_RE = re.compile(r"(\w+)\s+is speaking to you", re.IGNORECASE)

def speaker_name(prompt: str, responder: str | None) -> str | None:
    """Best-effort: who this bot is replying TO. Never raises -- a parse miss just
    means we fall back to the normal (non-banter) prompt."""
    try:
        m = _SPEAKER_RE.search(prompt or "")
        if m and m.group(1).lower() != (responder or "").lower():
            return m.group(1)
        lines = [ln.strip() for ln in (prompt or "").splitlines() if ln.strip()]
        if responder and lines and lines[-1].lower().rstrip(":").strip() == responder.lower():
            lines = lines[:-1]  # drop the trailing "<responder>:" completion cue
        if lines and ":" in lines[-1]:
            head = lines[-1].split(":", 1)[0].strip()
            if 0 < len(head) <= 24 and head.lower() != (responder or "").lower():
                return head
    except Exception:
        pass
    return None

# ---- per-bot conversation memory ----------------------------------------------
# In-process ring buffer (stdlib deque) keyed by bot name, so a persona remembers
# its last few exchanges and stays consistent instead of firing stateless one-shots.
# Deliberately NOT Redis: the shim stays dependency-free so it can run as a bare
# system service on stock Python with no site-packages.
_MEM_LOCK = threading.Lock()
_MEMORY: dict[str, deque] = defaultdict(lambda: deque(maxlen=MEM_TURNS))

def _incoming_line(prompt: str, name: str | None) -> str:
    """Best-effort: the line the bot is replying to (last speaker line, minus a
    trailing '<BotName>:' completion cue)."""
    lines = [ln.strip() for ln in (prompt or "").splitlines() if ln.strip()]
    if name and lines and lines[-1].lower().rstrip(":").strip() == name.lower():
        lines = lines[:-1]
    if not lines:
        return ""
    last = lines[-1]
    if ":" in last:
        head, tail = last.split(":", 1)
        if 0 < len(head) <= 24 and tail.strip():
            return tail.strip()[:200]
    return last[:200]

def memory_block(name: str | None) -> str:
    if not name:
        return ""
    with _MEM_LOCK:
        turns = list(_MEMORY.get(name.lower(), ()))
    if not turns:
        return ""
    lines = []
    for their, mine in turns:
        if their:
            lines.append(f'They said: "{their}"')
        if mine:
            lines.append(f'You said: "{mine}"')
    return ("\n\nRecent conversation so far (oldest first) -- stay consistent with it "
            "and don't repeat yourself:\n" + "\n".join(lines))

def remember(name: str | None, their: str, mine: str) -> None:
    if not name or not mine:
        return
    with _MEM_LOCK:
        _MEMORY[name.lower()].append((their, mine))

# ---- ollama call --------------------------------------------------------------
def ask_ollama(system: str, user: str, max_tokens: int) -> str:
    # Persona goes in the SYSTEM message (A/B-proven: user-message personas made
    # the RP models leak "assistant  Name says:" scaffolding). think:false because
    # some models (gemma/qwen3) are reasoning models that otherwise return empty.
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "think": False,
        "messages": msgs,
        "stream": False,
        "options": {
            "num_predict": max(24, min(max_tokens, 160)),
            "temperature": TEMPERATURE,
        },
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    return (data.get("message", {}).get("content") or "").strip()

# ---- one-line reply cleanup ---------------------------------------------------
# The bot's chat is one short line; strip role labels / newlines the model may add.
def clean(text: str, name: str | None) -> str:
    text = (text or "").strip()
    # first NON-empty line (models often lead with a blank line)
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if not line:
        return ""
    # drop an echoed "Name:" prefix and any wrapping quotes/asterisks
    if name and line.lower().startswith(f"{name.lower()}:"):
        line = line.split(":", 1)[1].strip()
    line = line.strip('"').strip("*").strip()
    return line[:230]  # bot chat is capped; keep it tight

# ---- http handler -------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        # health check + a KoboldCpp version ping some clients probe
        if self.path.startswith("/api/v1/model"):
            return self._json(200, {"result": f"ollama/{OLLAMA_MODEL}"})
        return self._json(200, {"ok": True, "model": OLLAMA_MODEL, "shim": "wow-llm-personas"})

    def do_POST(self):
        if not self.path.startswith("/api/v1/generate"):
            return self._json(404, {"error": "only /api/v1/generate"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error": f"bad request: {e}"})

        prompt = req.get("prompt", "")
        max_len = int(req.get("max_length", 100))
        name = bot_name(prompt)
        persona = persona_for(name)
        # banter mode: only when the SPEAKER is itself a known persona bot
        companion = None
        if persona:
            sp = speaker_name(prompt, name)
            if sp and persona_for(sp):
                companion = sp
        if companion:
            brevity = (f" You're trading quick banter with your companion {companion}. React to what "
                       f"{companion} just said, address them by name, keep it playful and in character. "
                       "Reply with ONE short line, under 25 words. No narration, no name prefix, no asterisks.")
        else:
            brevity = " Reply with ONE short in-character line, under 25 words. No narration, no name prefix, no asterisks."
        base = persona if persona else "Answer as a WoW roleplaying character."
        system = base + memory_block(name) + brevity  # inject this bot's recent memory

        try:
            raw = ask_ollama(system, prompt, max_len)
            reply = clean(raw, name)
        except Exception as e:
            sys.stderr.write(f"[shim] ollama error: {e}\n")
            return self._json(200, {"results": [{"text": ""}]})  # fail quiet: bot just stays silent

        remember(name, _incoming_line(prompt, name), reply)  # store this exchange for continuity

        tag = f"{name}{' *persona*' if persona else ''}" if name else "?"
        if companion:
            tag += f" ->banter@{companion}"
        sys.stderr.write(f"[shim] {tag}: {reply!r}\n")
        return self._json(200, {"results": [{"text": reply}]})

    def log_message(self, *a):  # silence default per-request noise
        pass

if __name__ == "__main__":
    os.makedirs(PERSONA_DIR, exist_ok=True)
    srv = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    sys.stderr.write(
        f"[shim] WoW LLM persona shim up on http://{LISTEN_HOST}:{LISTEN_PORT}/api/v1/generate\n"
        f"[shim] -> {OLLAMA_URL} ({OLLAMA_MODEL}); personas in {PERSONA_DIR}\n")
    srv.serve_forever()
