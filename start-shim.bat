@echo off
REM WoW LLM persona shim. Leave this window open while you play.
REM Override the model by setting OLLAMA_MODEL before running, e.g.:
REM   set OLLAMA_MODEL=hf.co/mradermacher/Fiendish_LLAMA_3B-GGUF:Q4_K_M  &  start-shim.bat
cd /d "%~dp0"
echo Starting WoW LLM persona shim on http://127.0.0.1:5005 ...
python shim.py
pause
