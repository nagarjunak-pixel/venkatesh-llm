#!/usr/bin/env python3
"""Interactive Q&A client for the Venkatesh Bommeparthi LLM served by serve.py.

The model (1,031,478-param char-level GPT) is loaded into RAM by serve.py and
exposed at http://127.0.0.1:11743/api/generate. This script is just a chat
front-end: it sends your prompt and prints the model's continuation.

Usage:
    cd ~/Desktop/venkatesh_llm
    ./venv/bin/python ask.py
Type a question, get the model's continuation. 'exit' or Ctrl-C to quit.

NOTE: at 1M params on ~1MB of text the output is gibberish-by-design — this is
an educational demo of the pipeline, not a capable assistant.
"""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:11743/api/generate"


def ask(prompt, max_tokens=200, temperature=0.8):
    body = json.dumps(
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode()).get("text", "")
    except Exception as e:  # pragma: no cover
        return f"[error contacting model server: {e}]"


def main():
    print("Venkatesh Bommeparthi LLM - interactive Q&A (char-level, 1.03M params)")
    print("Model served at", URL)
    print("Type a question. 'exit' to quit.\n")
    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        print("Model>", ask(q))


if __name__ == "__main__":
    main()
