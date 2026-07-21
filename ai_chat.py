"""
ai_chat.py - Conversation memory and the AI "brain"
----------------------------------------------------------------------
Everything about TALKING TO THE AI lives here: turning "what the user
is looking at" into context the AI can use, keeping track of the
conversation so far, and the one function that actually calls an LLM.

*** call_llm() is the only function that changes when you pick a
provider (Groq / Anthropic / OpenAI / Cerebras / ...). Everything else
in this file, and everything in server.py, stays the same. ***

API KEY SETUP:
    Create a file named ".env" in this same folder 
        GROQ_API_KEY=your_actual_key_here
    This keeps the key out of your source code entirely - the app
    reads it from the environment, never hardcoded, never sent to the
    browser. Since we're providing the key (not the end user), this is
    the only place it needs to live.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env into the environment, if the file exists
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"  # same conversational model TARZ uses

# LLMs are stateless - every API call is a blank slate unless YOU resend
# the past conversation each time. This list IS the memory: every
# message (yours and the AI's) gets appended here, and the FULL list
# gets sent on every request so the AI can see what was said before.
#
# NOTE: single global list - fine for a personal local tool like this
# (one person, one browser tab). A real multi-user product would need a
# separate history per session/user.
MAX_HISTORY_MESSAGES = 20  # keep this bounded - LLMs have a context limit

# Simple on-disk persistence so conversation memory survives server restarts.
MEMORY_FILE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "conversation_history.json")
conversation_history = []


def _load_history():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                # keep only the tail that's within the token/window budget
                conversation_history.extend(data[-MAX_HISTORY_MESSAGES:])
    except FileNotFoundError:
        # first run - no history yet
        return
    except Exception as e:
        print(f"Warning: failed to load conversation history: {e}")


def _save_history():
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(
                conversation_history[-MAX_HISTORY_MESSAGES:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to save conversation history: {e}")


# load history at import time
_load_history()


def build_context_message(context):
    """
    Turns 'what the user is currently looking at' into a system message -
    this is what lets someone ask "why is it red?" instead of typing
    "Mars" every time. The frontend auto-tracks whatever's nearest the
    crosshair every frame (see updateAutoContext in index.html) and sends
    it with every chat request - so this updates even without a click.
    """
    if not context:
        return ("The user isn't pointing at any specific catalog object right now - "
                "they may be asking a general astronomy question")

    name = context.get("name", "an unknown object")
    otype = context.get("type", "object")
    mag = context.get("mag")
    az = context.get("az")
    alt = context.get("alt")
    ang_dist = context.get("angularDistance")

    mag_part = f", magnitude {mag:.2f}" if mag is not None else ""

    direction_part = ""
    if az is not None and alt is not None:
        direction_part = f" It's at azimuth {az:.1f} degrees, {alt:.1f} degrees above the horizon."

    precision_part = ""
    if ang_dist is not None:
        precision_part = f" (about {ang_dist:.1f} degrees from dead-center of where they're pointing)"

    return (f"The user is currently pointing their phone at {name} "
            f"(a {otype}{mag_part}).{direction_part}{precision_part} "
            f"Answer as if you can see this too.")


def call_llm(messages):
    """
    *** THIS IS THE SPOT ***
    Calls Groq's chat API - free tier, generous token limits, good for
    "normal conversation" since we're publishing this without asking
    end users for their own API key.

    Every LLM provider accepts basically the same shape here: a list of
    {"role": "system"|"user"|"assistant", "content": "..."} dicts, and
    returns a single string reply - that's why swapping providers later
    only means changing what's INSIDE this function.

    NOTE ON TOOL CALLING: this function is for plain conversation only.
    Once we add actual tools (functions the AI can call, like TARZ has),
    that'll be a SEPARATE function - see call_llm_with_tools() below -
    using GPT-4o-mini via GitHub Models first, Cerebras as fallback,
    since tool-calling reliability matters more there than raw speed.
    """
    if not GROQ_API_KEY:
        return ("(No AI connected) GROQ_API_KEY isn't set. Create a .env file "
                "in this folder with GROQ_API_KEY=your_key_here, then restart the server.")

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": GROQ_MODEL, "messages": messages, "max_tokens": 500},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        return "(AI request timed out - try again.)"
    except requests.exceptions.HTTPError:
        # Common causes: bad/expired key (401), rate limit (429)
        status = response.status_code
        if status == 401:
            return "(AI error: API key was rejected - check GROQ_API_KEY in .env)"
        elif status == 429:
            return "(AI error: rate limit hit - wait a moment and try again)"
        else:
            return f"(AI error: request failed with status {status})"
    except Exception as e:
        return f"(AI error: {e})"


def call_llm_with_tools(messages, tools):
    """
    *** NOT BUILT YET - SCAFFOLD FOR LATER ***
    This is where the TARZ-style routing goes once we actually define
    tools (functions) the AI can call: try GPT-4o-mini via GitHub Models
    first, fall back to Cerebras if that fails. Not implemented yet
    because there's nothing to test it with until a real tool exists -
    build the first tool, then come back and wire this up around it.
    """
    raise NotImplementedError(
        "No tools defined yet - build a tool first, then this.")


def handle_chat_message(message, context):
    """
    The full flow for one chat turn: build the system context, add it plus
    history plus the new message, call the AI, save the reply, return it.
    This is what server.py's /api/chat route calls.
    """
    system_message = {"role": "system",
                      "content": build_context_message(context)}
    conversation_history.append({"role": "user", "content": message})

    messages = [system_message] + conversation_history
    reply = call_llm(messages)

    conversation_history.append({"role": "assistant", "content": reply})

    # keep bounded and persist
    del conversation_history[:-MAX_HISTORY_MESSAGES]
    _save_history()

    return reply
