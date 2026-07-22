"""
ai_chat.py - Conversation memory and the AI "brain" (LangGraph + LangChain edition)
----------------------------------------------------------------------
Everything about TALKING TO THE AI lives here: turning "what the user
is looking at" into context the AI can use, keeping track of the
conversation so far via a LangGraph graph, and calling an LLM through
LangChain's chat model interface.

*** The LLM is configured in one spot (see `llm = ChatGroq(...)` below).
Swapping providers later just means swapping that one object for
ChatOpenAI / ChatAnthropic / etc. - everything else, including
server.py, stays the same. ***

API KEY SETUP:
    Create a file named ".env" in this same folder with EITHER:

        GROQ_API_KEYS=key1,key2,key3,key4     (recommended - comma separated,
                                                no spaces. Rotates to the next
                                                key automatically whenever the
                                                current one hits a rate limit)
    or just:
        GROQ_API_KEY=your_single_key_here     (still works, treated as a
                                                list of one)

    This keeps keys out of your source code entirely - the app reads
    them from the environment, never hardcoded, never sent to the
    browser. Since we're providing the keys (not the end user), this is
    the only place they need to live.

Requires:
    pip install langgraph langchain langchain-groq python-dotenv
"""

import os
import json
import re
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict
from tools import STARGAZER_TOOLS

load_dotenv()  # reads .env into the environment, if the file exists
GROQ_MODEL = "llama-3.3-70b-versatile"  # same conversational model TARZ uses

# ---------------------------------------------------------------------
# API KEY ROTATION
# ---------------------------------------------------------------------
# Free-tier keys each have their own rate limit. Instead of the whole
# app breaking the moment ONE key gets rate limited, we keep a list and
# automatically switch to the next one when that happens. ChatGroq is
# built with a single key at construction time, so "rotating" means
# rebuilding it with the next key - _build_llm() below does that.
_keys_raw = os.environ.get(
    "GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY") or ""
GROQ_API_KEYS = [k.strip() for k in _keys_raw.split(",") if k.strip()]

_current_key_index = 0

MAX_HISTORY_MESSAGES = 20
THREAD_ID = "local-session"

MEMORY_FILE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "conversation_history.json")

TOOL_DISPLAY_NAMES = {
    "web_search": "web search", "nasa_articles": "NASA article search",
    "latest_discoveries": "latest discoveries", "weather_conditions": "weather check",
    "local_sky_time": "local sky-time check",
    "moon_phase": "moon phase check", "night_planner": "night planner",
    "photography_advice": "photography advice", "rise_set_times": "rise and set calculator",
    "light_pollution_report": "light-pollution report", "meteor_showers": "meteor-shower guide",
    "equipment_advice": "equipment advice", "object_information": "object lookup",
    "tle_updates": "TLE update", "satellite_status": "satellite status",
    "pass_predictions": "pass prediction", "satellite_visibility": "satellite visibility check",
}


def _clean_tool_syntax(reply):
    def replacement(match):
        tool_name = match.group(1)
        return TOOL_DISPLAY_NAMES.get(tool_name, tool_name.replace("_", " "))
    reply = re.sub(r"<function=([a-zA-Z0-9_]+)>", replacement, reply)
    reply = re.sub(r"</function>", "", reply)
    return reply


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def _build_llm(api_key):
    """Builds a fresh ChatGroq + tool-bound version using the given key."""
    llm = ChatGroq(model=GROQ_MODEL, api_key=api_key,
                   max_tokens=500, timeout=15)
    return llm, llm.bind_tools(STARGAZER_TOOLS)


def _rotate_key():
    """Switches to the next key in the list (wraps around) and rebuilds the LLM with it."""
    global _current_key_index, llm, llm_with_tools
    _current_key_index = (_current_key_index + 1) % len(GROQ_API_KEYS)
    llm, llm_with_tools = _build_llm(GROQ_API_KEYS[_current_key_index])
    print(
        f"[ai_chat] Rate limit hit - rotated to API key #{_current_key_index + 1}/{len(GROQ_API_KEYS)}")


llm = None
llm_with_tools = None
if GROQ_API_KEYS:
    llm, llm_with_tools = _build_llm(GROQ_API_KEYS[_current_key_index])


def build_context_message(context):
    tool_guidance = (
        "You are StarGazer's observing assistant. Use the provided tools whenever a user asks for "
        "current local time or whether it is day/twilight/night, current weather, moon conditions, NASA/news research, rise/set times, meteor showers, equipment "
        "advice, satellite status/visibility/passes, light-pollution assessment, or a catalog lookup. "
        "Do not invent live results. Give concise, practical answers and mention when a result is approximate."
        " Never show internal function names, tool-call tags, XML tool syntax, or instructions such as "
        "'<function=...>' to the user. Call a tool through the tool interface when needed; otherwise describe "
        "capabilities in plain language."
    )
    if not context:
        return ("The user isn't pointing at any specific catalog object right now - "
                f"they may be asking a general astronomy question. {tool_guidance}")

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
    location_part = ""
    lat = context.get("lat")
    lon = context.get("lon")
    if lat is not None and lon is not None:
        location_part = (f" The observing location is latitude {lat:.4f}, longitude {lon:.4f}. "
                         "Use these coordinates when calling weather, night planner, or satellite tools.")

    return (f"The user is currently pointing their phone at {name} "
            f"(a {otype}{mag_part}).{direction_part}{precision_part} "
            f"Answer as if you can see this too.{location_part} {tool_guidance}")


def call_model(state: ChatState):
    """
    *** THIS IS THE SPOT ***
    Calls the configured LangChain chat model with the running message
    list. On a rate-limit error, rotates to the next API key and retries
    - up to once per key, so a single request never loops forever even
    if every key happens to be exhausted at once.
    """
    if not GROQ_API_KEYS:
        reply = ("(No AI connected) No Groq API keys set. Add GROQ_API_KEYS=key1,key2,key3,key4 "
                 "to your .env file, then restart the server.")
        return {"messages": [AIMessage(content=reply)]}

    trimmed = trim_messages(
        state["messages"],
        max_tokens=MAX_HISTORY_MESSAGES,
        token_counter=len,
        strategy="last",
        start_on="human",
        include_system=True,
    )

    attempts = 0
    max_attempts = len(GROQ_API_KEYS)  # try each key at most once per request

    while attempts < max_attempts:
        try:
            response = llm_with_tools.invoke(trimmed)
            return {"messages": [response]}
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                attempts += 1
                if attempts < max_attempts:
                    _rotate_key()
                    continue
                reply = f"(AI error: all {len(GROQ_API_KEYS)} API keys are currently rate-limited - try again shortly.)"
                return {"messages": [AIMessage(content=reply)]}
            elif "401" in msg:
                reply = "(AI error: API key was rejected - check your keys in .env)"
                return {"messages": [AIMessage(content=reply)]}
            elif "timeout" in msg.lower() or "timed out" in msg.lower():
                reply = "(AI request timed out - try again.)"
                return {"messages": [AIMessage(content=reply)]}
            else:
                reply = f"(AI error: {e})"
                return {"messages": [AIMessage(content=reply)]}


_graph_builder = StateGraph(ChatState)
_graph_builder.add_node("model", call_model)
_graph_builder.add_node("tools", ToolNode(STARGAZER_TOOLS))
_graph_builder.add_edge(START, "model")
_graph_builder.add_conditional_edges("model", tools_condition)
_graph_builder.add_edge("tools", "model")

_checkpointer = MemorySaver()
graph = _graph_builder.compile(checkpointer=_checkpointer)


def _message_from_dict(d):
    role = d.get("role")
    content = d.get("content", "")
    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    return HumanMessage(content=content)


def _message_to_dict(m):
    if isinstance(m, HumanMessage):
        role = "user"
    elif isinstance(m, AIMessage):
        role = "assistant"
    elif isinstance(m, SystemMessage):
        role = "system"
    else:
        role = "user"
    return {"role": role, "content": m.content}


def _load_history():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                tail = data[-MAX_HISTORY_MESSAGES:]
                restored = [_message_from_dict(d) for d in tail]
                if restored:
                    config = {"configurable": {"thread_id": THREAD_ID}}
                    graph.update_state(config, {"messages": restored})
    except FileNotFoundError:
        return
    except Exception as e:
        print(f"Warning: failed to load conversation history: {e}")


def _save_history():
    try:
        config = {"configurable": {"thread_id": THREAD_ID}}
        snapshot = graph.get_state(config)
        messages = snapshot.values.get(
            "messages", []) if snapshot and snapshot.values else []
        serializable = [_message_to_dict(m) for m in messages
                        if isinstance(m, (HumanMessage, AIMessage))][-MAX_HISTORY_MESSAGES:]
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to save conversation history: {e}")


_load_history()


def handle_chat_message(message, context):
    system_message = SystemMessage(content=build_context_message(context))
    config = {"configurable": {"thread_id": THREAD_ID}}

    result = graph.invoke(
        {"messages": [system_message, HumanMessage(content=message)]},
        config=config,
    )

    reply = _clean_tool_syntax(result["messages"][-1].content)
    _save_history()
    return reply
