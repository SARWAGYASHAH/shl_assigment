"""
Core conversational agent for SHL Assessment Recommender.
Handles the conversation flow: clarify -> retrieve -> recommend/refine/compare.
Supports multiple LLM backends: Gemini (google.genai), Groq.
Uses only 1 LLM call per request to minimize token usage.
"""

import json
import os
import re
import time
from typing import Optional

from app.catalog import catalog, Assessment
from app.embeddings import get_relevant_assessments, hybrid_search
from app.prompts import (
    SYSTEM_PROMPT,
    CONTEXT_EXTRACTION_PROMPT,
    build_recommendation_prompt,
    build_comparison_prompt,
)
from app.schemas import ChatRequest, ChatResponse, Recommendation, Message


# ── LLM Backend ──────────────────────────────────────────────────────────────

_llm_backend = None  # "gemini" or "groq"
_gemini_client = None
_groq_client = None


def _init_llm():
    """Initialize LLM backend. Tries Gemini first, falls back to Groq."""
    global _llm_backend, _gemini_client, _groq_client

    if _llm_backend is not None:
        return

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")

    if gemini_key:
        try:
            from google import genai
            _gemini_client = genai.Client(api_key=gemini_key)
            _llm_backend = "gemini"
            print("LLM backend: Gemini")
            return
        except Exception as e:
            print(f"Gemini init failed: {e}")

    if groq_key:
        try:
            from groq import Groq
            _groq_client = Groq(api_key=groq_key)
            _llm_backend = "groq"
            print("LLM backend: Groq")
            return
        except Exception as e:
            print(f"Groq init failed: {e}")

    raise RuntimeError(
        "No LLM API key found. Set GEMINI_API_KEY or GROQ_API_KEY. "
        "Get free keys at https://aistudio.google.com/apikey or https://console.groq.com"
    )


def _call_llm(
    messages: list[dict],
    system_instruction: str = "",
    temperature: float = 0.3,
    max_tokens: int = 600,
) -> Optional[str]:
    """Call the configured LLM backend. Returns None if all retries fail."""
    try:
        _init_llm()
    except Exception as e:
        print(f"LLM init failed: {e}")
        return None

    max_retries = 2
    for attempt in range(max_retries):
        try:
            if _llm_backend == "gemini":
                return _call_gemini(messages, system_instruction, temperature, max_tokens)
            elif _llm_backend == "groq":
                return _call_groq(messages, system_instruction, temperature, max_tokens)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str or "quota" in err_str or "resource" in err_str:
                wait = (attempt + 1) * 2  # 2s, 4s — fast retries
                print(f"Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"LLM error: {e}")
                return None

    print("LLM rate limit exceeded after retries, falling back to retrieval-only")
    return None


def _call_gemini(messages, system_instruction, temperature, max_tokens):
    """Call Gemini API."""
    from google.genai import types

    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    response = _gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction or SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text


def _call_groq(messages, system_instruction, temperature, max_tokens):
    """Call Groq API (Llama 3)."""
    groq_messages = []
    if system_instruction:
        groq_messages.append({"role": "system", "content": system_instruction})
    else:
        groq_messages.append({"role": "system", "content": SYSTEM_PROMPT})

    for msg in messages:
        groq_messages.append({"role": msg["role"], "content": msg["content"]})

    response = _groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=groq_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ── Context Extraction (heuristic-first, no LLM call) ───────────────────────

def _extract_context(messages: list[dict]) -> dict:
    """Extract context using heuristics first; LLM only if ambiguous."""
    context = _basic_context_extraction(messages)

    # Only call LLM if we really can't figure out what the user wants
    user_msgs = [m for m in messages if m["role"] == "user"]
    if len(user_msgs) >= 2 and not context["has_enough_context"]:
        try:
            conversation_text = "\n".join(
                f"{msg['role'].upper()}: {msg['content']}" for msg in messages
            )
            prompt = f"""{CONTEXT_EXTRACTION_PROMPT}
CONVERSATION:
{conversation_text}
Return ONLY the JSON object."""

            response = _call_llm(
                [{"role": "user", "content": prompt}],
                system_instruction="Extract JSON from conversation. Return only valid JSON.",
                temperature=0.1,
                max_tokens=300,
            )
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                context = json.loads(json_match.group())
        except Exception as e:
            print(f"LLM context extraction failed: {e}")

    return context


def _basic_context_extraction(messages: list[dict]) -> dict:
    """Fast heuristic context extraction -- no LLM call needed."""
    all_user_text = " ".join(
        msg["content"].lower() for msg in messages if msg["role"] == "user"
    )

    context = {
        "job_role": None,
        "seniority": None,
        "skills": [],
        "domain": None,
        "test_type_preferences": [],
        "constraints": [],
        "is_comparison_request": False,
        "comparison_items": [],
        "is_refinement": False,
        "refinement_action": None,
        "has_enough_context": False,
        "missing_info": "job role",
    }

    # Detect comparison
    if any(w in all_user_text for w in ["compare", "difference", "vs", "versus", "between"]):
        context["is_comparison_request"] = True

    # Detect refinement
    if any(w in all_user_text for w in ["actually", "also add", "remove", "instead", "change", "update"]):
        context["is_refinement"] = True

    # Detect seniority
    seniority_map = {
        "entry": "entry", "junior": "entry", "graduate": "entry", "intern": "entry", "fresher": "entry",
        "mid": "mid", "mid-level": "mid", "intermediate": "mid",
        "senior": "senior", "lead": "senior", "principal": "senior",
        "executive": "executive", "director": "executive", "vp": "executive",
    }
    for keyword, level in seniority_map.items():
        if keyword in all_user_text:
            context["seniority"] = level
            break

    # Detect domain/skills keywords
    tech_keywords = ["java", "python", "javascript", "c#", "sql", "react", "angular", "node",
                      "aws", "azure", "docker", "kubernetes", "devops", "data science",
                      "machine learning", "coding", "programming", "developer", "engineer",
                      "frontend", "backend", "fullstack", "software"]
    biz_keywords = ["sales", "marketing", "finance", "accounting", "hr", "human resources",
                     "project management", "business analysis", "supply chain", "banking",
                     "insurance", "healthcare", "retail"]
    soft_keywords = ["leadership", "communication", "teamwork", "management", "customer service",
                      "personality", "motivation", "emotional intelligence"]

    found_skills = []
    for kw in tech_keywords:
        if kw in all_user_text:
            found_skills.append(kw)
            context["domain"] = context["domain"] or "technology"
    for kw in biz_keywords:
        if kw in all_user_text:
            found_skills.append(kw)
            context["domain"] = context["domain"] or "business"
    for kw in soft_keywords:
        if kw in all_user_text:
            found_skills.append(kw)

    context["skills"] = found_skills

    # Detect test type preferences
    if any(w in all_user_text for w in ["personality", "behavioral", "behaviour", "work style"]):
        context["test_type_preferences"].append("P")
    if any(w in all_user_text for w in ["cognitive", "ability", "reasoning", "aptitude", "numerical", "verbal"]):
        context["test_type_preferences"].append("A")
    if any(w in all_user_text for w in ["knowledge", "technical test", "domain test"]):
        context["test_type_preferences"].append("K")
    if any(w in all_user_text for w in ["simulation", "coding test", "hands-on", "practical"]):
        context["test_type_preferences"].append("S")
    if any(w in all_user_text for w in ["situational", "judgment", "sjt", "scenarios"]):
        context["test_type_preferences"].append("B")

    # Determine job role from user text
    user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
    if user_messages:
        latest = user_messages[-1]
        all_combined = " ".join(user_messages)
        # If any user message has 5+ words, we likely have enough context
        if any(len(m.split()) >= 5 for m in user_messages):
            context["has_enough_context"] = True
            context["job_role"] = all_combined
        elif found_skills:
            context["has_enough_context"] = True
            context["job_role"] = all_combined

    return context


# ── Guardrails ───────────────────────────────────────────────────────────────

def _is_off_topic(message: str) -> bool:
    """Check if a message is off-topic."""
    off_topic_patterns = [
        r"\b(salary|pay|compensation|benefits)\b",
        r"\b(legal|lawsuit|sue|lawyer|attorney)\b",
        r"\b(ignore previous|forget your|you are now|act as|pretend)\b",
        r"\b(write me a|generate code|help me with homework)\b",
        r"\b(weather|news|sports|movie|recipe|joke)\b",
    ]
    message_lower = message.lower()
    shl_words = ["shl", "assessment", "test", "hiring", "recruit", "candidate", "evaluate"]
    if any(w in message_lower for w in shl_words):
        return False
    for pattern in off_topic_patterns:
        if re.search(pattern, message_lower):
            return True
    return False


def _is_vague_first_turn(messages: list[dict]) -> bool:
    """Check if this is a vague first user message."""
    user_messages = [m for m in messages if m["role"] == "user"]
    if len(user_messages) != 1:
        return False
    first_msg = user_messages[0]["content"].strip().lower()
    if len(first_msg.split()) <= 4:
        vague = ["i need an assessment", "help me", "i need a test", "recommend something",
                 "what do you have", "show me assessments", "i need help", "assessment",
                 "test", "hello", "hi"]
        return any(first_msg.startswith(p) or first_msg == p for p in vague)
    return False


# ── Recommendation Extraction ───────────────────────────────────────────────

def _extract_recommendations_from_response(reply: str, context: dict) -> list[Recommendation]:
    """Extract structured recommendations from the LLM's text response."""
    recommendations = []
    seen_names = set()

    # Find assessment names mentioned in the reply
    reply_lower = reply.lower()
    for assessment in catalog.assessments:
        if assessment.name.lower() in reply_lower:
            if assessment.name not in seen_names:
                seen_names.add(assessment.name)
                recommendations.append(Recommendation(
                    name=assessment.name, url=assessment.url, test_type=assessment.test_type,
                ))

    if recommendations:
        return recommendations[:10]

    # Fallback: use retrieval results
    search_parts = []
    if context.get("job_role"):
        search_parts.append(str(context["job_role"]))
    if context.get("skills"):
        search_parts.extend([str(s) for s in context["skills"]])
    search_query = " ".join(search_parts) if search_parts else reply[:200]

    if search_query.strip():
        test_types = context.get("test_type_preferences", [])
        retrieved = get_relevant_assessments(
            search_query,
            test_types=test_types if test_types else None,
            top_k=10,
        )
        for a in retrieved:
            if a.name not in seen_names:
                seen_names.add(a.name)
                recommendations.append(Recommendation(
                    name=a.name, url=a.url, test_type=a.test_type,
                ))

    return recommendations[:10]


# ── Main Entry Point ────────────────────────────────────────────────────────

async def process_chat(request: ChatRequest) -> ChatResponse:
    """Process a chat request. Only 1 LLM call per request."""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if not messages:
        return ChatResponse(
            reply="Hello! I'm the SHL Assessment Recommender. Tell me about the role you're hiring for, and I'll help you find the right assessments.",
            recommendations=[], end_of_conversation=False,
        )

    last_user_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break

    # Guardrails (no LLM needed)
    if _is_off_topic(last_user_msg):
        return ChatResponse(
            reply="I can only help with SHL assessment recommendations. Could you tell me about the role you're hiring for?",
            recommendations=[], end_of_conversation=False,
        )

    if _is_vague_first_turn(messages):
        return ChatResponse(
            reply="I'd love to help! To recommend the right SHL assessments, could you tell me:\n\n1. **What role** are you hiring for?\n2. **What seniority level?** (entry, mid, senior)\n3. **Any specific skills** you want to assess?",
            recommendations=[], end_of_conversation=False,
        )

    # Extract context (heuristic-first, mostly no LLM call)
    context = _extract_context(messages)

    # Handle comparison
    if context.get("is_comparison_request"):
        return await _handle_comparison(messages, context)

    # Not enough context? Use LLM to ask clarifying questions
    if not context.get("has_enough_context", False):
        clarification = _call_llm(messages, temperature=0.4, max_tokens=250)
        if not clarification:
            clarification = "To give you the best recommendations, could you tell me more about the role? For example: what's the job title, seniority level, and key skills you need to assess?"
        return ChatResponse(reply=clarification, recommendations=[], end_of_conversation=False)

    # We have context -- make 1 LLM call for recommendations
    return await _handle_recommendation(messages, context)


async def _handle_recommendation(messages: list[dict], context: dict) -> ChatResponse:
    """Generate recommendations with a single LLM call."""
    # Build search query
    search_parts = []
    if context.get("job_role"):
        search_parts.append(str(context["job_role"]))
    if context.get("skills"):
        search_parts.extend([str(s) for s in context["skills"]])
    if context.get("domain"):
        search_parts.append(str(context["domain"]))
    if context.get("seniority"):
        search_parts.append(str(context["seniority"]))

    search_query = " ".join(search_parts) if search_parts else messages[-1]["content"]

    # Retrieve relevant assessments
    test_types = context.get("test_type_preferences", [])
    retrieved = get_relevant_assessments(
        search_query,
        test_types=test_types if test_types else None,
        top_k=15,
    )

    assessment_data = [a._raw for a in retrieved]
    prev_recs = None
    if context.get("is_refinement"):
        prev_recs = []
        for msg in messages:
            if msg["role"] == "assistant":
                for a in catalog.assessments:
                    if a.name.lower() in msg["content"].lower():
                        prev_recs.append(a.name)

    rec_prompt = build_recommendation_prompt(assessment_data, context, prev_recs)

    # Single LLM call: augment conversation with retrieval context
    augmented = messages.copy()
    augmented.append({"role": "user", "content": rec_prompt})

    reply = _call_llm(augmented, temperature=0.3, max_tokens=600)

    if reply:
        recommendations = _extract_recommendations_from_response(reply, context)
    else:
        recommendations = []

    # If LLM failed or didn't mention assessments, use retrieval results directly
    if not recommendations and retrieved:
        recommendations = [
            Recommendation(name=a.name, url=a.url, test_type=a.test_type)
            for a in retrieved[:10]
        ]

    # Generate template reply if LLM failed
    if not reply:
        role = context.get("job_role", "the specified role")
        rec_list = "\n".join(f"- **{r.name}** [{r.test_type}] - {r.url}" for r in recommendations[:10])
        reply = f"Based on your requirements for {role}, here are my recommended SHL assessments:\n\n{rec_list}"

    turn_count = len(messages) + 1
    end_of_conversation = turn_count >= 7 and len(recommendations) > 0

    return ChatResponse(
        reply=reply, recommendations=recommendations[:10],
        end_of_conversation=end_of_conversation,
    )


async def _handle_comparison(messages: list[dict], context: dict) -> ChatResponse:
    """Handle comparison requests."""
    comparison_items = context.get("comparison_items", [])
    last_msg = messages[-1]["content"] if messages else ""

    found = []
    for item in comparison_items:
        a = catalog.get_by_name(str(item))
        if a:
            found.append(a)
    for a in catalog.assessments:
        if a.name.lower() in last_msg.lower() and a not in found:
            found.append(a)

    if len(found) >= 2:
        comparison_prompt = build_comparison_prompt([a._raw for a in found])
        augmented = messages.copy()
        augmented.append({"role": "user", "content": comparison_prompt})
        reply = _call_llm(augmented, temperature=0.3, max_tokens=500)
        if not reply:
            lines = []
            for a in found:
                lines.append(f"**{a.name}** [{a.test_type}] - {a.category}, {a.duration}\n  {a.description[:150]}")
            reply = "Here's a comparison of the assessments:\n\n" + "\n\n".join(lines)
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)
    else:
        reply = _call_llm(messages, temperature=0.4, max_tokens=250)
        if not reply:
            reply = "Which assessments would you like to compare? Please mention them by name."
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)
