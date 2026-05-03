"""
brain_module.py — Luma's AI core
Handles all Gemini vision calls: scene description, text reading, danger assessment.
"""

import google.generativeai as genai
import PIL.Image
import os
from keys import GEMINI_API_KEY

# ── Init ──────────────────────────────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)

def _get_model():
    try:
        models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        for name in models:
            if "2.0-flash" in name or "1.5-flash" in name:
                return genai.GenerativeModel(name)
        return genai.GenerativeModel(models[0]) if models else None
    except:
        return None

gemini_model = _get_model()

def _call(prompt: str, image_path: str) -> str:
    """Internal helper — loads image and calls Gemini."""
    if not gemini_model:
        return "My brain is offline. Check your API key."
    if not os.path.exists(image_path):
        return "I cannot see anything right now."
    try:
        img = PIL.Image.open(image_path)
        response = gemini_model.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        return f"Vision error: {str(e)[:60]}"


def describe_scene(image_path: str, user_query: str = "What do you see?") -> str:
    """Mode 3 — Scene / Ask."""
    prompt = (
        "You are 'Luma', a warm and precise AI guide for a blind person. "
        f"The user just asked: '{user_query}'. "
        "Rules: "
        "1. Be conversational, clear, and concise — under 30 words unless detail is asked for. "
        "2. For object location, say left, right, center, near, or far. "
        "3. Always flag safety hazards first before answering anything else. "
        "4. If you cannot answer the question from the image, say so honestly."
    )
    return _call(prompt, image_path)


def read_text_in_scene(image_path: str) -> str:
    """Mode 4 — Text recognition (Gemini fallback)."""
    prompt = (
        "You are 'Luma', an AI assistant for a blind person. "
        "Read ALL text visible in this image — signs, labels, menus, price tags, buttons, screens. "
        "Speak it naturally as if reading aloud to someone: start with the most important or prominent text. "
        "If there is no text, say: 'I do not see any readable text right now.' "
        "Keep it under 50 words."
    )
    return _call(prompt, image_path)


def assess_danger(image_path: str) -> str:
    """Priority 0 — Gemini double-check on borderline danger situations."""
    prompt = (
        "You are a safety system for a blind pedestrian. "
        "Look at this image and answer in ONE short sentence only. "
        "Is there an immediate danger (moving vehicle, open pit, fire, aggressive animal, steep stairs)? "
        "If yes, describe it in under 10 words starting with 'DANGER:'. "
        "If no danger, reply exactly: SAFE"
    )
    result = _call(prompt, image_path)
    if result.startswith("DANGER:"):
        return result.replace("DANGER:", "").strip()
    return ""