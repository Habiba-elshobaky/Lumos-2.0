

import google.generativeai as genai
import PIL.Image
import os
import platform
from keys import GEMINI_API_KEY

# ── RAM disk path (Pi) or temp folder (Windows) ───────────────────────────────
if platform.system() == "Linux" and os.path.exists("/dev/shm"):
    TEMP_SCENE = "/dev/shm/latest_scene.jpg"
    TEMP_OCR   = "/dev/shm/latest_ocr.jpg"
else:
    TEMP_SCENE = "temp_scene.jpg"
    TEMP_OCR   = "temp_ocr.jpg"

# ── Init ──────────────────────────────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)


def _get_model():
    try:
        models = [m.name for m in genai.list_models()
                  if "generateContent" in m.supported_generation_methods]
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
        img      = PIL.Image.open(image_path)
        response = gemini_model.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        return f"Vision error: {str(e)[:80]}"


# ── Public API ────────────────────────────────────────────────────────────────

def describe_scene(user_query: str = "What do you see?") -> str:
    """
    Mode 3 — Scene / Ask.
    Reads from TEMP_SCENE path (written by vision.py).
    """
    prompt = (
        "You are 'Luma', a warm and precise AI guide for a blind person. "
        f"The user just asked: '{user_query}'. "
        "Rules: "
        "1. Be conversational, clear, and concise — under 30 words unless detail is asked for. "
        "2. For object location say: left, right, center, near, or far. "
        "3. Always flag safety hazards first. "
        "4. If you cannot answer from the image, say so honestly."
    )
    return _call(prompt, TEMP_SCENE)


def read_text_in_scene() -> str:
    """
    Mode 4 — Text recognition (Gemini fallback).
    Reads from TEMP_OCR path (written by vision.py).
    """
    prompt = (
        "You are 'Luma', an AI assistant for a blind person. "
        "Read ALL text visible in this image — signs, labels, menus, price tags, buttons, screens. "
        "Speak it naturally as if reading aloud. Start with the most prominent text. "
        "If there is no text, say: 'I do not see any readable text right now.' "
        "Keep it under 50 words."
    )
    return _call(prompt, TEMP_OCR)


def assess_danger() -> str:
    """
    Gemini double-check on borderline YOLO detections.
    Returns a short danger description or empty string if safe.
    """
    prompt = (
        "You are a safety system for a blind pedestrian. "
        "Answer in ONE sentence only. "
        "Is there an immediate danger (moving vehicle, open pit, fire, aggressive animal, steep stairs)? "
        "If yes, start with 'DANGER:' and describe in under 10 words. "
        "If safe, reply exactly: SAFE"
    )
    result = _call(prompt, TEMP_SCENE)
    if result.startswith("DANGER:"):
        return result.replace("DANGER:", "").strip()
    return ""