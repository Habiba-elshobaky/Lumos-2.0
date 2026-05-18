"""
vision.py — LUMOS Assistive AI v2.1
=====================================
Modes (switch by voice or key):
  W → Walking mode    (default, silent unless danger)
  N → Navigation mode (turn-by-turn with ORS)
  T → Text mode       (OCR: read signs, menus, labels)
  S → Scene mode      (Gemini describes the environment)

Always-on:
  EMERGENCY OVERRIDE  (vehicles < danger distance)
  CROWD ALERT         (5+ people in center zone)

Architecture (per inst.md):
  - All audio goes via server.send_event(SpeakEvent) — no gTTS here
  - All temp images saved to /dev/shm/ RAM disk (Pi) or local (Windows)
  - GPS comes from server.get_gps() — phone sends it via WebSocket
  - No time.sleep() in the main loop — cooldown logic only
  - Events throttled: only sent on state change or every 2 seconds
  - Edit hazards in hazards.py — press R to reload live
"""

import cv2
import threading
import time
import os
import re
import platform

import speech_recognition as sr
from ultralytics import YOLO

import brain_module as luma_brain
import ORS as navigator
from hazards import (
    DANGER_OBJECTS,
    TRIP_HAZARDS,
    DANGER_DISTANCE,
    HAZARD_DISTANCE,
    HAZARD_COOLDOWN,
    CROWD_THRESHOLD,
)
from nova_network_models import (
    SocialAlertEvent,
    SpeakEvent,
    OCREvent,
    SceneEvent,
    NavigationEvent,
    ModeChangeEvent,
)
from lumos_server import server

# ── RAM disk path (Pi) or local (Windows/Mac) ─────────────────────────────────
if platform.system() == "Linux" and os.path.exists("/dev/shm"):
    TEMP_SCENE = "/dev/shm/latest_scene.jpg"
    TEMP_OCR   = "/dev/shm/latest_ocr.jpg"
else:
    TEMP_SCENE = "temp_scene.jpg"
    TEMP_OCR   = "temp_ocr.jpg"

# ── Tesseract (optional — falls back to Gemini) ───────────────────────────────
try:
    import pytesseract
    from PIL import Image as PILImage
    TESSERACT_AVAILABLE = True
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    TESSERACT_AVAILABLE = False
    print(">>> [INFO] pytesseract not installed — OCR will use Gemini only.")

# ── Constants ─────────────────────────────────────────────────────────────────
ALERT_COOLDOWN         = HAZARD_COOLDOWN
CROWD_PERSON_THRESHOLD = CROWD_THRESHOLD
FOCAL_LENGTH           = 160
REAL_HEIGHT_PERSON     = 170   # cm
CAIRO_FALLBACK         = (31.2357, 30.0444)   # replace with your city coords


# ─────────────────────────────────────────────────────────────────────────────
# SPEAK HELPER — sends to server, never calls gTTS directly
# ─────────────────────────────────────────────────────────────────────────────
def speak(text: str, emergency: bool = False):
    """Send text to server to be spoken on the phone (or locally via stub)."""
    print(f">>> [LUMA]: {text}")
    event = SpeakEvent.create(text, emergency)
    server.send_event(event)


def play_beep(freq: int = 1000, duration: int = 200):
    """Cross-platform beep — only used for local feedback on Pi/PC screen."""
    try:
        import pygame
        import numpy as np
        sample_rate = 44100
        t    = np.linspace(0, duration / 1000, int(sample_rate * duration / 1000), False)
        wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        wave = np.column_stack([wave, wave])
        sound = pygame.sndarray.make_sound(wave)
        sound.play()
    except Exception:
        try:
            import winsound
            winsound.Beep(freq, duration)
        except:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# MODES
# ─────────────────────────────────────────────────────────────────────────────
MODE_WALKING    = "walking"
MODE_NAVIGATION = "navigation"
MODE_TEXT       = "text"
MODE_SCENE      = "scene"

current_mode = MODE_WALKING

MODE_KEYWORDS = {
    MODE_WALKING:    ["walking mode", "walk mode", "walking"],
    MODE_NAVIGATION: ["navigation mode", "navigate", "directions", "take me to", "go to"],
    MODE_TEXT:       ["text mode", "read mode", "read this", "what does it say"],
    MODE_SCENE:      ["scene mode", "describe", "what is around me", "look around"],
}

MODE_CONFIRMATIONS = {
    MODE_WALKING:    "Walking mode. I will only speak if there is danger.",
    MODE_NAVIGATION: "Navigation mode. Tell me your destination.",
    MODE_TEXT:       "Text mode. Point the camera at any text and I will read it.",
    MODE_SCENE:      "Scene mode. Ask me anything about your surroundings.",
}


def detect_mode_from_speech(text: str) -> str | None:
    text = text.lower()
    for mode, keywords in MODE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return mode
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────
emergency_active      = False
last_safety_alert     = ""
last_alert_time       = 0.0
last_event_time       = 0.0     # throttle: min 2 s between social alerts
nav_steps: list[str]  = []
nav_step_index        = 0


# ─────────────────────────────────────────────────────────────────────────────
# GPS — from server (phone WebSocket)
# ─────────────────────────────────────────────────────────────────────────────
def get_current_coords() -> tuple:
    """
    Returns (lon, lat) from phone GPS via server.
    Falls back to Cairo coords if not available.
    """
    coords = server.get_gps()
    if coords:
        return coords
    print(">>> [GPS]: No phone GPS yet, using fallback coords.")
    return CAIRO_FALLBACK


# ─────────────────────────────────────────────────────────────────────────────
# DISTANCE ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────
def estimate_distance(box_y1: float, box_y2: float) -> float:
    pixel_height = max(box_y2 - box_y1, 1)
    return (REAL_HEIGHT_PERSON * FOCAL_LENGTH) / (pixel_height * 100)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT RECOGNITION
# ─────────────────────────────────────────────────────────────────────────────
def read_text_from_frame(frame) -> str:
    """Two-stage OCR: Tesseract first, Gemini fallback."""
    cv2.imwrite(TEMP_OCR, frame)

    if TESSERACT_AVAILABLE:
        try:
            img     = PILImage.open(TEMP_OCR)
            img     = img.resize((img.width * 2, img.height * 2), PILImage.LANCZOS)
            raw     = pytesseract.image_to_string(img, config="--psm 6")
            cleaned = " ".join(raw.split())
            if len(cleaned) > 8:
                return f"I can see the following text: {cleaned}"
        except Exception as e:
            print(f">>> [OCR ERROR]: {e}")

    return luma_brain.read_text_in_scene()


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
def start_navigation(destination: str):
    global nav_steps, nav_step_index
    speak(f"Getting directions to {destination}. One moment.")
    coords = get_current_coords()
    steps  = navigator.navigate_to_place(destination, coords)
    if steps:
        nav_steps      = steps
        nav_step_index = 0
        event = NavigationEvent.create(steps[0], 0, len(steps))
        server.send_event(event)
        speak(f"Route found. {len(steps)} steps. {steps[0]}")
    else:
        speak(f"Sorry, I could not find directions to {destination}.")


def advance_nav_step():
    global nav_step_index
    if nav_step_index < len(nav_steps) - 1:
        nav_step_index += 1
        step  = nav_steps[nav_step_index]
        event = NavigationEvent.create(step, nav_step_index, len(nav_steps))
        server.send_event(event)
        speak(step)
    else:
        speak("You have arrived at your destination.")
        nav_steps.clear()


# ─────────────────────────────────────────────────────────────────────────────
# LIVE HAZARD RELOAD
# ─────────────────────────────────────────────────────────────────────────────
def reload_hazards():
    global DANGER_OBJECTS, TRIP_HAZARDS, DANGER_DISTANCE
    global HAZARD_DISTANCE, ALERT_COOLDOWN, CROWD_PERSON_THRESHOLD
    try:
        import hazards
        from importlib import reload
        reload(hazards)
        DANGER_OBJECTS         = hazards.DANGER_OBJECTS
        TRIP_HAZARDS           = hazards.TRIP_HAZARDS
        DANGER_DISTANCE        = hazards.DANGER_DISTANCE
        HAZARD_DISTANCE        = hazards.HAZARD_DISTANCE
        ALERT_COOLDOWN         = hazards.HAZARD_COOLDOWN
        CROWD_PERSON_THRESHOLD = hazards.CROWD_THRESHOLD
        print(">>> [LUMOS]: hazards.py reloaded.")
        speak("Hazard list updated.")
    except Exception as e:
        print(f">>> [RELOAD ERROR]: {e}")
        speak("Could not reload hazards.")


# ─────────────────────────────────────────────────────────────────────────────
# VOICE LISTENER
# ─────────────────────────────────────────────────────────────────────────────
def listen_and_respond(frame_snapshot):
    global current_mode, nav_steps

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300

    with sr.Microphone() as source:
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            play_beep(1200, 150)
            print(">>> [LUMA]: Listening...")
            audio   = recognizer.listen(source, timeout=6, phrase_time_limit=6)
            command = recognizer.recognize_google(audio).lower()
            print(f">>> [USER]: {command}")
        except Exception as e:
            print(f">>> [MIC]: {e}")
            if not emergency_active:
                speak("I did not catch that. Press space and try again.")
            return

    if emergency_active:
        return

    # ── Mode switch ───────────────────────────────────────────────────────────
    new_mode = detect_mode_from_speech(command)
    if new_mode and new_mode != current_mode:
        current_mode = new_mode
        server.send_event(ModeChangeEvent.create(new_mode))
        speak(MODE_CONFIRMATIONS[new_mode])
        return

    # ── Navigation destination ────────────────────────────────────────────────
    if current_mode == MODE_NAVIGATION or "take me to" in command or "navigate to" in command:
        match = re.search(r"(?:take me to|navigate to|go to|directions to)\s+(.+)", command)
        if match:
            dest = match.group(1).strip()
            threading.Thread(target=start_navigation, args=(dest,), daemon=True).start()
            return
        elif nav_steps:
            advance_nav_step()
            return

    # ── Where am I? ───────────────────────────────────────────────────────────
    if "where am i" in command or "my location" in command:
        coords = server.get_gps()
        if coords:
            addr = navigator.reverse_geocode(coords[0], coords[1])
            speak(f"You are near {addr}.")
        else:
            speak("I cannot determine your location right now.")
        return

    # ── Next nav step ─────────────────────────────────────────────────────────
    if command in ("next", "next step", "continue") and nav_steps:
        advance_nav_step()
        return

    # ── Mode-specific AI ──────────────────────────────────────────────────────
    cv2.imwrite(TEMP_SCENE, frame_snapshot)

    if current_mode == MODE_TEXT:
        def _ocr_and_send():
            result = read_text_from_frame(frame_snapshot)
            server.send_event(OCREvent.create(result))
            speak(result)
        threading.Thread(target=_ocr_and_send, daemon=True).start()

    else:
        def _scene_and_send():
            answer = luma_brain.describe_scene(command)
            server.send_event(SceneEvent.create(command, answer))
            speak(answer)
        threading.Thread(target=_scene_and_send, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# AUTO OCR
# ─────────────────────────────────────────────────────────────────────────────
def _auto_read(frame):
    result = read_text_from_frame(frame)
    if "do not see any readable text" not in result:
        server.send_event(OCREvent.create(result))
        speak(result)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global current_mode, emergency_active
    global last_safety_alert, last_alert_time, last_event_time

    model = YOLO("yolo11n.pt")
    cap   = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(">>> [ERROR] Cannot open camera.")
        return

    speak(MODE_CONFIRMATIONS[MODE_WALKING])
    print("\n═══════════════════════════════════════════════")
    print("  LUMOS v2.1 — Assistive AI for the visually impaired")
    print("  Keys: W=Walk  N=Navigate  T=Text  S=Scene")
    print("        SPACE=Listen  M=Next nav step")
    print("        R=Reload hazards.py  Q=Quit")
    print("═══════════════════════════════════════════════\n")

    _last_ocr_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now      = time.time()
        h, w, _  = frame.shape
        key      = cv2.waitKey(1) & 0xFF

        # ── Key handlers ──────────────────────────────────────────────────────
        if key == ord("q"):
            speak("Goodbye. Stay safe.")
            break

        elif key == ord("w") and current_mode != MODE_WALKING:
            current_mode = MODE_WALKING
            server.send_event(ModeChangeEvent.create(MODE_WALKING))
            speak(MODE_CONFIRMATIONS[MODE_WALKING])

        elif key == ord("n") and current_mode != MODE_NAVIGATION:
            current_mode = MODE_NAVIGATION
            server.send_event(ModeChangeEvent.create(MODE_NAVIGATION))
            speak(MODE_CONFIRMATIONS[MODE_NAVIGATION])

        elif key == ord("t") and current_mode != MODE_TEXT:
            current_mode = MODE_TEXT
            server.send_event(ModeChangeEvent.create(MODE_TEXT))
            speak(MODE_CONFIRMATIONS[MODE_TEXT])
            frame_copy = frame.copy()
            threading.Thread(
                target=lambda: _auto_read(frame_copy),
                daemon=True
            ).start()

        elif key == ord("s") and current_mode != MODE_SCENE:
            current_mode = MODE_SCENE
            server.send_event(ModeChangeEvent.create(MODE_SCENE))
            speak(MODE_CONFIRMATIONS[MODE_SCENE])

        elif key in (ord(" "), ord("v")):
            threading.Thread(
                target=listen_and_respond,
                args=(frame.copy(),),
                daemon=True
            ).start()

        elif key == ord("m") and nav_steps:
            advance_nav_step()

        elif key == ord("r"):
            threading.Thread(target=reload_hazards, daemon=True).start()

        # ═════════════════════════════════════════════════════════════════════
        # YOLO INFERENCE
        # ═════════════════════════════════════════════════════════════════════
        results = list(model.predict(frame, conf=0.35, stream=True, verbose=False))

        # ── Priority 0: Emergency ─────────────────────────────────────────────
        current_emergency   = False
        person_count_center = 0

        for r in results:
            for box in r.boxes:
                label        = model.names[int(box.cls[0])]
                x1, y1, x2, y2 = box.xyxy[0]
                dist         = estimate_distance(float(y1), float(y2))
                center_x     = (float(x1) + float(x2)) / 2

                if label in DANGER_OBJECTS and dist < DANGER_DISTANCE:
                    current_emergency = True
                    emergency_active  = True

                    # Send event to server (throttled to avoid flooding)
                    if now - last_event_time > 2.0:
                        payload = {
                            "label":   label,
                            "distance": round(dist, 1),
                            "urgency": "critical",
                        }
                        server.send_event(
                            SocialAlertEvent.create("OBJECT_DETECTION", payload)
                        )
                        play_beep(2500, 300)
                        speak(f"DANGER! {label} very close!", emergency=True)
                        last_event_time = now
                    break

                if label == "person" and (w / 3 < center_x < 2 * w / 3):
                    person_count_center += 1

            if current_emergency:
                break

        # ── Crowd alert ───────────────────────────────────────────────────────
        if not current_emergency and person_count_center >= CROWD_PERSON_THRESHOLD:
            if now - last_alert_time > ALERT_COOLDOWN * 2:
                payload = {
                    "label":   "crowd",
                    "count":   person_count_center,
                    "urgency": "medium",
                }
                server.send_event(
                    SocialAlertEvent.create("CROWD_ALERT", payload)
                )
                play_beep(900, 200)
                speak("Crowded area ahead. Move carefully.")
                last_alert_time = now

        if current_emergency:
            annotated = results[0].plot() if results else frame
            cv2.putText(annotated, "EMERGENCY", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
            cv2.imshow("LUMOS v2.1", annotated)
            continue

        emergency_active = False

        # ── Priority 1: Walking hazards ───────────────────────────────────────
        annotated_frame = results[0].plot() if results else frame

        for r in results:
            for box in r.boxes:
                label        = model.names[int(box.cls[0])]
                x1, y1, x2, y2 = box.xyxy[0]
                dist         = estimate_distance(float(y1), float(y2))
                center_x     = (float(x1) + float(x2)) / 2
                in_path      = w / 3 < center_x < 2 * w / 3

                if label in TRIP_HAZARDS and dist < HAZARD_DISTANCE and in_path:
                    # Only warn if label changed OR cooldown passed
                    if label != last_safety_alert or now - last_alert_time > ALERT_COOLDOWN:
                        payload = {
                            "label":    label,
                            "distance": round(dist, 1),
                            "urgency":  "low",
                        }
                        server.send_event(
                            SocialAlertEvent.create("OBJECT_DETECTION", payload)
                        )
                        play_beep(800, 200)
                        speak(f"Caution. {label} ahead.")
                        last_safety_alert = label
                        last_alert_time   = now

                elif dist > HAZARD_DISTANCE + 0.7:
                    if last_safety_alert == label:
                        last_safety_alert = ""

        # ── Auto OCR every 5 s in text mode ──────────────────────────────────
        if current_mode == MODE_TEXT and now - _last_ocr_time > 5.0:
            _last_ocr_time = now
            frame_copy     = frame.copy()
            threading.Thread(
                target=lambda: _auto_read(frame_copy),
                daemon=True
            ).start()

        # ── Save latest frame to RAM disk for Gemini ──────────────────────────
        cv2.imwrite(TEMP_SCENE, frame)

        # ── HUD overlay ───────────────────────────────────────────────────────
        cv2.putText(annotated_frame, f"Mode: {current_mode.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)

        status = "Phone: Connected" if server.connected else "Phone: Waiting..."
        cv2.putText(annotated_frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0) if server.connected else (100, 100, 255), 1)

        cv2.putText(annotated_frame, "R=Reload hazards | SPACE=Listen | Q=Quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        if nav_steps and nav_step_index < len(nav_steps):
            cv2.putText(annotated_frame, nav_steps[nav_step_index][:60],
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("LUMOS v2.1", annotated_frame)

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()