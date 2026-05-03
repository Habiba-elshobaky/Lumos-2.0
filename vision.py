"""
vision.py — LUMOS Assistive AI v2.0
=====================================
Modes (switch by voice or key):
  W → Walking mode    (default, silent unless danger)
  N → Navigation mode (turn-by-turn with ORS)
  T → Text mode       (OCR: read signs, menus, labels)
  S → Scene mode      (Gemini describes the environment)

Always-on:
  EMERGENCY OVERRIDE  (vehicles < 4 m)
  CROWD ALERT         (5+ people in center zone)
"""

import cv2
import threading
import time
import os
import queue
import re

import pygame
import speech_recognition as sr
from gtts import gTTS
from ultralytics import YOLO

import brain_module as luma_brain
import ORS as navigator

# ── Tesseract (optional — falls back to Gemini if not installed) ──────────────
try:
    import pytesseract
    from PIL import Image as PILImage
    TESSERACT_AVAILABLE = True
    # Windows: uncomment and set your path if needed:
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    TESSERACT_AVAILABLE = False
    print(">>> [INFO] pytesseract not installed — OCR will use Gemini only.")

# ── GPS (optional) ────────────────────────────────────────────────────────────
try:
    import geocoder as gps_lib
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False
    print(">>> [INFO] geocoder not installed — GPS features disabled.")

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO
# ─────────────────────────────────────────────────────────────────────────────
pygame.mixer.init()
_speech_queue: queue.Queue = queue.Queue()
_tts_lock = threading.Lock()


def _tts_worker():
    while True:
        text, emergency = _speech_queue.get()
        _do_speak(text, emergency)
        _speech_queue.task_done()


threading.Thread(target=_tts_worker, daemon=True).start()


def _do_speak(text: str, emergency: bool = False):
    with _tts_lock:
        try:
            if not text.strip():
                return
            if emergency:
                pygame.mixer.music.stop()
            print(f">>> [LUMA]: {text}")
            tts = gTTS(text=text, lang="en")
            fname = f"luma_voice_{threading.get_ident()}.mp3"
            tts.save(fname)
            pygame.mixer.music.load(fname)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.music.unload()
            try:
                os.remove(fname)
            except:
                pass
        except Exception as e:
            print(f">>> [VOICE ERROR]: {e}")


def speak(text: str, emergency: bool = False):
    """Non-blocking: pushes text onto the speech queue."""
    if emergency:
        while not _speech_queue.empty():
            try:
                _speech_queue.get_nowait()
            except:
                pass
    _speech_queue.put((text, emergency))


def play_beep(freq: int = 1000, duration: int = 200):
    """Cross-platform beep using pygame + numpy."""
    try:
        import numpy as np
        sample_rate = 44100
        t = np.linspace(0, duration / 1000, int(sample_rate * duration / 1000), False)
        wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        wave = np.column_stack([wave, wave])
        sound = pygame.sndarray.make_sound(wave)
        sound.play()
        time.sleep(duration / 1000)
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
# DETECTION CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FOCAL_LENGTH       = 160
REAL_HEIGHT_PERSON = 170  # cm

DANGER_OBJECTS = {"car", "truck", "bus", "motorcycle", "bicycle", "fire hydrant"}

TRIP_HAZARDS = {
    "chair", "bench", "potted plant", "suitcase", "backpack",
    "box", "person", "dog", "cat", "stroller", "shopping cart",
    "stairs", "step"
}

CROWD_PERSON_THRESHOLD = 5

# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────
emergency_active  = False
last_safety_alert = ""
last_alert_time   = 0.0
ALERT_COOLDOWN    = 4.0
nav_steps: list[str] = []
nav_step_index    = 0
pending_speech    = ""


# ─────────────────────────────────────────────────────────────────────────────
# GPS
# ─────────────────────────────────────────────────────────────────────────────
def get_current_coords() -> tuple | None:
    if not GPS_AVAILABLE:
        return None
    try:
        g = gps_lib.ip("me")
        if g and g.lnglat:
            return g.lnglat
    except:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TEXT RECOGNITION
# ─────────────────────────────────────────────────────────────────────────────
def read_text_from_frame(frame) -> str:
    cv2.imwrite("temp_ocr.jpg", frame)

    if TESSERACT_AVAILABLE:
        try:
            img = PILImage.open("temp_ocr.jpg")
            img = img.resize((img.width * 2, img.height * 2), PILImage.LANCZOS)
            raw = pytesseract.image_to_string(img, config="--psm 6")
            cleaned = " ".join(raw.split())
            if len(cleaned) > 8:
                return f"I can see the following text: {cleaned}"
        except Exception as e:
            print(f">>> [OCR ERROR]: {e}")

    return luma_brain.read_text_in_scene("temp_ocr.jpg")


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
def start_navigation(destination: str):
    global nav_steps, nav_step_index
    speak(f"Getting directions to {destination}. One moment.")
    coords = get_current_coords()
    if not coords:
        coords = (31.2357, 30.0444)  # Cairo fallback — replace with your city
        speak("I could not get your exact location. Using approximate position.")
    steps = navigator.navigate_to_place(destination, coords)
    if steps:
        nav_steps      = steps
        nav_step_index = 0
        speak(f"Route found. {len(steps)} steps. " + steps[0])
    else:
        speak(f"Sorry, I could not find directions to {destination}.")


def advance_nav_step():
    global nav_step_index
    if nav_step_index < len(nav_steps) - 1:
        nav_step_index += 1
        speak(nav_steps[nav_step_index])
    else:
        speak("You have arrived at your destination.")
        nav_steps.clear()


# ─────────────────────────────────────────────────────────────────────────────
# VOICE LISTENER
# ─────────────────────────────────────────────────────────────────────────────
def listen_and_respond(frame_snapshot):
    global current_mode, pending_speech, nav_steps

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300

    with sr.Microphone() as source:
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            play_beep(1200, 150)
            print(">>> [LUMA]: Listening...")
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=6)
            command = recognizer.recognize_google(audio).lower()
            print(f">>> [USER]: {command}")
        except Exception as e:
            print(f">>> [MIC]: {e}")
            if not emergency_active:
                pending_speech = "I did not catch that. Press space and try again."
            return

    if emergency_active:
        return

    # Mode switch?
    new_mode = detect_mode_from_speech(command)
    if new_mode and new_mode != current_mode:
        current_mode = new_mode
        pending_speech = MODE_CONFIRMATIONS[new_mode]
        return

    # Navigation destination
    if current_mode == MODE_NAVIGATION or "take me to" in command or "navigate to" in command:
        match = re.search(r"(?:take me to|navigate to|go to|directions to)\s+(.+)", command)
        if match:
            dest = match.group(1).strip()
            threading.Thread(target=start_navigation, args=(dest,), daemon=True).start()
            return
        elif nav_steps:
            advance_nav_step()
            return

    # Where am I?
    if "where am i" in command or "my location" in command:
        coords = get_current_coords()
        if coords:
            addr = navigator.reverse_geocode(coords[0], coords[1])
            pending_speech = f"You are near {addr}."
        else:
            pending_speech = "I cannot determine your location right now."
        return

    # Next nav step
    if command in ("next", "next step", "continue") and nav_steps:
        advance_nav_step()
        return

    # Mode-specific AI
    cv2.imwrite("temp_scene.jpg", frame_snapshot)

    if current_mode == MODE_TEXT:
        pending_speech = read_text_from_frame(frame_snapshot)
    else:
        pending_speech = luma_brain.describe_scene("temp_scene.jpg", command)


# ─────────────────────────────────────────────────────────────────────────────
# DISTANCE ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────
def estimate_distance(box_y1, box_y2) -> float:
    pixel_height = max(box_y2 - box_y1, 1)
    return (REAL_HEIGHT_PERSON * FOCAL_LENGTH) / (pixel_height * 100)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global current_mode, emergency_active, last_safety_alert, last_alert_time, pending_speech

    model = YOLO("yolo11n.pt")
    cap   = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(">>> [ERROR] Cannot open camera.")
        return

    speak(MODE_CONFIRMATIONS[MODE_WALKING])
    print("\n═══════════════════════════════════════════════")
    print("  LUMOS v2.0 — Assistive AI for the visually impaired")
    print("  Keys: W=Walk  N=Navigate  T=Text  S=Scene")
    print("        SPACE=Listen  M=Next nav step  Q=Quit")
    print("═══════════════════════════════════════════════\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        key      = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            speak("Goodbye. Stay safe.")
            break

        if key == ord("w") and current_mode != MODE_WALKING:
            current_mode = MODE_WALKING
            speak(MODE_CONFIRMATIONS[MODE_WALKING])

        elif key == ord("n") and current_mode != MODE_NAVIGATION:
            current_mode = MODE_NAVIGATION
            speak(MODE_CONFIRMATIONS[MODE_NAVIGATION])

        elif key == ord("t") and current_mode != MODE_TEXT:
            current_mode = MODE_TEXT
            speak(MODE_CONFIRMATIONS[MODE_TEXT])
            pending_speech = read_text_from_frame(frame)

        elif key == ord("s") and current_mode != MODE_SCENE:
            current_mode = MODE_SCENE
            speak(MODE_CONFIRMATIONS[MODE_SCENE])

        elif key in (ord(" "), ord("v")):
            threading.Thread(
                target=listen_and_respond,
                args=(frame.copy(),),
                daemon=True
            ).start()

        elif key == ord("m") and nav_steps:
            advance_nav_step()

        # ── YOLO inference ───────────────────────────────────────────────────
        results = list(model.predict(frame, conf=0.35, stream=True, verbose=False))

        # Priority 0: Emergency
        current_emergency   = False
        person_count_center = 0

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                x1, y1, x2, y2 = box.xyxy[0]
                dist = estimate_distance(float(y1), float(y2))

                if label in DANGER_OBJECTS and dist < 4.0:
                    current_emergency = True
                    emergency_active  = True
                    pending_speech    = ""
                    play_beep(2500, 300)
                    speak(f"DANGER! {label} very close!", emergency=True)
                    break

                center_x = (float(x1) + float(x2)) / 2
                if label == "person" and (w / 3 < center_x < 2 * w / 3):
                    person_count_center += 1

            if current_emergency:
                break

        # Crowd alert
        if not current_emergency and person_count_center >= CROWD_PERSON_THRESHOLD:
            now = time.time()
            if now - last_alert_time > ALERT_COOLDOWN * 2:
                play_beep(900, 200)
                threading.Thread(
                    target=speak,
                    args=("Crowded area ahead. Move carefully.",),
                    daemon=True
                ).start()
                last_alert_time = now

        if current_emergency:
            annotated = results[0].plot() if results else frame
            cv2.putText(annotated, "EMERGENCY", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
            cv2.imshow("LUMOS v2.0", annotated)
            continue

        emergency_active = False

        # Priority 1: Walking hazards
        annotated_frame = results[0].plot() if results else frame

        for r in results:
            for box in r.boxes:
                label    = model.names[int(box.cls[0])]
                x1, y1, x2, y2 = box.xyxy[0]
                dist     = estimate_distance(float(y1), float(y2))
                center_x = (float(x1) + float(x2)) / 2
                in_path  = w / 3 < center_x < 2 * w / 3

                if label in TRIP_HAZARDS and dist < 1.8 and in_path:
                    now = time.time()
                    if (label != last_safety_alert or
                            now - last_alert_time > ALERT_COOLDOWN):
                        play_beep(800, 200)
                        threading.Thread(
                            target=speak,
                            args=(f"Caution. {label} ahead.",),
                            daemon=True
                        ).start()
                        last_safety_alert = label
                        last_alert_time   = now
                elif dist > 2.5:
                    if last_safety_alert == label:
                        last_safety_alert = ""

        # Pending AI speech
        if pending_speech:
            text_to_say    = pending_speech
            pending_speech = ""
            threading.Thread(target=speak, args=(text_to_say,), daemon=True).start()

        # Auto OCR in text mode every 5 s
        if current_mode == MODE_TEXT:
            if not hasattr(main, "_last_ocr") or time.time() - main._last_ocr > 5.0:
                main._last_ocr = time.time()
                frame_copy = frame.copy()
                threading.Thread(
                    target=lambda: _auto_read(frame_copy),
                    daemon=True
                ).start()

        
        cv2.putText(annotated_frame, f"Mode: {current_mode.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)

        if nav_steps and nav_step_index < len(nav_steps):
            cv2.putText(annotated_frame, nav_steps[nav_step_index][:60], (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("LUMOS v2.0", annotated_frame)

    cap.release()
    cv2.destroyAllWindows()


def _auto_read(frame):
    global pending_speech
    result = read_text_from_frame(frame)
    if "do not see any readable text" not in result:
        pending_speech = result


if __name__ == "__main__":
    main()