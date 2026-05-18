"""
When main.py is available, replace this with the real LumosServer.

"""

import threading
import queue
import time
import os

# ── Try gTTS for local audio fallback ────────────────────────────────────────
try:
    import pygame
    import gtts
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except:
    AUDIO_AVAILABLE = False

from nova_network_models import BaseEvent, SpeakEvent


class LumosServer:
    """
    Drop-in stub for the real LumosServer in main.py.
    Replace this class import with the real one when main.py is available.
    """

    def __init__(self):
        self._event_queue  = queue.Queue()
        self._gps_lock     = threading.Lock()
        self._gps_coords   = None          # (lon, lat) set by phone
        self._tts_lock     = threading.Lock()
        self._connected    = False         # True when phone is connected

        # Start the event processor thread
        threading.Thread(target=self._process_events, daemon=True).start()
        print(">>> [SERVER]: LumosServer stub started. Waiting for phone connection...")

    # ── Public API ────────────────────────────────────────────────────────────

    def send_event(self, event: BaseEvent):
        """
        Thread-safe. Call this from your vision loop.
        Puts the event on the queue — never blocks.
        """
        self._event_queue.put(event)

    def get_gps(self) -> tuple | None:
        """Returns (lon, lat) from phone GPS, or None if unavailable."""
        with self._gps_lock:
            return self._gps_coords

    def set_gps(self, lon: float, lat: float):
        """Called when phone sends a GPS update via WebSocket."""
        with self._gps_lock:
            self._gps_coords = (lon, lat)

    @property
    def connected(self) -> bool:
        return self._connected

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_events(self):
        """Drains the event queue and handles each event."""
        while True:
            try:
                event = self._event_queue.get(timeout=1)
                self._handle(event)
                self._event_queue.task_done()
            except queue.Empty:
                continue

    def _handle(self, event: BaseEvent):
        d = event.to_dict()
        print(f">>> [EVENT] {d['type']} — {d['payload']}")

        # If it's a SPEAK event and no phone is connected, speak locally
        if d["type"] == "SPEAK" and not self._connected:
            self._speak_local(d["payload"]["text"], d["payload"].get("emergency", False))

        # OCR, SCENE, NAV — speak locally if no phone
        elif d["type"] in ("OCR_RESULT", "SCENE_DESCRIPTION", "NAVIGATION") and not self._connected:
            text = (
                d["payload"].get("text") or
                d["payload"].get("answer") or
                d["payload"].get("instruction") or ""
            )
            if text:
                self._speak_local(text)

    def _speak_local(self, text: str, emergency: bool = False):
        """Fallback TTS when phone is not connected."""
        if not AUDIO_AVAILABLE or not text.strip():
            return
        with self._tts_lock:
            try:
                from gtts import gTTS
                if emergency:
                    pygame.mixer.music.stop()
                tts = gTTS(text=text, lang="en")
                fname = f"luma_stub_{threading.get_ident()}.mp3"
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
                print(f">>> [AUDIO ERROR]: {e}")


# ── Singleton — import this in vision.py ─────────────────────────────────────
server = LumosServer()