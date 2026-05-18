import time
import uuid

class BaseEvent:
    """Base class for all Lumos events."""

    def __init__(self, event_type: str, payload: dict):
        self.event_type = event_type
        self.payload    = payload
        self.id         = str(uuid.uuid4())[:8]
        self.timestamp  = time.time()

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "type":       self.event_type,
            "payload":    self.payload,
            "timestamp":  self.timestamp,
        }

    def __repr__(self):
        return f"<{self.event_type} id={self.id}>"


class SocialAlertEvent(BaseEvent):
    """
    Used for YOLO detections, crowd alerts, and safety warnings.
    Matches the naming convention required by the network module.
    """
    @classmethod
    def create(cls, event_type: str, payload: dict) -> "SocialAlertEvent":
        # بيعمل Initialization للـ Class وبيمرر الـ type والـ payload للـ Base class
        return cls(event_type, payload)


class SpeakEvent(BaseEvent):
    """Tells the phone/server to speak a piece of text aloud."""
    @classmethod
    def create(cls, text: str, emergency: bool = False) -> "SpeakEvent":
        return cls("SPEAK", {"text": text, "emergency": emergency})


class OCREvent(BaseEvent):
    """Carries text recognized from the camera frame."""
    @classmethod
    def create(cls, text: str) -> "OCREvent":
        return cls("OCR_RESULT", {"text": text})


class SceneEvent(BaseEvent):
    """Carries Gemini's answer to a scene description question."""
    @classmethod
    def create(cls, question: str, answer: str) -> "SceneEvent":
        return cls("SCENE_DESCRIPTION", {"question": question, "answer": answer})


class NavigationEvent(BaseEvent):
    """Carries a single navigation instruction."""
    @classmethod
    def create(cls, instruction: str, step_index: int, total_steps: int) -> "NavigationEvent":
        return cls("NAVIGATION", {
            "instruction": instruction,
            "step":        step_index,
            "total":       total_steps,
        })


class ModeChangeEvent(BaseEvent):
    """Notifies the phone that the user switched modes."""
    @classmethod
    def create(cls, new_mode: str) -> "ModeChangeEvent":
        return cls("MODE_CHANGE", {"mode": new_mode})