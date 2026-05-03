from gtts import gTTS
import os
import pygame

def speak_test(text):
    """ Bypasses the Windows Speech Driver entirely """
    tts = gTTS(text=text, lang='en')
    tts.save("test.mp3")
    
    pygame.mixer.init()
    pygame.mixer.music.load("test.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    pygame.mixer.quit()
    os.remove("test.mp3")

speak_test("Luma is now using a different audio driver. Can you hear me?")