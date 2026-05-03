import easyocr

# Initialize the reader once globally
# It will download models (~100MB) on the first run
print(">>> [SYSTEM]: Loading OCR Models... Please wait.")
reader = easyocr.Reader(['en'])

def get_text_from_image(image_path):
    """
    Extracts text from a given image using EasyOCR.
    Returns a clean string of all detected text.
    """
    try:
        results = reader.readtext(image_path)
        # Combine all snippets into one string
        extracted_text = " ".join([res[1] for res in results])
        return extracted_text if extracted_text.strip() else "No specific text detected."
    except Exception as e:
        print(f">>> [OCR ERROR]: {e}")
        return ""