import cv2

def run_tesseract_ocr(image, label):
    """
    Dummy Tesseract call. In practice, call pytesseract.image_to_string or similar.
    """
    # Here we just return a fake string based on label
    if label == "Test Name":
        return "TOTAL TRIIODOTHYRONINE (T3)"
    elif label == "Value":
        return "79"
    elif label == "Unit":
        return "ng/dl"
    elif label == "Ref Value":
        return "60-200"
    return "N/A"
