import cv2
import pytesseract

def extract_number_plate_text(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:  # likely rectangular number plate
            x, y, w, h = cv2.boundingRect(approx)
            roi = frame[y:y+h, x:x+w]

            # OCR
            text = pytesseract.image_to_string(roi, config='--psm 8')
            return text.strip()

    return "Not Found"
