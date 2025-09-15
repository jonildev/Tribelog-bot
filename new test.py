import pytesseract
from PIL import ImageGrab, Image, ImageEnhance
import time
import requests

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update if needed

# Tribe log area on screen — adjust for your setup
LOG_REGION = (100, 100, 700, 600)  # CHANGE this

# Your Discord webhook
WEBHOOK_URL = "https://discord.com/api/webhooks/your_webhook_url"

# Store previous log lines
previous_lines = []

# 🔎 Keywords that trigger alerts
RED_KEYWORDS = ["was killed", "destroyed", "demolished", "death", "slain"]
STARVE_KEYWORDS = ["starved", "has starved", "died of starvation"]
RAID_KEYWORDS = ["destroyed by", "structure was destroyed", "demolished by", "enemy", "enemy foundation", "raided"]

def capture_and_process_image():
    image = ImageGrab.grab(bbox=LOG_REGION)
    gray = image.convert("L")
    enhancer = ImageEnhance.Contrast(gray)
    contrast = enhancer.enhance(3.0)
    bw = contrast.point(lambda x: 0 if x < 160 else 255, '1')
    return bw

def extract_lines(image):
    config = r'-c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:/[]().,! " --psm 6'
    text = pytesseract.image_to_string(image, config=config)
    lines = text.strip().split('\n')
    return [line.strip() for line in lines if line.strip()]

def send_to_discord(content, alert_type="normal"):
    if not content:
        return

    emoji = {
        "normal": "📜",
        "warning": "🚨",
        "starve": "⚠️",
        "raid": "🔥"
    }[alert_type]

    message = f"{emoji} **{'ALERT' if alert_type != 'normal' else 'New Tribe Log Entries'}:**\n```{content}```"
    data = {"content": message}

    response = requests.post(WEBHOOK_URL, json=data)
    if response.status_code != 204:
        print("❌ Failed to send to Discord:", response.text)

def detect_alert_type(line):
    lowered = line.lower()
    if any(word in lowered for word in STARVE_KEYWORDS):
        return "starve"
    if any(word in lowered for word in RAID_KEYWORDS):
        return "raid"
    if any(word in lowered for word in RED_KEYWORDS):
        return "warning"
    return "normal"

def main():
    global previous_lines
    print("✅ Monitoring tribe log with alert detection...")

    while True:
        image = capture_and_process_image()
        current_lines = extract_lines(image)

        new_lines = []
        for line in current_lines:
            if line not in previous_lines:
                new_lines.append(line)

        for line in new_lines:
            alert_type = detect_alert_type(line)
            send_to_discord(line, alert_type=alert_type)

        if new_lines:
            previous_lines = current_lines

        time.sleep(30)

if __name__ == "__main__":
    main()
