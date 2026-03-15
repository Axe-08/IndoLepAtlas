import easyocr
import os
from PIL import Image

reader = easyocr.Reader(["en"], gpu=True)
img_path = "data/butterflies/raw/Athyma_cama/Athyma_cama_Adult-Unknown_087.jpg"
if os.path.exists(img_path):
    img = Image.open(img_path)
    w, h = img.size
    # Crop bottom overlay
    bot = img.crop((0, int(h * 0.90), w, h))
    bot.save("bot_test.jpg")
    
    results = reader.readtext("bot_test.jpg")
    print("=== EasyOCR Results ===")
    for (bbox, text, prob) in results:
        print(f"[{prob:.2f}] {text}")
else:
    print("Image not found")
