from PIL import Image, ImageEnhance, ImageOps
import pytesseract

img_path = "data/butterflies/raw/Athyma_cama/Athyma_cama_Adult-Unknown_087.jpg"
img = Image.open(img_path)
w, h = img.size
top = img.crop((0, 0, w, int(h * 0.10)))
bot = img.crop((0, int(h * 0.90), w, h))

print("=== ORIGINAL ===")
print("TOP:", repr(pytesseract.image_to_string(top, lang="eng").strip()))
print("BOT:", repr(pytesseract.image_to_string(bot, lang="eng").strip()))

gray_t = top.convert("L")
gray_b = bot.convert("L")

print("\n=== GRAYSCALE ===")
print("TOP:", repr(pytesseract.image_to_string(gray_t, lang="eng").strip()))

def preprocess(im):
    gray = im.convert("L")
    # Invert so text is black, background is white (tesseract prefers black text on white)
    inv = ImageOps.invert(gray)
    # Enhance contrast 
    enhancer = ImageEnhance.Contrast(inv)
    inv = enhancer.enhance(2.0)
    # Thresholding
    # We want text (which was white -> now black/dark) to be 0, and background (dark -> now light) to be 255
    thresh = inv.point(lambda p: 255 if p > 160 else 0)
    return thresh

print("\n=== PREPROCESSED ===")
print("TOP:", repr(pytesseract.image_to_string(preprocess(top), lang="eng").strip()))
print("BOT:", repr(pytesseract.image_to_string(preprocess(bot), lang="eng").strip()))

