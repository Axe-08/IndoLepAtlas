import glob
files = glob.glob("annotations/butterflies/*/*.txt")
print(f"Generated {len(files)} COCO annotations so far")
