import pandas as pd
import os
import re

COMMON_OCR_NOISE = {"ee", "eee", "bo", "me", "cm", "gemma", "mma", "cma", "ey", "se", "mer", "conga", "pees", "aman", "te", "iam", "nae", "mgs", "gal", "en"}

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", 
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", 
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", 
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", 
    "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli", 
    "Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
]

def clean_refined(text):
    if not isinstance(text, str): return ""
    # Strip leading/trailing non-alphanumeric junk
    text = re.sub(r"^[^a-zA-Z0-9]+", "", text)
    text = re.sub(r"[^a-zA-Z0-9]+$", "", text)
    words = text.split()
    cleaned_words = []
    for w in words:
        wl = w.lower().strip(".,()/-")
        if wl in COMMON_OCR_NOISE: continue
        if re.match(r"^([a-z]\.)+[a-z]?$", wl): continue
        cleaned_words.append(w)
    text = " ".join(cleaned_words)
    text = text.replace("@", "").replace("{", "(").replace("}", ")")
    text = re.sub(r"\s+", " ", text).strip(" ,.-;")
    return text

def parse_polluted_location(row):
    loc = str(row.get("location", ""))
    sex = str(row.get("sex", ""))
    state = str(row.get("state", ""))
    date = str(row.get("date", ""))
    credit = str(row.get("credit", ""))

    # 1. Extract Date (YYYY/MM/DD)
    date_match = re.search(r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})", loc)
    if date_match:
        found_date = date_match.group(1).replace("-", "/")
        if date.lower() in ["", "nan", "none"]:
            date = found_date
        loc = loc.replace(date_match.group(0), "").strip(" ,.-;")

    # 2. Extract Sex (Male/Female at start or in middle)
    sex_match = re.search(r"\b(Male|Female)\b", loc, re.IGNORECASE)
    if sex_match:
        found_sex = sex_match.group(1).capitalize()
        if sex.lower() in ["", "nan", "none", "unknown"]:
            sex = found_sex
        # Remove sex from loc - handle potential punctuation after it
        loc = re.sub(r"\b" + re.escape(sex_match.group(1)) + r"[\.\s,;]*", "", loc, flags=re.IGNORECASE).strip(" ,.-;")

    # 3. Extract State
    for s in INDIAN_STATES:
        if s.lower() in loc.lower():
            if state.lower() in ["", "nan", "none"]:
                state = s
            # Don't necessarily remove state as it's part of the address, 
            # but we've successfully parsed it to the field.
            break

    # 4. Extract Credit (© or sequence after last landmark)
    if "©" in loc:
        parts = loc.split("©")
        loc = parts[0].strip(" ,;")
        found_credit = parts[1].strip()
        if credit.lower() in ["", "nan", "none"]:
            credit = found_credit
    
    # 5. Handle "India" pollution and extra spaces
    loc = re.sub(r"\bIndia\b", "", loc).strip(" ,.-;")
    loc = re.sub(r"\s+", " ", loc)

    return pd.Series([loc, sex, state, date, credit])

def finalize():
    for dataset in ["butterflies", "plants"]:
        csv_path = f"data/{dataset}/metadata.csv"
        if not os.path.exists(csv_path): continue
        
        df = pd.read_csv(csv_path)
        if "location_refined" in df.columns:
            mask = df["location_refined"].notna() & (df["location_refined"] != "")
            df.loc[mask, "location"] = df.loc[mask, "location_refined"].apply(clean_refined)
            df = df.drop(columns=["location_refined"])

        # Apply advanced parsing to all rows
        print(f"Parsing polluted fields for {dataset}...")
        df[["location", "sex", "state", "date", "credit"]] = df.apply(parse_polluted_location, axis=1)
        
        df.to_csv(csv_path, index=False)
        print(f"Finalized {dataset} metadata with advanced parsing.")

if __name__ == "__main__":
    finalize()
