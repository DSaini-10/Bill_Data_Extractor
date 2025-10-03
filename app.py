import requests
import json
import re
import pytesseract
from fastapi import FastAPI, UploadFile, Form, HTTPException
from typing import Dict, List, Union
from PIL import Image
import io
import os

# Tesseract is already installed in the system via Docker
# No need to set tesseract_cmd path in Docker

# -------------------------------
# Groq API setup
# -------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Receipt Processor API with OCR", version="6.0")


# -------------------------------
# OCR HELPER (with Tesseract)
# -------------------------------
def run_ocr(file_bytes: bytes) -> str:
    """
    Use Tesseract OCR for text extraction from images
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(file_bytes))

        # Use pytesseract to extract text
        text = pytesseract.image_to_string(image)

        print(f"Tesseract extracted text: {text[:200]}...")
        return text

    except Exception as e:
        print(f"Tesseract OCR failed: {e}")
        return f"OCR failed: {str(e)}"


# -------------------------------
# STEP 1: OCR / Text Extraction
# -------------------------------
def step1_extract(text: str) -> Dict[str, Union[List[str], str, float]]:
    if not text:
        return {"status": "no_amounts_found", "reason": "document too noisy"}

    # Extract numbers and percentages
    raw_tokens = re.findall(r'\d+[.,]?\d*%?', text)

    currency_hints = {
        "₹": "INR", "rs": "INR", "inr": "INR", "rs.": "INR",
        "rupee": "INR", "rupees": "INR", "rp": "INR", "rps": "INR",
        "rs ": "INR", "rs/-": "INR", "/-": "INR", "inr ": "INR",
        "lakh": "INR", "lac": "INR", "crore": "INR",
        "rm": "MYR", "myr": "MYR", "ringgit": "MYR",
        "$": "USD", "usd": "USD",
        "€": "EUR", "eur": "EUR",
        "£": "GBP", "gbp": "GBP",
        "₫": "VND", "vnd": "VND", "dong": "VND", "đồng": "VND",
    }

    currency_hint = None
    text_lower = text.lower()

    for symbol, code in currency_hints.items():
        if symbol in text or code in text_lower:
            currency_hint = code
            break

    confidence = 0.74 if raw_tokens else 0.0
    if not raw_tokens:
        return {"status": "no_amounts_found", "reason": "document too noisy"}

    return {
        "raw_tokens": raw_tokens,
        "currency_hint": currency_hint,
        "confidence": confidence
    }


# -------------------------------
# STEP 2: NORMALIZATION
# -------------------------------
def step2_normalize(raw_tokens: List[str]) -> Dict[str, Union[List[float], float]]:
    normalized = []
    for token in raw_tokens:
        cleaned = token.replace("l", "1").replace("O", "0").replace("o", "0")

        if '.' in token and ',' not in token and token.count('.') > 1:
            cleaned = token.replace('.', '')
        else:
            cleaned = cleaned.replace(",", "")

        if "%" in cleaned:
            cleaned = cleaned.replace("%", "")
            try:
                percentage_value = float(cleaned)
                if 0 <= percentage_value <= 100:
                    normalized.append(percentage_value)
            except:
                continue
        else:
            try:
                amount = float(cleaned)
                if 0.01 <= amount <= 10000000:
                    normalized.append(amount)
            except:
                continue

    confidence = 0.82 if normalized else 0.0
    return {
        "normalized_amounts": normalized,
        "normalization_confidence": confidence
    }


# -------------------------------
# GROQ API CLASSIFICATION
# -------------------------------
def classify_with_groq(text: str, numbers: List[float]) -> List[Dict]:
    """Groq API classification"""
    if not GROQ_API_KEY:
        return []

    try:
        prompt = f"""
        Analyze this receipt text and classify these numbers: {numbers}
        Receipt: "{text}"
        Categories: total_bill, paid, due, discount, tax, tip, change
        Return ONLY JSON: [{{"type": "category", "value": number}}]
        """

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "messages": [{"role": "user", "content": prompt}],
                "model": GROQ_MODEL,
                "temperature": 0.1,
                "max_tokens": 500
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Parse JSON response
            cleaned_response = content.strip()
            if "```json" in cleaned_response:
                cleaned_response = cleaned_response.split("```json")[1].split("```")[0]
            elif "```" in cleaned_response:
                cleaned_response = cleaned_response.split("```")[1].split("```")[0]

            classifications = json.loads(cleaned_response)
            return [item for item in classifications if isinstance(item, dict) and 'type' in item and 'value' in item]

    except Exception as e:
        print(f"Groq API failed: {e}")

    return []


# -------------------------------
# STEP 3: CLASSIFICATION
# -------------------------------
def step3_classify(text: str, normalized: List[float]) -> Dict[str, Union[List[Dict], float, str]]:
    classifications = []

    mapping = {
        "total": "total_bill",
        "tổng cộng": "total_bill",
        "amount": "total_bill",
        "subtotal": "total_bill",
        "grand total": "total_bill",
        "final amount": "total_bill",
        "paid": "paid",
        "thanh toán": "paid",
        "payment": "paid",
        "received": "paid",
        "cash": "paid",
        "due": "due",
        "balance": "due",
        "outstanding": "due",
        "discount": "discount",
        "off": "discount",
        "tax": "tax",
        "gst": "tax",
        "vat": "tax",
        "tip": "tip",
        "service charge": "tip",
        "change": "change"
    }

    unclassified_numbers = normalized.copy()

    # Exact pattern matching
    for keyword, label in mapping.items():
        pattern = fr"{keyword}\s*:?\s*(\d+(?:\.\d+)?)"
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            amount_str = match.group(1)
            try:
                amount = float(amount_str)
                if amount in unclassified_numbers:
                    classifications.append({"type": label, "value": amount})
                    unclassified_numbers.remove(amount)
            except:
                continue

    # Groq API fallback
    groq_used = False
    if unclassified_numbers and GROQ_API_KEY:
        groq_classifications = classify_with_groq(text, unclassified_numbers)
        if groq_classifications:
            classifications.extend(groq_classifications)
            groq_used = True

    # Simple fallback for remaining numbers
    if unclassified_numbers:
        for num in unclassified_numbers:
            if num == max(normalized):
                classifications.append({"type": "total_bill", "value": num})
            elif num < 100 and num > 0:
                classifications.append({"type": "discount", "value": num})
            else:
                classifications.append({"type": "paid", "value": num})

    confidence = 0.80 if classifications else 0.0
    result = {
        "amounts": classifications,
        "confidence": confidence
    }

    if groq_used:
        result["classification_method"] = "groq_assisted"
    else:
        result["classification_method"] = "rule_based"

    return result


# -------------------------------
# STEP 4: FINAL OUTPUT
# -------------------------------
def step4_output(text: str, currency: str, classified: Dict) -> Dict:
    meaningful_types = {"total_bill", "paid", "due", "discount", "tax", "tip", "change"}

    seen = set()
    final_amounts = []

    for entry in classified["amounts"]:
        if not isinstance(entry, dict):
            continue

        entry_type = entry.get("type")
        entry_value = entry.get("value")

        if entry_type not in meaningful_types or entry_value is None:
            continue

        entry_id = f"{entry_type}_{entry_value}"
        if entry_id in seen:
            continue
        seen.add(entry_id)

        num_str = str(
            int(entry_value) if isinstance(entry_value, float) and entry_value.is_integer() else str(entry_value))

        best_source = None
        lines = text.split('\n')
        for line in lines:
            if num_str in line:
                clean_line = re.sub(r'\s+', ' ', line.strip())
                best_source = f"text: '{clean_line}'"
                break

        if not best_source:
            match = re.search(fr'.{{0,20}}{re.escape(num_str)}.{{0,20}}', text)
            if match:
                clean_context = re.sub(r'\s+', ' ', match.group(0).strip())
                best_source = f"text: '{clean_context}'"
            else:
                best_source = f"text contains {entry_value}"

        final_amounts.append({
            "type": entry_type,
            "value": entry_value,
            "source": best_source
        })

    result = {
        "currency": currency or "UNKNOWN",
        "amounts": final_amounts,
        "status": "ok" if final_amounts else "no_amounts_found"
    }

    if "classification_method" in classified:
        result["classification_method"] = classified["classification_method"]

    return result


# -------------------------------
# PIPELINE WRAPPER
# -------------------------------
def process_document(text: str):
    if not text or not text.strip():
        return {"status": "no_amounts_found", "reason": "document too noisy"}

    step1 = step1_extract(text)
    if "status" in step1 and step1["status"] == "no_amounts_found":
        return step1

    step2 = step2_normalize(step1["raw_tokens"])
    step3 = step3_classify(text, step2["normalized_amounts"])
    step4 = step4_output(text, step1["currency_hint"], step3)

    return step4


# -------------------------------
# ENDPOINTS
# -------------------------------

@app.post("/text-form")
async def process_text_form(text: str = Form(...)):
    """
    Process text input only - Form data format
    """
    try:
        result = process_document(text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/file")
async def process_file(file: UploadFile):
    """
    Process image files with OCR
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        file_bytes = await file.read()
        text_content = run_ocr(file_bytes)

        result = process_document(text_content)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/health")
async def health_check():
    groq_status = "available" if GROQ_API_KEY else "unavailable"
    return {
        "status": "healthy",
        "groq_available": groq_status,
        "model": GROQ_MODEL,
        "ocr_available": True,
        "ocr_engine": "tesseract"
    }


@app.get("/")
async def root():
    return {
        "message": "Receipt Processor API with OCR",
        "endpoints": {
            "/text-form": "POST - Process text input",
            "/file": "POST - Process image files with OCR",
            "/health": "GET - Health check",
            "/docs": "API documentation"
        }
    }