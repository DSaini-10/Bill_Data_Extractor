import requests
import json
import re
import numpy as np
from fastapi import FastAPI, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Union, Optional
from PIL import Image
import io
import os

# Import EasyOCR
try:
    import easyocr

    # Initialize EasyOCR reader once (it's heavy to initialize)
    reader = easyocr.Reader(['en'])  # English only for better performance
    EASYOCR_AVAILABLE = True
    print("EasyOCR initialized successfully")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available, please install: pip install easyocr")
except Exception as e:
    EASYOCR_AVAILABLE = False
    print(f"EasyOCR initialization failed: {e}")

# -------------------------------
# Groq API setup
# -------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"  # Working model

# -------------------------------
# FastAPI app
# -------------------------------

app = FastAPI(title="AI-Powered Amount Detection API with Groq", version="3.0")

# -------------------------------
# OCR HELPER (using EasyOCR)
# -------------------------------
def run_ocr(file_bytes: bytes) -> str:
    """
    Use EasyOCR for text extraction from images
    No system dependencies required - pure Python
    """
    if not EASYOCR_AVAILABLE:
        return "OCR not available: EasyOCR not installed"

    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(file_bytes))

        # Convert PIL Image to numpy array (required by EasyOCR)
        image_np = np.array(image)

        # Use EasyOCR to extract text
        results = reader.readtext(image_np, paragraph=True)

        # Combine all detected text
        text_parts = []
        for (bbox, text, confidence) in results:
            text_parts.append(text.strip())

        full_text = ' '.join(text_parts)

        print(f"EasyOCR extracted text: {full_text[:200]}...")  # Log first 200 chars
        return full_text

    except Exception as e:
        print(f"EasyOCR failed: {e}")
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
        "¥": "JPY", "jpy": "JPY",
        "₩": "KRW", "krw": "KRW"
    }

    currency_hint = None
    text_lower = text.lower()

    # Check for currency symbols and codes
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
# STEP 2: NORMALIZATION (IMPROVED FOR INTERNATIONAL FORMATS)
# -------------------------------
def step2_normalize(raw_tokens: List[str]) -> Dict[str, Union[List[float], float]]:
    normalized = []
    for token in raw_tokens:
        # Fix common OCR errors
        cleaned = token.replace("l", "1").replace("O", "0").replace("o", "0")

        # Handle numbers with periods as thousands separators (like 1.620.000)
        if '.' in token and ',' not in token and token.count('.') > 1:
            # This is likely a number with thousands separators (Vietnamese/European format)
            cleaned = token.replace('.', '')
        else:
            # Regular number processing
            cleaned = cleaned.replace(",", "")

        # Handle percentages
        if "%" in cleaned:
            cleaned = cleaned.replace("%", "")
            try:
                percentage_value = float(cleaned)
                if 0 <= percentage_value <= 100:  # Valid percentage range
                    normalized.append(percentage_value)
            except:
                continue
        else:
            try:
                amount = float(cleaned)
                # Broader range for international currencies
                if 0.01 <= amount <= 10000000:  # Expanded range
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
    """
    Use Groq API for classification with llama-3.1-8b-instant
    """
    try:
        prompt = f"""
        Analyze this receipt text and classify the following numbers into these categories: 
        total_bill, paid, due, discount, tax, tip, change.

        RECEIPT TEXT:
        "{text}"

        NUMBERS TO CLASSIFY: {numbers}

        INSTRUCTIONS:
        1. Analyze the receipt context around each number
        2. Classify each number into the most appropriate category
        3. Return ONLY a valid JSON array, no other text
        4. Format: [{{"type": "category", "value": number}}]

        EXAMPLE:
        For receipt "Total: 1200, Paid: 1000, Due: 200, Discount: 10%"
        Return: [{{"type": "total_bill", "value": 1200}}, {{"type": "paid", "value": 1000}}, {{"type": "due", "value": 200}}, {{"type": "discount", "value": 10}}]
        """

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that classifies receipt amounts. Always return valid JSON without any explanations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "model": GROQ_MODEL,
                "temperature": 0.1,
                "max_tokens": 1000,
                "stream": False
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"Groq raw response: {content}")

            # Parse the JSON response
            return parse_groq_response(content, numbers)
        else:
            print(f"Groq API error: {response.status_code} - {response.text}")
            return []

    except requests.exceptions.Timeout:
        print("Groq API timeout")
        return []
    except requests.exceptions.ConnectionError:
        print("Groq API connection error")
        return []
    except Exception as e:
        print(f"Groq API failed: {e}")
        return []


def parse_groq_response(response_text: str, original_numbers: List[float]) -> List[Dict]:
    """
    Parse Groq's response and validate against original numbers
    """
    try:
        # Clean the response - remove markdown code blocks
        cleaned_response = response_text.strip()
        if "```json" in cleaned_response:
            cleaned_response = cleaned_response.split("```json")[1].split("```")[0]
        elif "```" in cleaned_response:
            cleaned_response = cleaned_response.split("```")[1].split("```")[0]

        # Parse JSON
        classifications = json.loads(cleaned_response)

        # Validate the response
        if not isinstance(classifications, list):
            return []

        valid_classifications = []
        for item in classifications:
            if (isinstance(item, dict) and
                    'type' in item and
                    'value' in item and
                    item['value'] in original_numbers):
                valid_classifications.append(item)

        return valid_classifications

    except json.JSONDecodeError as e:
        print(f"Failed to parse Groq JSON: {e}")
        # Try to extract JSON using regex as fallback
        json_match = re.search(r'\[.*\]', response_text)
        if json_match:
            try:
                classifications = json.loads(json_match.group())
                return [item for item in classifications if
                        isinstance(item, dict) and 'type' in item and 'value' in item]
            except:
                pass
        return []
    except Exception as e:
        print(f"Error parsing Groq response: {e}")
        return []


# -------------------------------
# STEP 3: CLASSIFICATION BY CONTEXT (WITH GROQ INTEGRATION)
# -------------------------------
def step3_classify(text: str, normalized: List[float]) -> Dict[str, Union[List[Dict], float, str]]:
    classifications = []

    # Enhanced mapping with priority order
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

    # Create a copy for tracking unclassified numbers
    unclassified_numbers = normalized.copy()

    # First pass: exact pattern matching with word boundaries
    for keyword, label in mapping.items():
        # Look for patterns like "Total: 1200", "Paid: 1000", etc.
        pattern = fr"{keyword}\s*:?\s*(\d+(?:\.\d+)?)"
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            amount_str = match.group(1)
            try:
                amount = float(amount_str)
                if amount in unclassified_numbers:
                    classifications.append({"type": label, "value": amount})
                    unclassified_numbers.remove(amount)
                    print(f"Exact match: {keyword} -> {amount}")
            except:
                continue

    # Second pass: proximity-based matching for remaining numbers
    for num in unclassified_numbers[:]:  # Iterate over copy
        num_str = str(int(num) if num.is_integer() else str(num))

        # Find the position of this number in text
        num_match = re.search(re.escape(num_str), text)
        if not num_match:
            continue

        num_pos = num_match.start()

        # Look for keywords near this number
        best_match = None
        best_distance = float('inf')

        for keyword, label in mapping.items():
            keyword_matches = list(re.finditer(re.escape(keyword), text, re.IGNORECASE))
            for kw_match in keyword_matches:
                kw_pos = kw_match.start()
                distance = abs(kw_pos - num_pos)

                # If keyword is within 50 characters of the number
                if distance <= 50 and distance < best_distance:
                    best_match = label
                    best_distance = distance

        if best_match and num in unclassified_numbers:
            classifications.append({"type": best_match, "value": num})
            unclassified_numbers.remove(num)
            print(f"Proximity match: {best_match} -> {num}")

    # Third pass: Use Groq API for remaining numbers
    groq_used = False
    if unclassified_numbers and GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        print(f"Using Groq API for {len(unclassified_numbers)} unclassified numbers")

        groq_classifications = classify_with_groq(text, unclassified_numbers)
        if groq_classifications:
            classifications.extend(groq_classifications)
            groq_used = True
            print(f"✅ Groq successfully classified {len(groq_classifications)} numbers")
        else:
            print("❌ Groq classification failed")

    # Fourth pass: Smart regex-based fallback when Groq fails
    if unclassified_numbers and not groq_used:
        print(f"Using regex fallback for {len(unclassified_numbers)} numbers")
        regex_classifications = regex_fallback_classification(text, unclassified_numbers)
        for item in regex_classifications:
            if item['value'] in unclassified_numbers:
                classifications.append(item)
                unclassified_numbers.remove(item['value'])
        print(f"Regex fallback classified {len(regex_classifications)} numbers")

    confidence = 0.80 if classifications else 0.0
    result = {
        "amounts": classifications,
        "confidence": confidence
    }

    # Add classification method info
    if groq_used:
        result["classification_method"] = "groq_assisted"
    elif unclassified_numbers:
        result["classification_method"] = "regex_based"
        if unclassified_numbers:
            result["notes"] = f"Used regex fallback for {len(unclassified_numbers)} numbers"
    else:
        result["classification_method"] = "rule_based"

    return result


# -------------------------------
# SMART REGEX FALLBACK CLASSIFICATION
# -------------------------------
def regex_fallback_classification(text: str, numbers: List[float]) -> List[Dict]:
    """Robust regex-based classification when Groq fails"""
    classifications = []
    text_lower = text.lower()

    for num in numbers:
        num_str = str(int(num) if num.is_integer() else str(num))

        # Find the line containing this number
        lines = text.split('\n')
        for line in lines:
            if num_str in line:
                line_lower = line.lower()

                # Check for total patterns
                if any(word in line_lower for word in ['total', 'tổng cộng', 'amount', 'subtotal', 'grand']):
                    classifications.append({"type": "total_bill", "value": num})
                    break
                # Check for payment patterns
                elif any(word in line_lower for word in ['paid', 'payment', 'cash', 'thanh toán', 'received']):
                    classifications.append({"type": "paid", "value": num})
                    break
                # Check for due patterns
                elif any(word in line_lower for word in ['due', 'balance', 'outstanding']):
                    classifications.append({"type": "due", "value": num})
                    break
                # Check for tax patterns
                elif any(word in line_lower for word in ['tax', 'gst', 'vat']):
                    classifications.append({"type": "tax", "value": num})
                    break
                # Check for discount patterns
                elif any(word in line_lower for word in ['discount', 'off', '%']):
                    classifications.append({"type": "discount", "value": num})
                    break
                # Check for tip patterns
                elif any(word in line_lower for word in ['tip', 'service charge']):
                    classifications.append({"type": "tip", "value": num})
                    break
                # Check for change patterns
                elif any(word in line_lower for word in ['change', 'refund']):
                    classifications.append({"type": "change", "value": num})
                    break
        else:
            # If no specific pattern found, use heuristics based on value
            if num == max(numbers):
                classifications.append({"type": "total_bill", "value": num})
            elif num < 100 and num > 0:
                classifications.append({"type": "discount", "value": num})
            else:
                classifications.append({"type": "paid", "value": num})

    return classifications


# -------------------------------
# STEP 4: FINAL OUTPUT (WITH CLASSIFICATION INFO)
# -------------------------------
def step4_output(text: str, currency: str, classified: Dict) -> Dict:
    meaningful_types = {"total_bill", "paid", "due", "discount", "tax", "tip", "change"}

    # Deduplicate amounts by value and type
    seen = set()
    final_amounts = []

    for entry in classified["amounts"]:
        if not isinstance(entry, dict):
            continue

        entry_type = entry.get("type")
        entry_value = entry.get("value")

        if entry_type not in meaningful_types:
            continue

        if entry_value is None:
            continue

        # Create a unique identifier for this amount-type combination
        entry_id = f"{entry_type}_{entry_value}"
        if entry_id in seen:
            continue
        seen.add(entry_id)

        # Find the exact source context
        num_str = str(
            int(entry_value) if isinstance(entry_value, float) and entry_value.is_integer() else str(entry_value))

        # Look for the specific line containing this amount with its label
        best_source = None
        lines = text.split('\n')
        for line in lines:
            if num_str in line:
                # Clean up the line for better source display
                clean_line = re.sub(r'\s+', ' ', line.strip())
                best_source = f"text: '{clean_line}'"
                break

        if not best_source:
            # Fallback: find context around the number
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

    # Add classification method info to final output
    if "classification_method" in classified:
        result["classification_method"] = classified["classification_method"]
    if "notes" in classified:
        result["notes"] = classified["notes"]

    return result


# -------------------------------
# PIPELINE WRAPPER
# -------------------------------
def process_document(text: str):
    if not text or not text.strip():
        return {"status": "no_amounts_found", "reason": "document too noisy"}

    print(f"Processing text: {text}")

    # Step 1: Extract
    step1 = step1_extract(text)
    print(f"Step 1 - Extraction: {step1}")

    if "status" in step1 and step1["status"] == "no_amounts_found":
        return step1

    # Step 2: Normalize
    step2 = step2_normalize(step1["raw_tokens"])
    print(f"Step 2 - Normalization: {step2}")

    # Step 3: Classify
    step3 = step3_classify(text, step2["normalized_amounts"])
    print(f"Step 3 - Classification: {step3}")

    # Step 4: Final Output
    step4 = step4_output(text, step1["currency_hint"], step3)
    print(f"Step 4 - Output: {step4}")

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
    Process image files only
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


# Health check endpoint
@app.get("/health")
async def health_check():
    groq_status = "available" if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here" else "unavailable"
    return {
        "status": "healthy",
        "groq_available": groq_status,
        "model": GROQ_MODEL,
        "ocr_engine": "easyocr",
        "ocr_available": EASYOCR_AVAILABLE
    }


# Test endpoint to verify OCR functionality
@app.post("/test-ocr")
async def test_ocr(file: UploadFile):
    """
    Test endpoint to verify EasyOCR is working
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        file_bytes = await file.read()
        text_content = run_ocr(file_bytes)

        return {
            "filename": file.filename,
            "extracted_text": text_content,
            "ocr_engine": "easyocr",
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR test failed: {str(e)}")