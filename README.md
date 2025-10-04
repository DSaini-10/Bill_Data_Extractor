# Bill Data Extractor — Backend (FastAPI + Groq + Tesseract)

AI-powered document parsing API that accepts **text** and **image** inputs, performs OCR (Tesseract), normalizes numbers, and classifies amounts (total, paid, due, discount, tax, tip, change) via rule-based logic and the Groq LLM. 
Built with FastAPI and deployed on Render.

**Intended users:** Institutional workflows (finance, reimbursements, medical claims).

## Repo contents
- `app.py` — FastAPI app (endpoints: `/text-form`, `/file`, `/health`)
- `requirements.txt` — List of Python dependencies
- `Procfile` & `Dockerfile` — For Render deployment
- `apt.txt` — For Tesseract installation on Render Server
- `README.md` — this file
- `samples/` — sample receipts and sample requests 

## Deployed Service Url: https://billdata-extractor.onrender.com

## Quick start Guide for local Server
1. Clone repo:
```bash
git clone https://github.com/DSaini-10/Bill_Data_Extractor.git
cd Bill_Data_Extractor
```
2. Create venv and install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
3.Create .env with secrets containing:

GROQ_API_KEY=your_groq_key_here

4.Ensure Tesseract is installed locally:

For macOS - 
```bash
brew install tesseract
```

For ubuntu
```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```
5.Start app:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## For Testing (On Deployed Version):
1. Head to Url : ## Deployed Service Url: https://billdata-extractor.onrender.com/docs
2. This opens the Swagger UI for testing 
3. Test the main endpoints : /text-form and /file


### Reporting any bugs or improvements is always welcomed
