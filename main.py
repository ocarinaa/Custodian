#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Document Sorter
This script processes documents from a specified folder using Tesseract OCR and Google Gemini AI,
renames them based on analysis, and moves them to target folders.
"""

import os
import shutil
import json
import logging
from datetime import datetime
from typing import List, Dict

# Required libraries (must be in requirements.txt)
import google.generativeai as genai
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import docx
import openpyxl
from dotenv import load_dotenv

# ==============================================================================
# --- SETTINGS SECTION ---
# PLEASE CREATE A `.env` FILE BY COPYING `.env.example`
# AND FILL IT WITH YOUR OWN INFORMATION.
# ==============================================================================
load_dotenv()

# --- SETTINGS FROM .env FILE ---

# Your Google Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Full path to the Tesseract OCR executable on your system
# Example (Windows): TESSERACT_PATH = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Full path to the Poppler 'bin' directory (for PDF processing on Windows)
# Example: POPPLER_PATH = "C:\\path\\to\\poppler-22.04.0\\Library\\bin"
POPPLER_PATH = os.getenv("POPPLER_PATH")

# The main folder containing the documents to be processed
SOURCE_FOLDER = os.getenv("SOURCE_FOLDER")

# Target folders for processed files
RENAMED_FOLDER = os.getenv("RENAMED_FOLDER")
NEEDS_REVIEW_FOLDER = os.getenv("NEEDS_REVIEW_FOLDER")

# --- DIRECTLY EDITABLE SETTINGS ---

# The list of document types for the AI to choose from.
# You can customize this list to fit your needs.
DOCUMENT_CATEGORIES = [
    "Invoice", "Bank Statement", "Contract", "Payslip", "ID Card",
    "Passport", "Vehicle Registration", "Property Deed", "Medical Report",
    "Resume (CV)", "Certificate", "Letter", "Other"
]

# Define the file naming format. You can use the following placeholders:
# {category}, {entity}, {date}, {original_name}
# Example: "{date}_{entity}_{category}" -> "2025-08-21_ACME-Inc_Invoice.pdf"
FILENAME_FORMAT = "{date}_{entity}_{category}"

# Confidence score threshold (from 0 to 100).
# Results below this threshold will be moved to the "needs review" folder.
CONFIDENCE_THRESHOLD = 75

# ==============================================================================
# --- SCRIPT CODE --- (No changes needed below this line)
# ==============================================================================

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# --- CORE FUNCTIONS ---

def extract_text_from_file(file_path: str) -> str:
    """Extracts text from various file formats using the most efficient method."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == '.pdf':
            # Use poppler_path if provided in .env
            poppler_kwargs = {'poppler_path': POPPLER_PATH} if POPPLER_PATH else {}
            images = convert_from_path(file_path, **poppler_kwargs)
            for img in images:
                text += pytesseract.image_to_string(img, lang='eng+deu+tur') + "\n"
        elif ext in ['.png', '.jpg', '.jpeg']:
            text = pytesseract.image_to_string(Image.open(file_path), lang='eng+deu+tur')
        elif ext == '.docx':
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext == '.xlsx':
            workbook = openpyxl.load_workbook(file_path)
            full_text = []
            for sheet in workbook.worksheets:
                for row in sheet.iter_rows():
                    for cell in row:
                        if cell.value: full_text.append(str(cell.value))
            text = "\n".join(full_text)
    except pytesseract.TesseractNotFoundError:
        logging.critical("Tesseract is not installed or not in your PATH. Please check the TESSERACT_PATH in your .env file.")
        raise
    except FileNotFoundError:
        logging.error(f"File not found: '{file_path}'")
    except Exception as e:
        # Catching pdf2image/poppler errors specifically
        if "Poppler" in str(e):
            logging.error(f"Poppler error processing '{os.path.basename(file_path)}'. Is POPPLER_PATH set correctly in .env? Error: {e}")
        else:
            logging.error(f"Error reading '{os.path.basename(file_path)}': {e}")
    return text.strip()

def analyze_text_with_gemini(text: str) -> Dict:
    """Analyzes text with Gemini and returns a structured JSON."""
    # Default error response
    error_response = {
        "entity": None,
        "category": "Other",
        "date": datetime.now().strftime('%Y-%m-%d'),
        "confidence_score": 0,
        "reason_for_review": "AI analysis failed or returned an invalid format."
    }

    if not GOOGLE_API_KEY:
        logging.error("Google API Key is not configured.")
        return error_response

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(
            'gemini-1.5-flash-latest',
            generation_config={"response_mime_type": "application/json"}
        )

        prompt = f"""
        Analyze the document text below and return ONLY a JSON object.

        DOCUMENT TEXT:
        ---
        {text[:4000]}
        ---

        TASKS:
        1.  `entity`: Identify the primary person, company, or organization. If none, use the original filename.
        2.  `category`: Classify the document. It MUST be one of these: {json.dumps(DOCUMENT_CATEGORIES)}.
        3.  `date`: Find the most relevant date and format it as `YYYY-MM-DD`. If no date is found, use today's date: {datetime.now().strftime('%Y-%m-%d')}.
        4.  `confidence_score`: Rate your confidence (0-100) in the `entity` and `category` extraction.
        5.  `reason_for_review`: If confidence is below 80, briefly explain why (e.g., "blurry text", "unclear entity"). Otherwise, set to null.

        REQUIRED JSON FORMAT:
        {{
          "entity": "string",
          "category": "string",
          "date": "YYYY-MM-DD",
          "confidence_score": "integer",
          "reason_for_review": "string or null"
        }}
        """
        response = model.generate_content(prompt)
        # It's safer to parse the JSON inside the try block
        analysis_result = json.loads(response.text)
        return analysis_result

    except json.JSONDecodeError:
        logging.error(f"Gemini API returned invalid JSON. Response text: {response.text}")
        return error_response
    except Exception as e:
        logging.error(f"Gemini API call failed: {e}")
        return error_response

def sanitize_filename(name: str) -> str:
    """Cleans a string to be safe for use as a filename."""
    # Replace problematic characters with an underscore
    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '.') else '_' for c in name)
    # Replace spaces with underscores and remove any leading/trailing whitespace/underscores
    return safe_name.strip().replace(" ", "_").strip('_')

def main():
    """Main script execution flow."""
    # Check for essential settings
    if not all([GOOGLE_API_KEY, TESSERACT_PATH, SOURCE_FOLDER, RENAMED_FOLDER, NEEDS_REVIEW_FOLDER]):
        logging.critical("CRITICAL ERROR: A required setting is missing! Please check your .env file.")
        return

    # Create target directories if they don't exist
    for folder in [RENAMED_FOLDER, NEEDS_REVIEW_FOLDER]:
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as e:
            logging.critical(f"Could not create directory '{folder}'. Error: {e}")
            return
    
    logging.info(f"Scanning folder: '{SOURCE_FOLDER}'")
    
    if not os.path.isdir(SOURCE_FOLDER):
        logging.critical(f"Source folder '{SOURCE_FOLDER}' does not exist or is not a directory.")
        return

    for filename in os.listdir(SOURCE_FOLDER):
        file_path = os.path.join(SOURCE_FOLDER, filename)
        if os.path.isdir(file_path) or filename.startswith('.'):
            continue

        logging.info(f"\n[Processing] -> {filename}")
        
        try:
            text = extract_text_from_file(file_path)
        except Exception:
            # Errors from extract_text_from_file are already logged.
            # Move the file for manual review.
            shutil.move(file_path, os.path.join(NEEDS_REVIEW_FOLDER, filename))
            continue

        if not text:
            logging.warning("  -> Could not extract text. Moving to needs review folder.")
            shutil.move(file_path, os.path.join(NEEDS_REVIEW_FOLDER, filename))
            continue
            
        analysis = analyze_text_with_gemini(text)
        
        confidence = analysis.get('confidence_score', 0)
        entity_name = analysis.get('entity')
        reason = analysis.get('reason_for_review')
        
        log_message = f"  -> Analysis: Category={analysis.get('category')}, Entity='{entity_name}', Confidence={confidence}%"
        if reason:
            log_message += f", Reason: {reason}"
        logging.info(log_message)

        if confidence >= CONFIDENCE_THRESHOLD and entity_name:
            # Rename the file
            new_name_parts = {
                'category': analysis.get('category', 'Unknown'),
                'entity': entity_name,
                'date': analysis.get('date', ''),
                'original_name': os.path.splitext(filename)[0]
            }
            new_filename_base = FILENAME_FORMAT.format(**new_name_parts)
            ext = os.path.splitext(filename)[1]
            
            # Sanitize the entire filename base
            safe_filename_base = sanitize_filename(new_filename_base)
            if not safe_filename_base: # Handle cases where sanitization results in an empty string
                safe_filename_base = f"unnamed_document_{new_name_parts['date']}"

            safe_filename = f"{safe_filename_base}{ext}"
            target_path = os.path.join(RENAMED_FOLDER, safe_filename)
            
            # Prevent overwriting existing files
            counter = 1
            while os.path.exists(target_path):
                base, extension = os.path.splitext(safe_filename)
                # To avoid infinitely adding _1, _2 etc., we remove previous suffixes if they exist
                base = base.rsplit('_', 1)[0] if base.endswith(f"_{counter-1}") else base
                target_path = os.path.join(RENAMED_FOLDER, f"{base}_{counter}{extension}")
                counter += 1

            shutil.move(file_path, target_path)
            logging.info(f"  -> [SUCCESS] File renamed and moved to '{os.path.basename(target_path)}'")
        else:
            # Move to needs review folder
            shutil.move(file_path, os.path.join(NEEDS_REVIEW_FOLDER, filename))
            review_reason = "Low confidence" if confidence < CONFIDENCE_THRESHOLD else "Entity not found"
            logging.info(f"  -> [REVIEW NEEDED] {review_reason}. File moved for manual review.")

    logging.info("\nAll operations completed.")

if __name__ == "__main__":
    main()
