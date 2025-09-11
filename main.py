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
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import docx
import openpyxl
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF processing

# ==============================================================================
# --- SETTINGS SECTION ---
# PLEASE CREATE A `.env` FILE BY COPYING `.env.example`
# AND FILL IT WITH YOUR OWN INFORMATION.
# ==============================================================================
load_dotenv()

# --- SETTINGS FROM .env FILE ---

# Your Google Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Full path to the dots.ocr model directory
# Example: DOTS_OCR_MODEL_PATH = "C:\\path\\to\\DotsOCR"
DOTS_OCR_MODEL_PATH = os.getenv("DOTS_OCR_MODEL_PATH", "./weights/DotsOCR")

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

# --- DOTS.OCR INITIALIZATION ---
dots_ocr_model = None
dots_ocr_processor = None

def initialize_dots_ocr():
    """Initialize dots.ocr model and processor."""
    global dots_ocr_model, dots_ocr_processor
    
    if dots_ocr_model is None:
        try:
            logging.info("Loading dots.ocr model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            dots_ocr_model = AutoModelForCausalLM.from_pretrained(
                DOTS_OCR_MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            dots_ocr_processor = AutoProcessor.from_pretrained(
                DOTS_OCR_MODEL_PATH,
                trust_remote_code=True
            )
            
            logging.info(f"dots.ocr model loaded successfully on {device}")
            
        except Exception as e:
            logging.error(f"Failed to load dots.ocr model: {e}")
            logging.error("Please ensure dots.ocr is properly installed and model weights are downloaded")
            raise

# --- CORE FUNCTIONS ---

def extract_text_with_dots_ocr(image) -> str:
    """Extract text from an image using dots.ocr model."""
    try:
        initialize_dots_ocr()
        
        # Prepare the prompt for OCR task
        prompt = "Please perform OCR on this document and extract all the text content. Provide the extracted text in a clean, readable format."
        
        # Process the image and prompt
        inputs = dots_ocr_processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        
        # Move inputs to the same device as the model
        device = next(dots_ocr_model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = dots_ocr_model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                temperature=0.0
            )
        
        # Decode the response
        response = dots_ocr_processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated text (remove the prompt part)
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
        
    except Exception as e:
        logging.error(f"dots.ocr processing failed: {e}")
        return ""

def extract_text_from_file(file_path: str) -> str:
    """Extracts text from various file formats using dots.ocr for images/PDFs and direct extraction for Office docs."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    try:
        if ext == '.pdf':
            # Convert PDF pages to images and process with dots.ocr
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render page as image (200 DPI for better quality)
                pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72))
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                from io import BytesIO
                image = Image.open(BytesIO(img_data))
                
                # Extract text using dots.ocr
                page_text = extract_text_with_dots_ocr(image)
                if page_text:
                    text += page_text + "\n"
            
            doc.close()
            
        elif ext in ['.png', '.jpg', '.jpeg']:
            # Process image directly with dots.ocr
            image = Image.open(file_path)
            text = extract_text_with_dots_ocr(image)
            
        elif ext == '.docx':
            # Direct text extraction for DOCX files
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            
        elif ext == '.xlsx':
            # Direct text extraction for Excel files
            workbook = openpyxl.load_workbook(file_path)
            full_text = []
            for sheet in workbook.worksheets:
                for row in sheet.iter_rows():
                    for cell in row:
                        if cell.value: 
                            full_text.append(str(cell.value))
            text = "\n".join(full_text)
            
    except FileNotFoundError:
        logging.error(f"File not found: '{file_path}'")
    except Exception as e:
        logging.error(f"Error processing '{os.path.basename(file_path)}': {e}")
    
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
    if not all([GOOGLE_API_KEY, DOTS_OCR_MODEL_PATH, SOURCE_FOLDER, RENAMED_FOLDER, NEEDS_REVIEW_FOLDER]):
        logging.critical("CRITICAL ERROR: A required setting is missing! Please check your .env file.")
        return
    
    # Check if dots.ocr model path exists
    if not os.path.exists(DOTS_OCR_MODEL_PATH):
        logging.critical(f"dots.ocr model not found at: {DOTS_OCR_MODEL_PATH}")
        logging.critical("Please run: python3 tools/download_model.py to download the model weights")
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
