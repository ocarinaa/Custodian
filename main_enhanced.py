#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Document Sorter with Advanced OCR
This enhanced version includes performance optimization, robust error handling, 
comprehensive logging, and fallback mechanisms.
"""

import os
import shutil
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import traceback
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from contextlib import contextmanager

# Required libraries
import google.generativeai as genai
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import docx
import openpyxl
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF processing
from tqdm import tqdm
import psutil

# Fallback OCR import (optional Tesseract)
try:
    import pytesseract
    from pdf2image import convert_from_path
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract OCR not available - fallback disabled")

# ==============================================================================
# --- ENHANCED SETTINGS SECTION ---
# ==============================================================================
load_dotenv()

# --- SETTINGS FROM .env FILE ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DOTS_OCR_MODEL_PATH = os.getenv("DOTS_OCR_MODEL_PATH", "./weights/DotsOCR")
SOURCE_FOLDER = os.getenv("SOURCE_FOLDER")
RENAMED_FOLDER = os.getenv("RENAMED_FOLDER")
NEEDS_REVIEW_FOLDER = os.getenv("NEEDS_REVIEW_FOLDER")

# Fallback Tesseract settings (optional)
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
POPPLER_PATH = os.getenv("POPPLER_PATH")

# --- ENHANCED PERFORMANCE SETTINGS ---
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))  # Parallel processing threads
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))    # Files per batch
GPU_MEMORY_FRACTION = float(os.getenv("GPU_MEMORY_FRACTION", "0.8"))
ENABLE_FALLBACK = os.getenv("ENABLE_FALLBACK", "true").lower() == "true"
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2"))  # seconds

# --- DIRECTLY EDITABLE SETTINGS ---
DOCUMENT_CATEGORIES = [
    "Invoice", "Bank Statement", "Contract", "Payslip", "ID Card",
    "Passport", "Vehicle Registration", "Property Deed", "Medical Report",
    "Resume (CV)", "Certificate", "Letter", "Receipt", "Tax Document",
    "Insurance Policy", "Legal Document", "Other"
]

FILENAME_FORMAT = "{date}_{entity}_{category}"
CONFIDENCE_THRESHOLD = 75

# ==============================================================================
# --- ENHANCED LOGGING SETUP ---
# ==============================================================================

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors and structured output."""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: blue + "%(asctime)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(levelname)s - %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

# File handler for detailed logs
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
fh = logging.FileHandler(log_dir / f"custodian_{datetime.now().strftime('%Y%m%d')}.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'))
logger.addHandler(fh)

# ==============================================================================
# --- PERFORMANCE MONITORING ---
# ==============================================================================

class PerformanceMonitor:
    """Monitor system performance and processing statistics."""
    
    def __init__(self):
        self.stats = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'review_files': 0,
            'processing_times': [],
            'memory_usage': [],
            'start_time': None,
            'end_time': None
        }
    
    def start_processing(self):
        """Start timing the processing."""
        self.stats['start_time'] = time.time()
        logger.info("Processing started")
    
    def end_processing(self):
        """End timing and log final statistics."""
        self.stats['end_time'] = time.time()
        total_time = self.stats['end_time'] - self.stats['start_time']
        
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"Successful: {self.stats['successful_files']}")
        logger.info(f"Failed: {self.stats['failed_files']}")
        logger.info(f"Needs review: {self.stats['review_files']}")
        logger.info(f"Total processing time: {timedelta(seconds=int(total_time))}")
        
        if self.stats['processing_times']:
            avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            logger.info(f"Average time per file: {avg_time:.2f} seconds")
        
        if self.stats['total_files'] > 0:
            success_rate = (self.stats['successful_files'] / self.stats['total_files']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
    
    def log_file_processed(self, filename: str, success: bool, processing_time: float, needs_review: bool = False):
        """Log a processed file."""
        self.stats['total_files'] += 1
        self.stats['processing_times'].append(processing_time)
        
        if success and not needs_review:
            self.stats['successful_files'] += 1
        elif needs_review:
            self.stats['review_files'] += 1
        else:
            self.stats['failed_files'] += 1
        
        # Log memory usage
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.stats['memory_usage'].append(memory_mb)
        
        logger.debug(f"File: {filename}, Success: {success}, Time: {processing_time:.2f}s, Memory: {memory_mb:.1f}MB")

monitor = PerformanceMonitor()

# ==============================================================================
# --- ENHANCED MODEL MANAGEMENT ---
# ==============================================================================

class ModelManager:
    """Enhanced model management with caching and memory optimization."""
    
    def __init__(self):
        self.dots_ocr_model = None
        self.dots_ocr_processor = None
        self.model_loaded = False
        self.device = None
        self._lock = threading.Lock()
    
    def initialize_dots_ocr(self) -> bool:
        """Initialize dots.ocr model with enhanced error handling."""
        if self.model_loaded:
            return True
        
        with self._lock:
            if self.model_loaded:  # Double-check pattern
                return True
            
            try:
                logger.info("Loading dots.ocr model...")
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
                if self.device == "cuda":
                    # Set memory fraction for GPU
                    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
                    logger.info(f"GPU memory fraction set to {GPU_MEMORY_FRACTION}")
                
                # Load model with optimizations
                self.dots_ocr_model = AutoModelForCausalLM.from_pretrained(
                    DOTS_OCR_MODEL_PATH,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )
                
                self.dots_ocr_processor = AutoProcessor.from_pretrained(
                    DOTS_OCR_MODEL_PATH,
                    trust_remote_code=True
                )
                
                # Optimize model for inference
                if self.device == "cuda":
                    self.dots_ocr_model.eval()
                    self.dots_ocr_model = torch.compile(self.dots_ocr_model, mode="reduce-overhead")
                
                self.model_loaded = True
                logger.info(f"dots.ocr model loaded successfully on {self.device}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load dots.ocr model: {e}")
                logger.error(f"Model path: {DOTS_OCR_MODEL_PATH}")
                logger.error(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "No GPU available")
                return False
    
    @contextmanager
    def model_inference(self):
        """Context manager for model inference with memory management."""
        try:
            yield self
        finally:
            # Clean up GPU cache after each inference
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model_loaded:
            logger.info("Cleaning up model resources...")
            del self.dots_ocr_model
            del self.dots_ocr_processor
            self.model_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

model_manager = ModelManager()

# ==============================================================================
# --- ENHANCED ERROR HANDLING ---
# ==============================================================================

class RetryableException(Exception):
    """Exception that can be retried."""
    pass

class NonRetryableException(Exception):
    """Exception that should not be retried."""
    pass

def retry_on_failure(max_attempts: int = RETRY_ATTEMPTS, delay: int = RETRY_DELAY):
    """Decorator for retrying failed operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except NonRetryableException:
                    # Don't retry non-retryable exceptions
                    raise
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        logger.warning(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator

# ==============================================================================
# --- ENHANCED OCR FUNCTIONS ---
# ==============================================================================

@retry_on_failure()
def extract_text_with_dots_ocr(image) -> str:
    """Extract text from an image using dots.ocr model with enhanced error handling."""
    try:
        if not model_manager.initialize_dots_ocr():
            raise RetryableException("Failed to initialize dots.ocr model")
        
        with model_manager.model_inference():
            # Prepare the prompt for OCR task
            prompt = "Please perform OCR on this document and extract all the text content. Provide the extracted text in a clean, readable format."
            
            # Process the image and prompt
            inputs = model_manager.dots_ocr_processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to the same device as the model
            device = next(model_manager.dots_ocr_model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model_manager.dots_ocr_model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=model_manager.dots_ocr_processor.tokenizer.eos_token_id
                )
            
            # Decode the response
            response = model_manager.dots_ocr_processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated text (remove the prompt part)
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            if not response or len(response) < 10:
                raise RetryableException("dots.ocr returned insufficient text")
            
            return response
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error("GPU out of memory - falling back to CPU or smaller batch")
        raise RetryableException(f"GPU memory error: {e}")
    except Exception as e:
        logger.error(f"dots.ocr processing failed: {e}")
        raise RetryableException(f"OCR processing error: {e}")

def extract_text_with_tesseract_fallback(image) -> str:
    """Fallback OCR using Tesseract."""
    if not TESSERACT_AVAILABLE:
        raise NonRetryableException("Tesseract not available for fallback")
    
    try:
        if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        
        text = pytesseract.image_to_string(image, lang='eng+deu+tur')
        return text.strip()
    
    except Exception as e:
        logger.error(f"Tesseract fallback failed: {e}")
        raise NonRetryableException(f"Fallback OCR failed: {e}")

def extract_text_from_file_enhanced(file_path: str) -> Tuple[str, Dict]:
    """Enhanced text extraction with comprehensive error handling and fallback."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    metadata = {
        'file_type': ext,
        'file_size': os.path.getsize(file_path),
        'ocr_method': None,
        'processing_time': 0,
        'error_occurred': False,
        'error_message': None
    }
    
    start_time = time.time()
    
    try:
        if ext == '.pdf':
            # Enhanced PDF processing
            doc = fitz.open(file_path)
            metadata['page_count'] = len(doc)
            
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    # Try text extraction first (for text-based PDFs)
                    page_text = page.get_text()
                    
                    if page_text.strip() and len(page_text.strip()) > 50:
                        # Sufficient text extracted directly
                        text += page_text + "\n"
                        metadata['ocr_method'] = 'direct_pdf_text'
                    else:
                        # Render page as image for OCR
                        pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72))
                        img_data = pix.tobytes("png")
                        
                        from io import BytesIO
                        image = Image.open(BytesIO(img_data))
                        
                        # Try dots.ocr first
                        try:
                            page_text = extract_text_with_dots_ocr(image)
                            metadata['ocr_method'] = 'dots_ocr'
                        except Exception as e:
                            if ENABLE_FALLBACK:
                                logger.warning(f"dots.ocr failed for page {page_num}, trying fallback: {e}")
                                page_text = extract_text_with_tesseract_fallback(image)
                                metadata['ocr_method'] = 'tesseract_fallback'
                            else:
                                raise
                        
                        if page_text:
                            text += page_text + "\n"
                
                except Exception as e:
                    logger.error(f"Error processing page {page_num} of {os.path.basename(file_path)}: {e}")
                    metadata['error_occurred'] = True
                    metadata['error_message'] = str(e)
                    continue
            
            doc.close()
            
        elif ext in ['.png', '.jpg', '.jpeg']:
            # Enhanced image processing
            image = Image.open(file_path)
            metadata['image_size'] = image.size
            
            # Try dots.ocr first
            try:
                text = extract_text_with_dots_ocr(image)
                metadata['ocr_method'] = 'dots_ocr'
            except Exception as e:
                if ENABLE_FALLBACK:
                    logger.warning(f"dots.ocr failed for {os.path.basename(file_path)}, trying fallback: {e}")
                    text = extract_text_with_tesseract_fallback(image)
                    metadata['ocr_method'] = 'tesseract_fallback'
                else:
                    raise
            
        elif ext == '.docx':
            # Enhanced DOCX processing
            doc = docx.Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            text = "\n".join(paragraphs)
            metadata['ocr_method'] = 'direct_docx'
            metadata['paragraph_count'] = len(paragraphs)
            
        elif ext == '.xlsx':
            # Enhanced Excel processing
            workbook = openpyxl.load_workbook(file_path, read_only=True)
            full_text = []
            sheet_count = 0
            
            for sheet in workbook.worksheets:
                sheet_count += 1
                for row in sheet.iter_rows():
                    row_text = []
                    for cell in row:
                        if cell.value:
                            row_text.append(str(cell.value))
                    if row_text:
                        full_text.append(" | ".join(row_text))
            
            text = "\n".join(full_text)
            metadata['ocr_method'] = 'direct_excel'
            metadata['sheet_count'] = sheet_count
            workbook.close()
        
        else:
            raise NonRetryableException(f"Unsupported file type: {ext}")
    
    except Exception as e:
        logger.error(f"Error processing '{os.path.basename(file_path)}': {e}")
        metadata['error_occurred'] = True
        metadata['error_message'] = str(e)
        raise
    
    finally:
        metadata['processing_time'] = time.time() - start_time
    
    return text.strip(), metadata

# ==============================================================================
# --- ENHANCED GEMINI ANALYSIS ---
# ==============================================================================

@retry_on_failure()
def analyze_text_with_gemini_enhanced(text: str, metadata: Dict = None) -> Dict:
    """Enhanced Gemini analysis with better error handling and context."""
    error_response = {
        "entity": None,
        "category": "Other",
        "date": datetime.now().strftime('%Y-%m-%d'),
        "confidence_score": 0,
        "reason_for_review": "AI analysis failed or returned an invalid format.",
        "processing_metadata": metadata or {}
    }

    if not GOOGLE_API_KEY:
        logger.error("Google API Key is not configured.")
        return error_response

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(
            'gemini-1.5-flash-latest',
            generation_config={"response_mime_type": "application/json"}
        )

        # Enhanced prompt with more context
        enhanced_prompt = f"""
        Analyze the document text below and return ONLY a JSON object.
        
        FILE METADATA:
        - File type: {metadata.get('file_type', 'unknown') if metadata else 'unknown'}
        - OCR method: {metadata.get('ocr_method', 'unknown') if metadata else 'unknown'}
        - Text length: {len(text)} characters

        DOCUMENT TEXT:
        ---
        {text[:4000]}
        ---

        TASKS:
        1. `entity`: Identify the primary person, company, or organization. Be specific and use proper capitalization.
        2. `category`: Classify the document. It MUST be one of these: {json.dumps(DOCUMENT_CATEGORIES)}.
        3. `date`: Find the most relevant date (issue date, due date, etc.) and format as `YYYY-MM-DD`. If no date found, use today: {datetime.now().strftime('%Y-%m-%d')}.
        4. `confidence_score`: Rate confidence (0-100) in entity and category extraction. Consider text quality and clarity.
        5. `reason_for_review`: If confidence < 80, explain why (e.g., "unclear text", "ambiguous entity", "poor OCR quality").
        6. `extracted_info`: Key information found (amounts, dates, reference numbers, etc.)

        REQUIRED JSON FORMAT:
        {{
          "entity": "string",
          "category": "string", 
          "date": "YYYY-MM-DD",
          "confidence_score": integer,
          "reason_for_review": "string or null",
          "extracted_info": {{
            "amounts": ["list of monetary amounts"],
            "reference_numbers": ["list of reference/invoice numbers"],
            "dates": ["list of important dates"]
          }}
        }}
        """
        
        response = model.generate_content(enhanced_prompt)
        analysis_result = json.loads(response.text)
        
        # Add processing metadata
        analysis_result['processing_metadata'] = metadata or {}
        
        # Validate required fields
        required_fields = ['entity', 'category', 'date', 'confidence_score']
        for field in required_fields:
            if field not in analysis_result:
                logger.warning(f"Missing required field '{field}' in Gemini response")
                return error_response
        
        return analysis_result

    except json.JSONDecodeError as e:
        logger.error(f"Gemini API returned invalid JSON: {e}")
        logger.error(f"Response text: {response.text[:500]}...")
        return error_response
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return error_response

# ==============================================================================
# --- ENHANCED UTILITY FUNCTIONS ---
# ==============================================================================

def sanitize_filename_enhanced(name: str) -> str:
    """Enhanced filename sanitization with length limits and better handling."""
    if not name:
        return "unnamed_document"
    
    # Replace problematic characters
    safe_chars = []
    for c in name:
        if c.isalnum():
            safe_chars.append(c)
        elif c in (' ', '-', '.', '_'):
            safe_chars.append('_' if c == ' ' else c)
        else:
            safe_chars.append('_')
    
    safe_name = ''.join(safe_chars)
    
    # Remove consecutive underscores and clean up
    while '__' in safe_name:
        safe_name = safe_name.replace('__', '_')
    
    safe_name = safe_name.strip('._')
    
    # Limit length (Windows has 255 char limit for full path)
    if len(safe_name) > 100:
        safe_name = safe_name[:100]
    
    return safe_name if safe_name else "unnamed_document"

def validate_configuration() -> List[str]:
    """Validate system configuration and return list of issues."""
    issues = []
    
    # Check required settings
    if not GOOGLE_API_KEY:
        issues.append("GOOGLE_API_KEY is missing")
    
    if not DOTS_OCR_MODEL_PATH:
        issues.append("DOTS_OCR_MODEL_PATH is missing")
    elif not os.path.exists(DOTS_OCR_MODEL_PATH):
        issues.append(f"dots.ocr model not found at: {DOTS_OCR_MODEL_PATH}")
    
    if not SOURCE_FOLDER:
        issues.append("SOURCE_FOLDER is missing")
    elif not os.path.exists(SOURCE_FOLDER):
        issues.append(f"Source folder not found: {SOURCE_FOLDER}")
    
    if not RENAMED_FOLDER:
        issues.append("RENAMED_FOLDER is missing")
    
    if not NEEDS_REVIEW_FOLDER:
        issues.append("NEEDS_REVIEW_FOLDER is missing")
    
    # Check system requirements
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 4.0:
            issues.append(f"GPU memory ({gpu_memory:.1f}GB) may be insufficient for dots.ocr")
    
    # Check fallback availability
    if ENABLE_FALLBACK and not TESSERACT_AVAILABLE:
        issues.append("Fallback OCR requested but Tesseract is not available")
    
    return issues

def setup_directories():
    """Create necessary directories with proper error handling."""
    directories = [RENAMED_FOLDER, NEEDS_REVIEW_FOLDER, "logs"]
    
    for directory in directories:
        if directory:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Directory ensured: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory '{directory}': {e}")
                raise

# ==============================================================================
# --- ENHANCED MAIN PROCESSING ---
# ==============================================================================

def process_single_file(file_info: Tuple[str, str]) -> Dict:
    """Process a single file and return results."""
    file_path, filename = file_info
    result = {
        'filename': filename,
        'success': False,
        'needs_review': False,
        'error': None,
        'processing_time': 0,
        'final_path': None,
        'metadata': {}
    }
    
    start_time = time.time()
    
    try:
        logger.info(f"Processing: {filename}")
        
        # Extract text with metadata
        try:
            text, metadata = extract_text_from_file_enhanced(file_path)
            result['metadata'] = metadata
        except Exception as e:
            logger.error(f"Text extraction failed for '{filename}': {e}")
            result['error'] = f"Text extraction failed: {e}"
            # Move to review folder
            shutil.move(file_path, os.path.join(NEEDS_REVIEW_FOLDER, filename))
            result['needs_review'] = True
            result['final_path'] = os.path.join(NEEDS_REVIEW_FOLDER, filename)
            return result

        if not text or len(text.strip()) < 10:
            logger.warning(f"Insufficient text extracted from '{filename}'")
            result['error'] = "Insufficient text extracted"
            shutil.move(file_path, os.path.join(NEEDS_REVIEW_FOLDER, filename))
            result['needs_review'] = True
            result['final_path'] = os.path.join(NEEDS_REVIEW_FOLDER, filename)
            return result
            
        # Analyze with Gemini
        analysis = analyze_text_with_gemini_enhanced(text, metadata)
        
        confidence = analysis.get('confidence_score', 0)
        entity_name = analysis.get('entity')
        reason = analysis.get('reason_for_review')
        category = analysis.get('category', 'Other')
        
        # Log analysis results
        log_message = f"Analysis: Category={category}, Entity='{entity_name}', Confidence={confidence}%"
        if reason:
            log_message += f", Reason: {reason}"
        if 'extracted_info' in analysis:
            extracted = analysis['extracted_info']
            if extracted.get('amounts'):
                log_message += f", Amounts: {extracted['amounts'][:3]}"  # Show first 3
        logger.info(f"  → {log_message}")

        if confidence >= CONFIDENCE_THRESHOLD and entity_name:
            # Rename and move file
            try:
                new_name_parts = {
                    'category': sanitize_filename_enhanced(category),
                    'entity': sanitize_filename_enhanced(entity_name),
                    'date': analysis.get('date', datetime.now().strftime('%Y-%m-%d')),
                    'original_name': sanitize_filename_enhanced(os.path.splitext(filename)[0])
                }
                
                new_filename_base = FILENAME_FORMAT.format(**new_name_parts)
                ext = os.path.splitext(filename)[1]
                
                safe_filename_base = sanitize_filename_enhanced(new_filename_base)
                if not safe_filename_base:
                    safe_filename_base = f"document_{new_name_parts['date']}"

                safe_filename = f"{safe_filename_base}{ext}"
                target_path = os.path.join(RENAMED_FOLDER, safe_filename)
                
                # Handle file conflicts
                counter = 1
                while os.path.exists(target_path):
                    base, extension = os.path.splitext(safe_filename)
                    target_path = os.path.join(RENAMED_FOLDER, f"{base}_{counter:03d}{extension}")
                    counter += 1
                    
                    if counter > 999:  # Prevent infinite loop
                        logger.error(f"Too many file conflicts for {filename}")
                        break

                shutil.move(file_path, target_path)
                result['success'] = True
                result['final_path'] = target_path
                logger.info(f"  → SUCCESS: Renamed to '{os.path.basename(target_path)}'")
                
            except Exception as e:
                logger.error(f"Failed to rename/move '{filename}': {e}")
                result['error'] = f"File operation failed: {e}"
                # Move to review as fallback
                shutil.move(file_path, os.path.join(NEEDS_REVIEW_FOLDER, filename))
                result['needs_review'] = True
                result['final_path'] = os.path.join(NEEDS_REVIEW_FOLDER, filename)
        else:
            # Move to needs review folder
            shutil.move(file_path, os.path.join(NEEDS_REVIEW_FOLDER, filename))
            result['needs_review'] = True
            result['final_path'] = os.path.join(NEEDS_REVIEW_FOLDER, filename)
            
            review_reason = "Low confidence" if confidence < CONFIDENCE_THRESHOLD else "Entity not found"
            logger.info(f"  → REVIEW NEEDED: {review_reason}")

    except Exception as e:
        logger.error(f"Unexpected error processing '{filename}': {e}")
        logger.error(f"Error traceback: {traceback.format_exc()}")
        result['error'] = f"Unexpected error: {e}"
        
        # Try to move file to review folder as fallback
        try:
            if os.path.exists(file_path):
                shutil.move(file_path, os.path.join(NEEDS_REVIEW_FOLDER, filename))
                result['final_path'] = os.path.join(NEEDS_REVIEW_FOLDER, filename)
        except Exception as move_error:
            logger.error(f"Failed to move file to review folder: {move_error}")
    
    finally:
        result['processing_time'] = time.time() - start_time
    
    return result

def main_enhanced():
    """Enhanced main processing function with comprehensive validation and monitoring."""
    logger.info("Starting Enhanced Document Sorter")
    logger.info("=" * 60)
    
    # Validate configuration
    issues = validate_configuration()
    if issues:
        logger.critical("Configuration validation failed:")
        for issue in issues:
            logger.critical(f"  - {issue}")
        return False
    
    logger.info("Configuration validation passed")
    
    # Setup directories
    try:
        setup_directories()
    except Exception as e:
        logger.critical(f"Failed to setup directories: {e}")
        return False
    
    # Start monitoring
    monitor.start_processing()
    
    # Get list of files to process
    try:
        all_files = []
        for filename in os.listdir(SOURCE_FOLDER):
            file_path = os.path.join(SOURCE_FOLDER, filename)
            if os.path.isfile(file_path) and not filename.startswith('.'):
                all_files.append((file_path, filename))
        
        if not all_files:
            logger.warning(f"No files found in source folder: {SOURCE_FOLDER}")
            return True
        
        logger.info(f"Found {len(all_files)} files to process")
        
    except Exception as e:
        logger.critical(f"Failed to scan source folder: {e}")
        return False
    
    # Initialize model before processing
    logger.info("Initializing OCR model...")
    if not model_manager.initialize_dots_ocr():
        if ENABLE_FALLBACK and TESSERACT_AVAILABLE:
            logger.warning("dots.ocr initialization failed, fallback available")
        else:
            logger.critical("OCR initialization failed and no fallback available")
            return False
    
    # Process files
    try:
        if MAX_WORKERS > 1 and len(all_files) > 1:
            # Parallel processing
            logger.info(f"Processing files in parallel with {MAX_WORKERS} workers")
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit tasks
                future_to_file = {
                    executor.submit(process_single_file, file_info): file_info[1] 
                    for file_info in all_files
                }
                
                # Process results with progress bar
                with tqdm(total=len(all_files), desc="Processing files") as pbar:
                    for future in as_completed(future_to_file):
                        filename = future_to_file[future]
                        try:
                            result = future.result()
                            monitor.log_file_processed(
                                filename, 
                                result['success'], 
                                result['processing_time'],
                                result['needs_review']
                            )
                        except Exception as e:
                            logger.error(f"Error in parallel processing for {filename}: {e}")
                            monitor.log_file_processed(filename, False, 0)
                        
                        pbar.update(1)
        else:
            # Sequential processing
            logger.info("Processing files sequentially")
            
            for file_info in tqdm(all_files, desc="Processing files"):
                result = process_single_file(file_info)
                monitor.log_file_processed(
                    result['filename'],
                    result['success'],
                    result['processing_time'],
                    result['needs_review']
                )
    
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        return False
    except Exception as e:
        logger.critical(f"Critical error during processing: {e}")
        logger.critical(f"Error traceback: {traceback.format_exc()}")
        return False
    
    finally:
        # Cleanup
        model_manager.cleanup()
        monitor.end_processing()
    
    logger.info("Enhanced Document Sorter completed successfully")
    return True

if __name__ == "__main__":
    try:
        success = main_enhanced()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        exit(130)
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        exit(1)