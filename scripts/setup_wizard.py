#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration Setup Wizard for Enhanced Document Sorter
This script helps users configure the system interactively.
"""

import os
import sys
from pathlib import Path
import shutil
from typing import Dict, List, Optional
import json

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_colored(text: str, color: str = Colors.END):
    """Print colored text to terminal."""
    print(f"{color}{text}{Colors.END}")

def print_header(text: str):
    """Print a formatted header."""
    print_colored("\n" + "=" * 60, Colors.BLUE)
    print_colored(text.center(60), Colors.HEADER + Colors.BOLD)
    print_colored("=" * 60, Colors.BLUE)

def print_section(text: str):
    """Print a section header."""
    print_colored(f"\n{text}", Colors.CYAN + Colors.BOLD)
    print_colored("-" * len(text), Colors.CYAN)

def get_user_input(prompt: str, default: str = None, required: bool = True) -> Optional[str]:
    """Get user input with optional default value."""
    if default:
        display_prompt = f"{prompt} [{default}]: "
    else:
        display_prompt = f"{prompt}: "
    
    while True:
        value = input(display_prompt).strip()
        
        if value:
            return value
        elif default:
            return default
        elif not required:
            return None
        else:
            print_colored("This field is required. Please enter a value.", Colors.RED)

def get_yes_no(prompt: str, default: bool = None) -> bool:
    """Get yes/no input from user."""
    if default is True:
        display_prompt = f"{prompt} [Y/n]: "
    elif default is False:
        display_prompt = f"{prompt} [y/N]: "
    else:
        display_prompt = f"{prompt} [y/n]: "
    
    while True:
        value = input(display_prompt).strip().lower()
        
        if value in ['y', 'yes']:
            return True
        elif value in ['n', 'no']:
            return False
        elif default is not None and value == "":
            return default
        else:
            print_colored("Please enter 'y' for yes or 'n' for no.", Colors.RED)

def validate_path(path: str, must_exist: bool = True, create_if_missing: bool = False) -> bool:
    """Validate a file or directory path."""
    if not path:
        return False
    
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        if create_if_missing:
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                print_colored(f"Created directory: {path}", Colors.GREEN)
                return True
            except Exception as e:
                print_colored(f"Failed to create directory: {e}", Colors.RED)
                return False
        else:
            return False
    
    return True

def check_system_requirements() -> Dict[str, bool]:
    """Check system requirements and return status."""
    requirements = {}
    
    print_section("Checking System Requirements")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 9):
        requirements['python'] = True
        print_colored(f"✓ Python {python_version.major}.{python_version.minor} - OK", Colors.GREEN)
    else:
        requirements['python'] = False
        print_colored(f"✗ Python {python_version.major}.{python_version.minor} - Need 3.9+", Colors.RED)
    
    # Check PyTorch
    try:
        import torch
        requirements['torch'] = True
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print_colored(f"✓ PyTorch with CUDA - GPU: {gpu_name}", Colors.GREEN)
        else:
            print_colored("✓ PyTorch (CPU only)", Colors.YELLOW)
    except ImportError:
        requirements['torch'] = False
        print_colored("✗ PyTorch not installed", Colors.RED)
    
    # Check Transformers
    try:
        import transformers
        requirements['transformers'] = True
        print_colored(f"✓ Transformers {transformers.__version__}", Colors.GREEN)
    except ImportError:
        requirements['transformers'] = False
        print_colored("✗ Transformers not installed", Colors.RED)
    
    # Check optional Tesseract
    try:
        import pytesseract
        requirements['tesseract'] = True
        print_colored("✓ Tesseract OCR available (for fallback)", Colors.GREEN)
    except ImportError:
        requirements['tesseract'] = False
        print_colored("✗ Tesseract OCR not available (optional)", Colors.YELLOW)
    
    return requirements

def setup_api_keys() -> Dict[str, str]:
    """Setup API keys."""
    print_section("API Configuration")
    
    config = {}
    
    print("You need a Google Gemini API key for document analysis.")
    print("Get one from: https://aistudio.google.com/app/apikey")
    
    api_key = get_user_input("Enter your Google Gemini API key")
    if api_key and len(api_key) > 20:  # Basic validation
        config['GOOGLE_API_KEY'] = api_key
        print_colored("✓ API key configured", Colors.GREEN)
    else:
        print_colored("⚠ Invalid API key format", Colors.YELLOW)
        config['GOOGLE_API_KEY'] = api_key
    
    return config

def setup_paths() -> Dict[str, str]:
    """Setup file paths."""
    print_section("Directory Configuration")
    
    config = {}
    
    # Source folder
    print("\n1. Source Folder (where your documents are)")
    while True:
        source = get_user_input("Enter path to source folder")
        if validate_path(source, must_exist=True):
            config['SOURCE_FOLDER'] = source
            break
        else:
            print_colored(f"Directory not found: {source}", Colors.RED)
            if get_yes_no("Do you want to create it?"):
                if validate_path(source, must_exist=False, create_if_missing=True):
                    config['SOURCE_FOLDER'] = source
                    break
    
    # Renamed folder
    print("\n2. Output Folder (for successfully processed files)")
    renamed = get_user_input("Enter path for processed files", default=f"{source}_renamed")
    if validate_path(renamed, must_exist=False, create_if_missing=True):
        config['RENAMED_FOLDER'] = renamed
    
    # Review folder
    print("\n3. Review Folder (for files needing manual attention)")
    review = get_user_input("Enter path for review files", default=f"{source}_review")
    if validate_path(review, must_exist=False, create_if_missing=True):
        config['NEEDS_REVIEW_FOLDER'] = review
    
    return config

def setup_ocr_model() -> Dict[str, str]:
    """Setup OCR model configuration."""
    print_section("OCR Model Configuration")
    
    config = {}
    
    print("dots.ocr model configuration:")
    
    # Model path
    default_model_path = "./weights/DotsOCR"
    model_path = get_user_input(
        "Enter path to dots.ocr model", 
        default=default_model_path
    )
    
    if not validate_path(model_path, must_exist=True):
        print_colored(f"Model not found at: {model_path}", Colors.YELLOW)
        print("To download the model:")
        print("1. git clone https://github.com/rednote-hilab/dots.ocr.git")
        print("2. cd dots.ocr")
        print("3. python3 tools/download_model.py")
        
        if get_yes_no("Continue with this path anyway?"):
            config['DOTS_OCR_MODEL_PATH'] = model_path
        else:
            model_path = get_user_input("Enter correct model path")
            config['DOTS_OCR_MODEL_PATH'] = model_path
    else:
        config['DOTS_OCR_MODEL_PATH'] = model_path
        print_colored("✓ Model path validated", Colors.GREEN)
    
    return config

def setup_performance() -> Dict[str, str]:
    """Setup performance settings."""
    print_section("Performance Configuration")
    
    config = {}
    
    # Check system specs
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=False)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"System: {cpu_count} CPU cores, {memory_gb:.1f}GB RAM")
    except ImportError:
        cpu_count = 2
        print("Unable to detect system specifications")
    
    # Max workers
    default_workers = min(2, max(1, cpu_count // 2))
    workers = get_user_input(
        f"Number of parallel processing threads (1-{cpu_count})", 
        default=str(default_workers)
    )
    try:
        workers_int = int(workers)
        if 1 <= workers_int <= cpu_count:
            config['MAX_WORKERS'] = workers
        else:
            config['MAX_WORKERS'] = str(default_workers)
            print_colored(f"Using default: {default_workers} workers", Colors.YELLOW)
    except ValueError:
        config['MAX_WORKERS'] = str(default_workers)
    
    # GPU settings
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU Memory: {gpu_memory:.1f}GB")
            
            default_fraction = "0.8" if gpu_memory >= 8 else "0.6"
            fraction = get_user_input(
                "GPU memory fraction to use (0.1-0.9)", 
                default=default_fraction
            )
            config['GPU_MEMORY_FRACTION'] = fraction
        else:
            print("No GPU detected - using CPU only")
            config['GPU_MEMORY_FRACTION'] = "0.8"
    except ImportError:
        config['GPU_MEMORY_FRACTION'] = "0.8"
    
    # Fallback settings
    fallback = get_yes_no("Enable Tesseract fallback for failed OCR?", default=True)
    config['ENABLE_FALLBACK'] = "true" if fallback else "false"
    
    return config

def create_env_file(config: Dict[str, str], filename: str = ".env") -> bool:
    """Create .env file with configuration."""
    try:
        with open(filename, 'w') as f:
            f.write("# Enhanced Document Sorter Configuration\n")
            f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("# ===== REQUIRED SETTINGS =====\n")
            for key in ['GOOGLE_API_KEY', 'DOTS_OCR_MODEL_PATH', 'SOURCE_FOLDER', 'RENAMED_FOLDER', 'NEEDS_REVIEW_FOLDER']:
                if key in config:
                    f.write(f'{key}="{config[key]}"\n')
            
            f.write("\n# ===== PERFORMANCE SETTINGS =====\n")
            for key in ['MAX_WORKERS', 'GPU_MEMORY_FRACTION', 'ENABLE_FALLBACK']:
                if key in config:
                    f.write(f'{key}={config[key]}\n')
            
            f.write("\n# ===== OPTIONAL SETTINGS =====\n")
            f.write('RETRY_ATTEMPTS=3\n')
            f.write('RETRY_DELAY=2\n')
            f.write('BATCH_SIZE=5\n')
        
        return True
    except Exception as e:
        print_colored(f"Error creating .env file: {e}", Colors.RED)
        return False

def main():
    """Main setup wizard."""
    from datetime import datetime
    
    print_header("Enhanced Document Sorter - Setup Wizard")
    print_colored("This wizard will help you configure the system.", Colors.CYAN)
    
    # Check requirements
    requirements = check_system_requirements()
    missing_requirements = [k for k, v in requirements.items() if not v]
    
    if missing_requirements:
        print_colored(f"\n⚠ Missing requirements: {', '.join(missing_requirements)}", Colors.YELLOW)
        print("Please install missing dependencies before continuing.")
        print("Run: pip install -r requirements_enhanced.txt")
        
        if not get_yes_no("Continue anyway?", default=False):
            return False
    
    # Collect configuration
    all_config = {}
    
    # API keys
    api_config = setup_api_keys()
    all_config.update(api_config)
    
    # Paths
    path_config = setup_paths()
    all_config.update(path_config)
    
    # OCR model
    ocr_config = setup_ocr_model()
    all_config.update(ocr_config)
    
    # Performance
    perf_config = setup_performance()
    all_config.update(perf_config)
    
    # Summary
    print_section("Configuration Summary")
    for key, value in all_config.items():
        if 'API_KEY' in key:
            display_value = value[:8] + "..." if len(value) > 8 else value
        else:
            display_value = value
        print(f"{key}: {display_value}")
    
    # Create .env file
    if get_yes_no("\nSave configuration to .env file?", default=True):
        if create_env_file(all_config):
            print_colored("✓ Configuration saved to .env", Colors.GREEN)
        else:
            print_colored("✗ Failed to save configuration", Colors.RED)
            return False
    
    # Final instructions
    print_header("Setup Complete!")
    print_colored("Next steps:", Colors.GREEN)
    print("1. Ensure dots.ocr model is downloaded")
    print("2. Place documents in your source folder")
    print("3. Run: python main_enhanced.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_colored("\n\nSetup cancelled by user.", Colors.YELLOW)
        sys.exit(130)
    except Exception as e:
        print_colored(f"\nUnexpected error: {e}", Colors.RED)
        sys.exit(1)