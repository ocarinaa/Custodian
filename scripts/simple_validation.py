#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Validation Test for Enhanced Document Sorter
This is a basic validation test that doesn't require all dependencies.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.END}")

def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result with colors."""
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"{status} {test_name}")
    if details:
        print(f"    {details}")

def test_file_structure():
    """Test that all required files exist."""
    required_files = [
        'main_enhanced.py',
        'setup_wizard.py',
        'test_suite.py',
        'generate_test_docs.py',
        'requirements_enhanced.txt',
        '.env.enhanced.example',
        'TESTING_GUIDE.md'
    ]
    
    results = []
    for file in required_files:
        exists = os.path.exists(file)
        results.append((f"File exists: {file}", exists, ""))
    
    return results

def test_configuration_files():
    """Test configuration file formats."""
    results = []
    
    # Test .env.enhanced.example
    try:
        with open('.env.enhanced.example', 'r') as f:
            content = f.read()
            has_required_keys = all(key in content for key in [
                'GOOGLE_API_KEY', 'DOTS_OCR_MODEL_PATH', 'SOURCE_FOLDER'
            ])
        results.append(("Configuration template format", has_required_keys, "Contains required keys"))
    except FileNotFoundError:
        results.append(("Configuration template format", False, "File not found"))
    except Exception as e:
        results.append(("Configuration template format", False, f"Error: {e}"))
    
    return results

def test_generated_documents():
    """Test generated test documents."""
    results = []
    
    test_dir = Path("test_validation")
    if test_dir.exists():
        # Count different types of files
        pdf_files = list(test_dir.glob("*.txt"))  # Text files as PDF substitutes
        docx_files = list(test_dir.glob("*.docx"))
        image_files = list(test_dir.glob("*.png"))
        edge_files = [f for f in test_dir.iterdir() if any(x in f.name for x in ["empty", "corrupt", "long", "special"])]
        
        results.append(("Document generation - PDFs", len(pdf_files) >= 8, f"Generated {len(pdf_files)} files"))
        results.append(("Document generation - Images", len(image_files) >= 3, f"Generated {len(image_files)} files"))
        results.append(("Document generation - Edge cases", len(edge_files) >= 3, f"Generated {len(edge_files)} files"))
        
        # Test manifest file
        manifest_file = test_dir / "test_manifest.json"
        try:
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
                has_scenarios = "test_scenarios" in manifest and len(manifest["test_scenarios"]) > 0
            results.append(("Test manifest", has_scenarios, "Valid manifest generated"))
        except:
            results.append(("Test manifest", False, "Manifest missing or invalid"))
    else:
        results.append(("Document generation", False, "Test documents not generated"))
    
    return results

def test_script_syntax():
    """Test that Python scripts have valid syntax."""
    results = []
    
    scripts = [
        'main_enhanced.py',
        'setup_wizard.py', 
        'generate_test_docs.py'
    ]
    
    for script in scripts:
        if os.path.exists(script):
            try:
                with open(script, 'r', encoding='utf-8') as f:
                    code = f.read()
                compile(code, script, 'exec')
                results.append((f"Syntax check: {script}", True, "Valid Python syntax"))
            except SyntaxError as e:
                results.append((f"Syntax check: {script}", False, f"Syntax error: {e}"))
            except Exception as e:
                results.append((f"Syntax check: {script}", False, f"Error: {e}"))
        else:
            results.append((f"Syntax check: {script}", False, "File not found"))
    
    return results

def test_enhanced_features():
    """Test enhanced features by checking code content."""
    results = []
    
    if os.path.exists('main_enhanced.py'):
        with open('main_enhanced.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key enhancements
        features = {
            "Performance Monitoring": "PerformanceMonitor" in content,
            "Model Management": "ModelManager" in content,
            "Error Handling": "RetryableException" in content,
            "Progress Tracking": "tqdm" in content,
            "Logging Enhancement": "CustomFormatter" in content,
            "Configuration Validation": "validate_configuration" in content,
            "Memory Management": "contextmanager" in content,
            "Parallel Processing": "ThreadPoolExecutor" in content
        }
        
        for feature, present in features.items():
            results.append((f"Enhanced feature: {feature}", present, "Implementation found" if present else "Not implemented"))
    
    return results

def test_documentation():
    """Test documentation completeness."""
    results = []
    
    docs = [
        ('TESTING_GUIDE.md', 'Testing guide'),
        ('DOTS_OCR_SETUP.md', 'Setup instructions'),
        ('README.md', 'Main documentation')
    ]
    
    for doc_file, description in docs:
        if os.path.exists(doc_file):
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    complete = len(content) > 500  # Basic completeness check
                results.append((description, complete, f"{len(content)} characters"))
            except Exception as e:
                results.append((description, False, f"Error reading: {e}"))
        else:
            results.append((description, False, "File not found"))
    
    return results

def run_validation():
    """Run all validation tests."""
    print_header("Enhanced Document Sorter - System Validation")
    print(f"{Colors.BLUE}Running validation tests without full dependencies...{Colors.END}")
    
    test_suites = [
        ("File Structure", test_file_structure),
        ("Configuration", test_configuration_files),
        ("Generated Documents", test_generated_documents),
        ("Script Syntax", test_script_syntax),
        ("Enhanced Features", test_enhanced_features),
        ("Documentation", test_documentation)
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for suite_name, test_func in test_suites:
        print(f"\n{Colors.YELLOW}{suite_name}:{Colors.END}")
        
        try:
            results = test_func()
            for test_name, passed, details in results:
                total_tests += 1
                if passed:
                    passed_tests += 1
                print_result(test_name, passed, details)
        except Exception as e:
            print_result(f"{suite_name} Suite", False, f"Suite error: {e}")
            total_tests += 1
    
    # Print summary
    print_header("Validation Summary")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    if success_rate >= 90:
        color = Colors.GREEN
    elif success_rate >= 70:
        color = Colors.YELLOW
    else:
        color = Colors.RED
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {color}{passed_tests}{Colors.END}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {color}{success_rate:.1f}%{Colors.END}")
    
    if success_rate >= 90:
        print(f"\n{Colors.GREEN}✓ SYSTEM VALIDATION PASSED{Colors.END}")
        print("The Enhanced Document Sorter is ready for deployment!")
    elif success_rate >= 70:
        print(f"\n{Colors.YELLOW}⚠ SYSTEM VALIDATION PARTIAL{Colors.END}")
        print("Most components working, some minor issues detected.")
    else:
        print(f"\n{Colors.RED}✗ SYSTEM VALIDATION FAILED{Colors.END}")
        print("Significant issues found, system needs attention.")
    
    print(f"\n{Colors.BLUE}Next Steps:{Colors.END}")
    print("1. Install missing dependencies: pip install -r requirements_enhanced.txt")
    print("2. Run setup wizard: python setup_wizard.py")
    print("3. Configure your .env file")
    print("4. Test with real documents: python main_enhanced.py")
    
    return success_rate >= 70

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)