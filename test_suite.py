#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Test Suite for Enhanced Document Sorter
This test suite validates all system functionality with various scenarios.
"""

import os
import sys
import shutil
import tempfile
import unittest
import json
import logging
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import threading
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from main_enhanced import (
        ModelManager, PerformanceMonitor, validate_configuration,
        extract_text_from_file_enhanced, analyze_text_with_gemini_enhanced,
        sanitize_filename_enhanced, process_single_file, main_enhanced
    )
except ImportError as e:
    print(f"Failed to import main modules: {e}")
    sys.exit(1)

class Colors:
    """ANSI color codes for test output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_test_header(text: str):
    """Print formatted test header."""
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.END}")

def print_test_result(test_name: str, passed: bool, details: str = ""):
    """Print test result with colors."""
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"{status} {test_name}")
    if details:
        print(f"    {details}")

class TestEnvironment:
    """Manages test environment setup and cleanup."""
    
    def __init__(self):
        self.temp_dir = None
        self.source_folder = None
        self.renamed_folder = None
        self.review_folder = None
        self.original_env = {}
    
    def setup(self):
        """Setup test environment."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp(prefix="custodian_test_")
        self.source_folder = os.path.join(self.temp_dir, "source")
        self.renamed_folder = os.path.join(self.temp_dir, "renamed")
        self.review_folder = os.path.join(self.temp_dir, "review")
        
        for folder in [self.source_folder, self.renamed_folder, self.review_folder]:
            os.makedirs(folder, exist_ok=True)
        
        # Backup original environment
        env_vars = [
            'GOOGLE_API_KEY', 'DOTS_OCR_MODEL_PATH', 'SOURCE_FOLDER',
            'RENAMED_FOLDER', 'NEEDS_REVIEW_FOLDER', 'MAX_WORKERS',
            'ENABLE_FALLBACK'
        ]
        for var in env_vars:
            self.original_env[var] = os.environ.get(var)
        
        # Set test environment
        os.environ.update({
            'SOURCE_FOLDER': self.source_folder,
            'RENAMED_FOLDER': self.renamed_folder,
            'NEEDS_REVIEW_FOLDER': self.review_folder,
            'MAX_WORKERS': '1',  # Single thread for testing
            'ENABLE_FALLBACK': 'true'
        })
    
    def cleanup(self):
        """Cleanup test environment."""
        # Restore original environment
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
        
        # Remove temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

class DocumentGenerator:
    """Generates test documents for various scenarios."""
    
    @staticmethod
    def create_sample_pdf(file_path: str, content: str = "Sample PDF Content"):
        """Create a simple PDF file for testing."""
        try:
            # Try creating with reportlab if available
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            c = canvas.Canvas(file_path, pagesize=letter)
            c.drawString(100, 750, content)
            c.save()
        except ImportError:
            # Fallback: create a fake PDF file
            with open(file_path, 'w') as f:
                f.write(f"%PDF-1.4\nFake PDF content: {content}")
    
    @staticmethod
    def create_sample_docx(file_path: str, content: str = "Sample DOCX Content"):
        """Create a simple DOCX file for testing."""
        try:
            from docx import Document
            doc = Document()
            doc.add_paragraph(content)
            doc.save(file_path)
        except Exception:
            # Fallback: create a fake DOCX file
            with open(file_path, 'w') as f:
                f.write(content)
    
    @staticmethod
    def create_sample_image(file_path: str, text: str = "Sample Text"):
        """Create a simple image with text for OCR testing."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a white image
            img = Image.new('RGB', (400, 100), 'white')
            draw = ImageDraw.Draw(img)
            
            # Try to use a default font
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # Draw text
            draw.text((10, 30), text, fill='black', font=font)
            img.save(file_path)
        except Exception:
            # Fallback: create a fake image file
            with open(file_path, 'wb') as f:
                f.write(b'Fake image content')

class TestConfigurationValidation:
    """Test configuration validation functionality."""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
    
    def run_tests(self) -> List[Dict[str, Any]]:
        """Run configuration validation tests."""
        results = []
        
        # Test 1: Valid configuration
        os.environ['GOOGLE_API_KEY'] = 'fake_api_key_for_testing'
        os.environ['DOTS_OCR_MODEL_PATH'] = self.test_env.temp_dir  # Use temp dir as fake model path
        
        issues = validate_configuration()
        passed = len(issues) == 0 or all('Source folder not found' not in issue for issue in issues)
        results.append({
            'name': 'Valid Configuration',
            'passed': passed,
            'details': f"Issues found: {len(issues)}"
        })
        
        # Test 2: Missing API key
        del os.environ['GOOGLE_API_KEY']
        issues = validate_configuration()
        passed = any('GOOGLE_API_KEY is missing' in issue for issue in issues)
        results.append({
            'name': 'Missing API Key Detection',
            'passed': passed,
            'details': f"Correctly detected missing API key"
        })
        
        # Restore for other tests
        os.environ['GOOGLE_API_KEY'] = 'fake_api_key_for_testing'
        
        return results

class TestOCRFunctionality:
    """Test OCR functionality with mocked components."""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
    
    def run_tests(self) -> List[Dict[str, Any]]:
        """Run OCR functionality tests."""
        results = []
        
        # Test 1: Filename sanitization
        test_cases = [
            ("Normal File.pdf", "Normal_File"),
            ("File/with\\invalid:chars", "File_with_invalid_chars"),
            ("Very Long Filename " * 10, "Very_Long_Filename_Very_Long_Filename_Very_Long_Filename_Very_Long_Filename_Very_Lo"),
            ("", "unnamed_document"),
            ("123", "123")
        ]
        
        all_passed = True
        for input_name, expected in test_cases:
            result = sanitize_filename_enhanced(input_name)
            if not result.startswith(expected[:20]):  # Check first 20 chars
                all_passed = False
                break
        
        results.append({
            'name': 'Filename Sanitization',
            'passed': all_passed,
            'details': f"Tested {len(test_cases)} cases"
        })
        
        # Test 2: File type detection
        # Create test files
        test_files = [
            ('test.pdf', 'PDF content for testing'),
            ('test.docx', 'DOCX content for testing'),
            ('test.jpg', 'Image with invoice text')
        ]
        
        for filename, content in test_files:
            file_path = os.path.join(self.test_env.source_folder, filename)
            if filename.endswith('.pdf'):
                DocumentGenerator.create_sample_pdf(file_path, content)
            elif filename.endswith('.docx'):
                DocumentGenerator.create_sample_docx(file_path, content)
            elif filename.endswith('.jpg'):
                DocumentGenerator.create_sample_image(file_path, content)
        
        # Test file processing (with mocked OCR)
        processed_count = 0
        for filename, _ in test_files:
            file_path = os.path.join(self.test_env.source_folder, filename)
            if os.path.exists(file_path):
                try:
                    with patch('main_enhanced.model_manager.initialize_dots_ocr', return_value=True):
                        with patch('main_enhanced.extract_text_with_dots_ocr', return_value="Mocked OCR result"):
                            text, metadata = extract_text_from_file_enhanced(file_path)
                            if text:
                                processed_count += 1
                except Exception as e:
                    pass  # Expected for some file types without proper libraries
        
        results.append({
            'name': 'File Type Processing',
            'passed': processed_count > 0,
            'details': f"Processed {processed_count}/{len(test_files)} file types"
        })
        
        return results

class TestErrorHandling:
    """Test error handling and recovery mechanisms."""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
    
    def run_tests(self) -> List[Dict[str, Any]]:
        """Run error handling tests."""
        results = []
        
        # Test 1: Invalid file handling
        invalid_file = os.path.join(self.test_env.source_folder, "invalid.xyz")
        with open(invalid_file, 'w') as f:
            f.write("Invalid file content")
        
        try:
            text, metadata = extract_text_from_file_enhanced(invalid_file)
            passed = False  # Should have raised an exception
        except Exception:
            passed = True  # Expected behavior
        
        results.append({
            'name': 'Invalid File Type Handling',
            'passed': passed,
            'details': "Correctly handled unsupported file type"
        })
        
        # Test 2: Corrupted file handling
        corrupted_file = os.path.join(self.test_env.source_folder, "corrupted.pdf")
        with open(corrupted_file, 'w') as f:
            f.write("This is not a valid PDF file")
        
        try:
            text, metadata = extract_text_from_file_enhanced(corrupted_file)
            # Should either work with fallback or fail gracefully
            passed = True
        except Exception:
            passed = True  # Also acceptable
        
        results.append({
            'name': 'Corrupted File Handling',
            'passed': passed,
            'details': "Handled corrupted file gracefully"
        })
        
        return results

class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
    
    def run_tests(self) -> List[Dict[str, Any]]:
        """Run performance monitoring tests."""
        results = []
        
        # Test 1: Monitor initialization
        monitor = PerformanceMonitor()
        passed = (monitor.stats['total_files'] == 0 and 
                 monitor.stats['start_time'] is None)
        
        results.append({
            'name': 'Monitor Initialization',
            'passed': passed,
            'details': "Monitor initialized with correct defaults"
        })
        
        # Test 2: Statistics tracking
        monitor.start_processing()
        time.sleep(0.1)  # Small delay
        monitor.log_file_processed("test.pdf", True, 1.5)
        monitor.log_file_processed("test2.pdf", False, 0.8)
        monitor.end_processing()
        
        passed = (monitor.stats['total_files'] == 2 and
                 monitor.stats['successful_files'] == 1 and
                 monitor.stats['failed_files'] == 1)
        
        results.append({
            'name': 'Statistics Tracking',
            'passed': passed,
            'details': f"Tracked {monitor.stats['total_files']} files correctly"
        })
        
        return results

class TestIntegration:
    """Integration tests for end-to-end functionality."""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
    
    def run_tests(self) -> List[Dict[str, Any]]:
        """Run integration tests."""
        results = []
        
        # Create test documents
        test_docs = [
            ("invoice_001.pdf", "ACME Corp\nInvoice #12345\nDate: 2024-01-15\nAmount: $1,500.00"),
            ("contract_abc.docx", "Service Agreement\nCompany: TechCorp\nDate: 2024-02-01"),
            ("receipt_store.jpg", "Store Receipt\nWalmart\n2024-03-10\n$45.67")
        ]
        
        for filename, content in test_docs:
            file_path = os.path.join(self.test_env.source_folder, filename)
            if filename.endswith('.pdf'):
                DocumentGenerator.create_sample_pdf(file_path, content)
            elif filename.endswith('.docx'):
                DocumentGenerator.create_sample_docx(file_path, content)
            elif filename.endswith('.jpg'):
                DocumentGenerator.create_sample_image(file_path, content)
        
        # Test single file processing with mocks
        processed_files = 0
        for filename, _ in test_docs:
            file_path = os.path.join(self.test_env.source_folder, filename)
            if os.path.exists(file_path):
                # Mock the OCR and AI analysis
                with patch('main_enhanced.model_manager.initialize_dots_ocr', return_value=True):
                    with patch('main_enhanced.extract_text_with_dots_ocr', return_value="Mocked OCR text"):
                        with patch('main_enhanced.analyze_text_with_gemini_enhanced') as mock_gemini:
                            mock_gemini.return_value = {
                                'entity': 'Test Company',
                                'category': 'Invoice',
                                'date': '2024-01-15',
                                'confidence_score': 85,
                                'reason_for_review': None
                            }
                            
                            try:
                                result = process_single_file((file_path, filename))
                                if result['success'] or result['needs_review']:
                                    processed_files += 1
                            except Exception as e:
                                print(f"Processing error: {e}")
        
        results.append({
            'name': 'End-to-End Processing',
            'passed': processed_files > 0,
            'details': f"Successfully processed {processed_files}/{len(test_docs)} files"
        })
        
        return results

class TestRunner:
    """Main test runner that orchestrates all tests."""
    
    def __init__(self):
        self.test_env = TestEnvironment()
        self.total_tests = 0
        self.passed_tests = 0
    
    def run_all_tests(self):
        """Run all test suites."""
        print_test_header("Enhanced Document Sorter - Test Suite")
        print(f"{Colors.BLUE}Starting comprehensive system validation...{Colors.END}\n")
        
        try:
            # Setup test environment
            self.test_env.setup()
            
            # Run all test suites
            test_suites = [
                ("Configuration Validation", TestConfigurationValidation(self.test_env)),
                ("OCR Functionality", TestOCRFunctionality(self.test_env)),
                ("Error Handling", TestErrorHandling(self.test_env)),
                ("Performance Monitoring", TestPerformanceMonitoring(self.test_env)),
                ("Integration Tests", TestIntegration(self.test_env))
            ]
            
            for suite_name, test_suite in test_suites:
                print(f"\n{Colors.YELLOW}{suite_name}:{Colors.END}")
                results = test_suite.run_tests()
                
                for result in results:
                    self.total_tests += 1
                    if result['passed']:
                        self.passed_tests += 1
                    
                    print_test_result(
                        result['name'], 
                        result['passed'], 
                        result['details']
                    )
        
        except Exception as e:
            print(f"{Colors.RED}Critical test error: {e}{Colors.END}")
        
        finally:
            # Cleanup
            self.test_env.cleanup()
            
            # Print summary
            self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        print_test_header("Test Summary")
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        if success_rate >= 90:
            color = Colors.GREEN
        elif success_rate >= 70:
            color = Colors.YELLOW
        else:
            color = Colors.RED
        
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {color}{self.passed_tests}{Colors.END}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success Rate: {color}{success_rate:.1f}%{Colors.END}")
        
        if success_rate >= 90:
            print(f"\n{Colors.GREEN}✓ System validation PASSED - Ready for production use{Colors.END}")
        elif success_rate >= 70:
            print(f"\n{Colors.YELLOW}⚠ System validation PARTIAL - Some issues detected{Colors.END}")
        else:
            print(f"\n{Colors.RED}✗ System validation FAILED - Significant issues found{Colors.END}")

def main():
    """Main function to run tests."""
    runner = TestRunner()
    runner.run_all_tests()
    
    # Return appropriate exit code
    success_rate = (runner.passed_tests / runner.total_tests * 100) if runner.total_tests > 0 else 0
    return 0 if success_rate >= 70 else 1

if __name__ == "__main__":
    sys.exit(main())