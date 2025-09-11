# Enhanced Document Sorter - Testing Guide

## Overview

This guide covers comprehensive testing and validation of the Enhanced Document Sorter system. The testing framework includes automated test suites, document generation, and validation scenarios.

## Test Suite Components

### 1. Core Test Files

- **`test_suite.py`** - Main comprehensive test suite
- **`generate_test_docs.py`** - Test document generator
- **`setup_wizard.py`** - Interactive configuration wizard

### 2. Test Categories

#### A. Configuration Validation Tests
- API key validation
- Model path verification
- Directory structure checks
- System requirements validation
- Performance settings validation

#### B. OCR Functionality Tests
- File type detection and processing
- Text extraction accuracy
- Filename sanitization
- Multi-format document handling
- Fallback mechanism testing

#### C. Error Handling Tests
- Invalid file format handling
- Corrupted document processing
- Network failure recovery
- Memory management under stress
- Graceful degradation scenarios

#### D. Performance Tests
- Processing speed benchmarks
- Memory usage monitoring
- Parallel processing validation
- GPU/CPU utilization tests
- Batch processing efficiency

#### E. Integration Tests
- End-to-end document processing
- AI analysis accuracy
- File organization logic
- Review queue functionality
- Complete workflow validation

## Running Tests

### Quick Test Run

```bash
# Run basic system validation
python test_suite.py
```

### Comprehensive Testing

```bash
# 1. Generate test documents
python generate_test_docs.py --output test_docs

# 2. Run configuration wizard (optional)
python setup_wizard.py

# 3. Run full test suite
python test_suite.py

# 4. Run actual system with test documents
python main_enhanced.py
```

### Custom Test Scenarios

```bash
# Generate specific number of test documents
python generate_test_docs.py --count 50 --output custom_test_docs

# Run tests with custom configuration
SOURCE_FOLDER=custom_test_docs python test_suite.py
```

## Test Document Types

### Generated Documents

1. **PDF Documents** (10 files)
   - Invoices with realistic content
   - Contracts with company information
   - Reports with structured data
   - Mixed language content

2. **DOCX Documents** (5 files)
   - Business letters
   - Service agreements
   - Monthly reports
   - Correspondence

3. **Image Documents** (5 files)
   - Scanned invoices
   - Receipt images
   - ID card mockups
   - Mixed quality images

4. **Edge Cases** (4 files)
   - Empty documents
   - Corrupted files
   - Long filenames
   - Special characters

### Document Content Examples

#### Invoice Document
```
ACME Corporation
Invoice #12345
Date: 2024-01-15
Amount: $1,500.00
Due Date: 2024-02-14

Items:
- Software License: $1,200.00
- Support Services: $300.00
```

#### Contract Document
```
Service Agreement

Company: TechFlow Ltd
Contact: John Smith
Date: February 1, 2024

Contract Duration: 24 months
Contract Value: $50,000
```

## Validation Criteria

### Success Metrics

1. **Text Extraction Rate**: >95% for standard documents
2. **Categorization Accuracy**: >85% correct categories
3. **Entity Extraction**: >90% company/person names identified
4. **Date Parsing**: >95% dates correctly formatted
5. **File Processing**: >98% files processed without crashes

### Performance Benchmarks

- **Processing Speed**: <30 seconds per document average
- **Memory Usage**: <4GB peak usage
- **GPU Utilization**: 60-90% when available
- **Error Recovery**: <5% unrecoverable failures

## Test Results Interpretation

### Test Output Format

```
===============================================================
Enhanced Document Sorter - Test Suite
===============================================================
Starting comprehensive system validation...

Configuration Validation:
✓ PASS Valid Configuration
✓ PASS Missing API Key Detection

OCR Functionality:
✓ PASS Filename Sanitization (Tested 5 cases)
✓ PASS File Type Processing (Processed 3/3 file types)

Error Handling:
✓ PASS Invalid File Type Handling
✓ PASS Corrupted File Handling

Performance Monitoring:
✓ PASS Monitor Initialization
✓ PASS Statistics Tracking (Tracked 2 files correctly)

Integration Tests:
✓ PASS End-to-End Processing (Successfully processed 3/3 files)

===============================================================
Test Summary
===============================================================
Total Tests: 10
Passed: 10
Failed: 0
Success Rate: 100.0%

✓ System validation PASSED - Ready for production use
```

### Success Rate Interpretation

- **90-100%**: System ready for production
- **70-89%**: System functional with minor issues
- **50-69%**: System requires significant fixes
- **<50%**: System not ready for use

## Troubleshooting Test Failures

### Common Issues and Solutions

#### 1. Model Loading Failures
```
Error: Failed to load dots.ocr model
```
**Solution**: 
- Check DOTS_OCR_MODEL_PATH in .env
- Run dots.ocr installation steps
- Verify model weights downloaded

#### 2. GPU Memory Issues
```
Error: CUDA out of memory
```
**Solution**:
- Reduce GPU_MEMORY_FRACTION in .env
- Enable CPU-only mode
- Reduce batch size

#### 3. API Key Issues
```
Error: Google API Key is not configured
```
**Solution**:
- Set GOOGLE_API_KEY in .env file
- Verify key is valid
- Check API quota limits

#### 4. File Permission Errors
```
Error: Permission denied accessing folder
```
**Solution**:
- Check folder permissions
- Run with appropriate user rights
- Verify folder paths exist

## Advanced Testing

### Stress Testing

```bash
# Generate large document set
python generate_test_docs.py --count 100 --output stress_test

# Run with performance monitoring
MAX_WORKERS=4 python main_enhanced.py
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Run with memory profiling
python -m memory_profiler main_enhanced.py
```

### Custom Validation Scripts

Create custom validation for specific use cases:

```python
#!/usr/bin/env python3
# custom_validation.py

import os
from pathlib import Path

def validate_specific_documents():
    """Custom validation for specific document types."""
    test_dir = Path("custom_test_docs")
    
    # Your specific validation logic here
    for file in test_dir.glob("*.pdf"):
        # Process and validate specific requirements
        pass

if __name__ == "__main__":
    validate_specific_documents()
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Document Sorter Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.12
    
    - name: Install dependencies
      run: |
        pip install -r requirements_enhanced.txt
    
    - name: Generate test documents
      run: |
        python generate_test_docs.py
    
    - name: Run test suite
      run: |
        python test_suite.py
      env:
        GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
```

## Test Data Privacy

### Important Notes

- All test documents contain **synthetic data only**
- No real personal or business information is used
- Test files are automatically generated
- Safe to share and distribute for testing purposes

### Data Cleanup

```bash
# Remove all test documents
rm -rf test_documents/
rm -rf test_docs/
rm -rf logs/
```

## Reporting Issues

When reporting test failures, please include:

1. **Test output logs** (from test_suite.py)
2. **System specifications** (OS, Python version, GPU)
3. **Configuration file** (.env contents, with API keys redacted)
4. **Error stack traces** (from log files)
5. **Test document types** that failed

## Next Steps

After successful testing:

1. **Production Setup**: Use setup_wizard.py for real configuration
2. **Monitor Performance**: Check logs/ directory for ongoing monitoring
3. **Scale Testing**: Test with larger document volumes
4. **Custom Categories**: Modify DOCUMENT_CATEGORIES for your needs
5. **Integration**: Integrate with your existing document workflow