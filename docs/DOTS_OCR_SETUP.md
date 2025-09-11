# dots.ocr Setup Instructions

## What is dots.ocr?

dots.ocr is a powerful, multilingual document parser that unifies layout detection and content recognition within a single vision-language model. It provides significant improvements over traditional OCR solutions like Tesseract:

- **SOTA Performance**: Achieves state-of-the-art results on text, tables, and reading order
- **Multilingual Support**: Works with 100+ languages including low-resource languages
- **Layout Understanding**: Detects document structure, tables, formulas, and maintains reading order
- **Efficient**: Built on a compact 1.7B parameter model for faster inference

## Installation Steps

### 1. Clone dots.ocr Repository

```bash
git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr
```

### 2. Install PyTorch

Install PyTorch according to your system. For CUDA-enabled systems:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only systems:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install dots.ocr

```bash
pip install -e .
```

### 4. Download Model Weights

```bash
python3 tools/download_model.py
```

This will create a `weights/DotsOCR` directory with the model files.

### 5. Update Your .env File

Copy `.env.example` to `.env` and update the paths:

```bash
DOTS_OCR_MODEL_PATH="./dots.ocr/weights/DotsOCR"
```

## Performance Comparison

| Model | Overall Performance | Multilingual | Layout Detection |
|-------|-------------------|-------------|------------------|
| Tesseract OCR | Basic text only | Limited | No |
| dots.ocr | SOTA | 100+ languages | Yes |
| GPT-4o | Good | Good | Limited |
| Gemini 2.5 Pro | Good | Good | Limited |

## Features Gained

1. **Better Text Recognition**: Especially for complex layouts and multilingual documents
2. **Table Detection**: Automatically detects and processes tables
3. **Formula Recognition**: Handles mathematical formulas and equations
4. **Reading Order**: Maintains proper text flow across columns
5. **Layout Understanding**: Recognizes headers, footers, and document structure

## Troubleshooting

- **GPU Memory**: The model requires significant GPU memory. Use CPU mode if encountering memory issues.
- **Model Path**: Ensure the model path doesn't contain periods (use "DotsOCR" not "dots.ocr")
- **CUDA Version**: Make sure your PyTorch CUDA version matches your system CUDA version