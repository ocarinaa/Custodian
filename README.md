# Simple Document Sorter

This Python script scans a folder of your documents (`.pdf`, `.docx`, `.xlsx`, images), understands their content using `Tesseract OCR` and `Google Gemini` AI, automatically renames them according to your rules, and archives them into corresponding folders.

## Features

-   **Multi-Format Support:** Processes PDF, DOCX, XLSX, PNG, and JPG files.
-   **OCR Capability:** Extracts text from images and scanned PDFs using `Tesseract OCR`.
-   **AI-Powered Analysis:** Uses `Google Gemini` to categorize documents and extract key information like the relevant entity (person/company) and date.
-   **Smart Archiving:** Automatically renames and moves files with high-confidence analysis results.
-   **Manual Review Queue:** Isolates low-confidence or problematic files into a separate folder for your manual review.
-   **Fully Customizable:** Easily define your own document categories and file naming format right inside the script.
-   **100% Private:** All processing happens on your local machine. Only the extracted text content is sent to the Google Gemini API for analysis.

---

## Setup

### Step 1: Install Prerequisites

1.  **Python:** Ensure you have **Python 3.9 or newer** installed on your system.
2.  **Tesseract OCR:**
    *   **Windows:** Download and install the installer from [this link](https://github.com/UB-Mannheim/tesseract/wiki). During installation, make sure to add any additional languages you need (e.g., Turkish (`tur`), German (`deu`)) by checking them under "Additional language data".
    *   **macOS:** `brew install tesseract`
    *   **Linux:** `sudo apt-get install tesseract-ocr`
3.  **Poppler (for PDF processing):**
    *   **Windows:** Download the latest release from [this link](https://github.com/oschwartz10612/poppler-windows/releases/), extract the `.zip` file, and take note of the path to the `bin/` directory inside. You will need to add this path to your system's PATH environment variable.
    *   **macOS:** `brew install poppler`
    *   **Linux:** `sudo apt-get install poppler-utils`

    **Important for Windows Users:** After downloading and extracting Poppler, you must either add the `bin` folder to your system's PATH environment variable OR provide the full path to it in the `.env` file (`POPPLER_PATH`).

### Step 2: Set Up the Project

1.  Download or `git clone` this project to your computer.
2.  Open a terminal in the project folder.
3.  Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### Step 1: Configure Your `.env` File

1.  In the project folder, make a copy of the `.env.example` file.
2.  Rename the copy to **`.env`**.
3.  Open this new `.env` file with a text editor and fill in all the required values:
    *   `GOOGLE_API_KEY`: Your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   `TESSERACT_PATH`: The full path to the `tesseract.exe` file on your system (e.g., `C:\\Program Files\\Tesseract-OCR\\tesseract.exe` on Windows).
    *   `SOURCE_FOLDER`: The full path to the folder you want to process.
    *   `RENAMED_FOLDER`: The full path to the folder where successfully processed files will be moved.
    *   `NEEDS_REVIEW_FOLDER`: The full path for files that require your manual attention.
    *   `POPPLER_PATH`: (Optional on Mac/Linux, recommended on Windows) The full path to your Poppler `bin` directory.

### Step 2: (Optional) Customize the Script

*   If you wish, you can open `main.py` and edit the `DOCUMENT_CATEGORIES` list and the `FILENAME_FORMAT` string to better suit your needs.

### Step 3: Run the Script

1.  After saving your settings, run the script from your terminal:
    ```bash
    python main.py
    ```
2.  Watch as the script processes your files one by one and moves them to the appropriate folders.
