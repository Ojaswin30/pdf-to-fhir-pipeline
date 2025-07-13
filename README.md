# 🏥 OCR to FHIR Converter

This project provides a complete pipeline to extract structured clinical data from unstructured medical PDF files using OCR (Optical Character Recognition) and convert it into standardized **FHIR (Fast Healthcare Interoperability Resources)** JSON format. It enables interoperability between legacy medical records and modern healthcare data systems.

---

## 📂 Project Structure

```
.
├── fhir/                           # Output directory for FHIR-compliant JSON files
│   ├── combined_2006.json
│   ├── combined_2008.json
│   └── ...
├── pdf/                            # Input directory containing raw medical PDFs
├── vector_store/                   # (Optional) Vector DB or embeddings for downstream tasks
├── full_medical_summary.csv        # CSV file summarizing structured info extracted from PDFs
├── ocr4_multiple_files.py          # Script to extract text from multiple PDF files using OCR
├── to_fhir.py                      # Script to convert structured data into FHIR-compliant format
├── requirements.txt                # Python dependencies
├── LICENSE
└── README.md
```

---

## 🚀 Features

- ✅ Extracts text from scanned or text-based PDFs using OCR
- ✅ Parses and structures medical information
- ✅ Converts structured data to **FHIR-compliant JSON**
- ✅ Generates a tabular summary (`.csv`) of extracted data
- ✅ Supports multiple files at once
- ✅ Handles both scanned images and text-based PDFs

---

## ⚙️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/ocr-to-fhir.git
cd ocr-to-fhir
```

2. **Create a virtual environment and activate it:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **(Required) Install Tesseract OCR:**
   - **Windows:** Download from https://github.com/tesseract-ocr/tesseract
   - **Ubuntu/Debian:**
     ```bash
     sudo apt update
     sudo apt install tesseract-ocr
     ```
   - **macOS:**
     ```bash
     brew install tesseract
     ```

---

## 🧪 Usage

1. **Place your medical PDF files in the `pdf/` directory.**

2. **Run OCR on all PDFs:**

```bash
python ocr4_multiple_files.py
```

3. **Convert the structured text data to FHIR format:**

```bash
python to_fhir.py
```

4. **View the results:**
   - `fhir/` directory will contain FHIR `.json` outputs
   - `full_medical_summary.csv` will contain a structured CSV summary

---

## 🛠️ Dependencies

Create a `requirements.txt` file with the following dependencies:

```
pytesseract>=0.3.10
pdfplumber>=0.9.0
pandas>=1.5.0
Pillow>=9.0.0
opencv-python>=4.5.0
numpy>=1.21.0
```

---

## 📋 Input/Output Examples

### Input
- Medical PDF files (scanned or text-based)
- Prescription forms, lab reports, discharge summaries, etc.

### Output
- **FHIR JSON files** following HL7 FHIR R4 standard
- **CSV summary** with structured medical data
- Extracted text files for debugging

---

## 🔧 Configuration

You can modify the following parameters in the scripts:

- **OCR Language**: Change language in `ocr4_multiple_files.py`
- **Output Format**: Customize FHIR structure in `to_fhir.py`
- **File Paths**: Update input/output directories as needed

---

## 📊 Supported Data Types

- Patient demographics
- Medical diagnoses
- Medications and prescriptions
- Laboratory results
- Clinical observations
- Healthcare provider information

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙌 Acknowledgements

- Built for healthcare data interoperability and automation
- Inspired by HL7's FHIR standard
- Uses open-source OCR technologies (Tesseract)
- Supports healthcare digitization initiatives

---

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

## ⚠️ Disclaimer

This tool is intended for educational and research purposes. Always ensure compliance with healthcare data privacy regulations (HIPAA, GDPR, etc.) when processing medical records.
