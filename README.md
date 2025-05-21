# 📝 Handwritten OCR with TrOCR

A Streamlit web application that uses Microsoft's TrOCR model to recognize handwritten text from images.

## Features

- 🖼️ Upload and process handwritten text images
- 🤖 Powered by Microsoft's TrOCR model
- 🚀 GPU acceleration support
- 📱 User-friendly web interface
- ⚡ Fast inference with model caching

## Setup

1. Clone the repository:
```bash
git clone https://github.com/lakshaybishnoi/Handwritten-Text-Recognization-Using-TrOCR.git
cd Handwritten-Text-Recognization-Using-TrOCR
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Running the App

1. Activate the virtual environment (if not already activated):
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser at `http://localhost:8501`

## First Run

- The first time you run the app, it will download the TrOCR model (about 1.75GB)
- This might take a few minutes depending on your internet connection
- Subsequent runs will be much faster as the model is cached locally

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- See `requirements.txt` for Python package dependencies

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── evaluate.py         # Evaluation script
├── main.py            # CLI interface
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## License

MIT License

## Acknowledgments

- Microsoft's TrOCR model
- Hugging Face Transformers library
- Streamlit framework 