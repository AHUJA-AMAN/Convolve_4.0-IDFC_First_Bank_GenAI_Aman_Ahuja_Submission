
# Intelligent Document AI for Invoice Field Extraction
# Setup and Execution Guide

Please follow these instructions carefully to ensure the environment is configured correctly.

## 🛠 Critical Requirements

1.  **Python Version**: You **must** use **Python 3.10**. Using other versions may cause dependency conflicts with the AI frameworks.
2.  **Dependencies**: Install the `requirements.txt` file **before** attempting to run the executable file.
3.  **Git Installation**: You must have **Git** installed and added to your system **PATH**. 
    > *Note: This is required because specific libraries (like Transformers) are installed directly from GitHub repositories.*

---

## 🚀 Running the Application

To run the file via the Command Line Interface (CLI), follow these steps:

1.  **Navigate to Directory**: Open your terminal and go to the directory where you have stored the submission folder.
    ```bash
    cd path/to/your/submission_folder
    ```

2.  **Execute the Script**: Run the following command, replacing `<img_path>` with the path to your image file:
    ```bash
    python executable.py <img_path>
    ```

---

## ⚠️ Important Notes

* **Initial Run**: The processing of the **first image** will take significantly longer than others.
* **Initialization**: During this first run, the system completes the library installations and sets up the model weights.
* **Internet Access**: An active internet connection is **required for the first image** to download necessary model components. 
* **Subsequent Runs**: After the first run is complete, the application will work offline and much faster for all subsequent images.
## Overview

This project was developed as part of the **Intelligent Document AI for Field Extraction from Invoices** problem statement given by the IDFC bank in Convolve 4.0.
The objective is to build an end-to-end system that extracts structured information from semi-structured invoice/quotation documents, handling variations in layout, language, and document quality.

The system outputs a structured JSON per document containing extracted fields, bounding boxes for visual elements, and confidence estimates, in accordance with the evaluation guidelines.

---

## Extracted Fields

For each input document, the following fields are extracted:

* **Dealer Name** (Text, fuzzy match)
* **Model Name** (Text, fuzzy match)
* **Horse Power** (Numeric, exact match)
* **Asset Cost** (Numeric, exact match)
* **Dealer Signature** (Binary + bounding box)
* **Dealer Stamp** (Binary + bounding box)

### Sample Output

```json
{
    "doc_id": "173177534_1_pg37",
    "fields": {
        "dealer_name": "PrOp.O.S.AUTOMOBILES PRIVATE LIMITED",
        "model_name": "MF 251 32 02",
        "horse_power": 42,
        "asset_cost": 590000,
        "signature": {
            "present": true,
            "bbox": [
                794,
                1055,
                929,
                1119
            ]
        },
        "stamp": {
            "present": true,
            "bbox": [
                669,
                1025,
                966,
                1145
            ]
        }
    },
    "confidence": 0.96,
    "processing_time_sec": 52.841713666915894,
    "cost_estimate_usd": 0
}
```

---

## System Architecture (High Level)

The overall pipeline is divided into two major components:

1. **Textual Field Extraction**
2. **Visual Marker Detection (Signature & Stamp)**

Each component was developed independently and later integrated into a unified inference pipeline.

---

## Part 1: Text-Based Field Extraction



This module is responsible for extracting the following fields:

* Dealer Name
* Model Name
* Horse Power
* Asset Cost

### Approach

Algorithm InvoiceFieldExtraction(image):

    # Step 1: OCR
    ocr_lines = PaddleOCR(image)
    layout_blocks = LayoutDetector(image)

    # Step 2: Rule-Based Extraction
    dealer_candidates = find_dealer_by_layout(ocr_lines)
    model_candidates  = find_model_by_keywords(ocr_lines)
    hp_candidates     = find_hp_by_regex(ocr_lines)
    cost_candidates   = find_cost_by_numbers(ocr_lines)

    # Step 3: Confidence Scoring
    dealer, dealer_conf = score_and_select(dealer_candidates)
    model,  model_conf  = score_and_select(model_candidates)
    hp,     hp_conf     = score_and_select(hp_candidates)
    cost,   cost_conf   = score_and_select(cost_candidates)

    # Step 4: GenAI Fallback (Selective)
    if dealer_conf < THRESHOLD:
        dealer = QwenVL(image_header, prompt="Extract dealer name")

    if model_conf < THRESHOLD:
        model = QwenVL(image_body, prompt="Extract model name")

    if hp_conf < THRESHOLD:
        hp = QwenVL(image_body, prompt="Extract horsepower")

    if cost_conf < THRESHOLD:
        cost = QwenVL(image_footer, prompt="Extract total cost")

    # Step 5: Output
    return {
        "dealer_name": dealer,
        "model_name": model,
        "horse_power": hp,
        "asset_cost": cost
    }


Suggested details to include:

* OCR engine and preprocessing steps
* Text normalization and cleanup
* Rule-based or model-based extraction logic
* Fuzzy matching strategy for dealer name
* Multilingual handling
* Confidence estimation methodology

---

## Part 2: Visual Detection of Signature & Stamp



This module detects the **presence and bounding boxes** of:

* Dealer Signature
* Dealer Stamp

### Motivation

The problem statement emphasizes the lack of ground-truth annotations and recommends **manual sampling and annotation** as a practical strategy.
Inspired by active learning research (Settles, 2009), we adopt the idea that a **small, high-quality annotated dataset** can effectively fine-tune lightweight models for robust performance.

---

### Dataset Creation

* **40 manually annotated images**
* Bounding box annotations for:

  * Signatures
  * Stamps
* Images selected to cover:

  * Diverse layouts
  * Different scan qualities
  * Noise, background artifacts, and ink variations

---

### Model Details

* **Model**: YOLO11s
* **Task**: Object Detection
* **Framework**: Ultralytics YOLO
* **Classes**:

  * `signature`
  * `stamp`

YOLO11s was chosen due to its low inference latency, small model size, and strong performance when trained on limited data.

---

### Training Strategy

* Fine-tuned YOLO11s on the curated dataset
* Applied default data augmentations (scaling, flipping, brightness adjustments)
* Early stopping to prevent overfitting given the small dataset size

---

### Inference & Post-processing

* Model predicts bounding boxes and confidence scores
* Presence flag is set based on a confidence threshold
* Bounding boxes are returned in image coordinates
* Evaluation aligns with hackathon metrics (IoU ≥ 0.5)

---


## Performance & Design Considerations

* Lightweight models for low-cost inference
* Modular architecture for explainability and debugging
* Designed to generalize beyond tractor invoices
* Suitable for CPU or low-tier GPU deployment

---


