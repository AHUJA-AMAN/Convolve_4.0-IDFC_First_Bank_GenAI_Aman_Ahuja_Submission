#How to run in cmd-
# python executable.py C:/invoices/tractor_1.jpg


import torch  
import torchvision
import transformers

# FIX FOR: "is_autocast_enabled() takes no arguments"

original_is_autocast = torch.is_autocast_enabled
def patched_is_autocast(device_type=None):
    # This version of Torch doesn't care about 'device_type'
    # so we just call the original function without it.
    return original_is_autocast()

torch.is_autocast_enabled = patched_is_autocast

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"



import sys

# Strictly require Python 3.10
if sys.version_info.major != 3 or sys.version_info.minor != 10:
    print("CRITICAL ERROR: This submission is optimized specifically for Python 3.10.")
    print(f"You are currently using: {sys.version}")
    print("Please use a Python 3.10 environment to avoid library conflicts.")
    sys.exit(1)


#---------------------------------------------------
#--------------------Input--------------------------
#---------------------------------------------------


# sys.argv[0] is always the name of the script
# sys.argv[1] would be your image path
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    print("No image path provided!")





input=sys.argv[1]

import time

start_time = time.time()



from pathlib import Path

# 1. Capture the path passed from CMD
# sys.argv[1] is the image path (e.g., "C:/images/invoice_001.jpg")
image_input_path = sys.argv[1]

# 2. Extract the doc_id (filename without extension)
# .stem returns 'invoice_001' from 'invoice_001.jpg'
doc_id = Path(image_input_path).stem

print(f"Processing ID: {doc_id}")



#things to output
final_dealer=None
final_cost=None
final_hp=None
final_model=None




# 1. Get the directory where executable.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Point to the 'utils' folder relative to BASE_DIR
UTILS_DIR = os.path.join(BASE_DIR, "utils")

# 3. Define your model paths dynamically
OFFLINE_PATH = os.path.join(UTILS_DIR, "qwen2_vl_offline") # model path for qwen
MODEL_PATH = os.path.join(UTILS_DIR, "best.pt") #model path for YOLO

# Optional: Print for debugging (use stderr so it doesn't break competition graders)
# import sys
# print(f"Loading model from: {OFFLINE_PATH}", file=sys.stderr)


#---------------------------------------------------
#-----------------Libraries-------------------------
#---------------------------------------------------



from paddleocr import PaddleOCR
#from paddleocr.tools.infer.utility import draw_ocr
import cv2 
import matplotlib.pyplot as plt
from paddleocr import PPStructure
import warnings
warnings.filterwarnings('ignore')
from shapely.geometry import box as shapely_box
import numpy as np
import pandas as pd
import re
import math
import easyocr

## use this other wise paddleocr will not work, chinese servers are being blocked
os.environ["PADDLEOCR_DOWNLOAD_SOURCE"] = "huggingface"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

language='en'
img_path=input
img=cv2.imread(img_path)



def run_ocr(img_path):
    ocr = PaddleOCR(use_angle_cls = True,lang=language)
    ocr_result = ocr.ocr(img_path)
    return ocr_result


ocr_result=run_ocr(img_path)

ocr_lines = []
for page in ocr_result:
    for line in page:
        points = line[0]        # 4 points
        text = line[1][0]
        conf = line[1][1]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        ocr_lines.append({
            "text": text,
            "bbox": [x1, y1, x2, y2],
            "conf": conf
        })


def compute_text_vertical_bounds(ocr_lines):
    ys = []
    for line in ocr_lines:
        _, y1, _, y2 = line["bbox"]
        ys.append(y1)
        ys.append(y2)
    return min(ys), max(ys)

def crop_header_body_footer(img, ocr_lines,
                            header_ratio=0.25,
                            footer_ratio=0.30,
                            overlap_ratio=0.08):
    """
    header_ratio  : % of text height considered header
    footer_ratio  : % of text height considered footer
    overlap_ratio : overlap between body & footer
    """

    H, W = img.shape[:2]

    text_top, text_bottom = compute_text_vertical_bounds(ocr_lines)
    text_height = text_bottom - text_top

    header_end = int(text_top + header_ratio * text_height)
    footer_start = int(text_bottom - footer_ratio * text_height)

    overlap = int(overlap_ratio * text_height)

    header_crop = img[0:header_end, :]
    body_crop = img[header_end:footer_start + overlap, :]
    footer_crop = img[footer_start - overlap:H, :]

    return {
        "header": header_crop,
        "body": body_crop,
        "footer": footer_crop
    }

crops = crop_header_body_footer(img, ocr_lines)




def language_detection():
    

    reader = easyocr.Reader(['en', 'hi'])  # English, Hindi, Gujarati
    result = reader.readtext(img_path, detail=0)

    text = " ".join(result)

    from langdetect import detect
    language_detected=detect(text)
    #print(language_detected)
    return detect(text)



dealer_to_qwen=1
dealer_normal_way=1
if language_detection()!='en':

    reader = easyocr.Reader(['hi','en'])  # English, Hindi, Gujarati
    result = reader.readtext(img_path, detail=1)
    ocr_lines_header = []

    for box, text, conf in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        ocr_lines_header.append({
            "text": text,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "conf": conf
        })

    def extract_dealer_from_header(header_img, ocr_lines):
        H, W = header_img.shape[:2]

        heights = [(l["bbox"][3] - l["bbox"][1]) for l in ocr_lines]
        median_height = np.median(heights)

        best_line = None
        best_score = 0

        for line in ocr_lines:
            text = line["text"].strip()
            if len(text) < 3:
                continue

            x1, y1, x2, y2 = line["bbox"]
            h = y2 - y1
            cx = (x1 + x2) / 2

            score = 0

            # 1. Big font (primary signal)
            if h > 1.6 * median_height:
                score += 0.6

            # 2. Centered
            center_dist = abs(cx - W/2) / W
            if center_dist < 0.2:
                score += 0.3

            # 3. Not long (avoid paragraphs)
            if len(text) < 40:
                score += 0.1

            if score > best_score:
                best_score = score
                best_line = line

        if best_line:
            return best_line["text"], best_score
        else:
            return None, 0

    final_dealer,dealer_confidence=extract_dealer_from_header(crops['header'],ocr_lines_header)
    dealer_to_qwen=0
    dealer_normal_way=0
    




#---------------------------------------------------
#-----------------Rule Based Methods----------------
#---------------------------------------------------

#-----------------Dealer Extractor------------------


def dealer_extractor():
    
    PAGE_HEIGHT, PAGE_WIDTH = img.shape[:2]

    heights = []

    for line in ocr_lines:
        x1, y1, x2, y2 = line["bbox"]
        heights.append(y2 - y1)

    MEDIAN_TEXT_HEIGHT = np.median(heights)

    # Features for dealer extractor :-
    def top_score(line, page_height):
        y1 = line["bbox"][1]
        rel_y = y1 / page_height

        if rel_y < 0.10:
            return 1.0
        elif rel_y < 0.20:
            return 0.7
        elif rel_y < 0.30:
            return 0.4
        else:
            return 0.0

    def size_score(line, median_height):
        x1, y1, x2, y2 = line["bbox"]
        height = y2 - y1

        ratio = height / median_height

        if ratio > 2.0:
            return 1.0
        elif ratio > 1.5:
            return 0.7
        elif ratio > 1.2:
            return 0.4
        else:
            return 0.0

    def center_score(line, page_width):
        x1, y1, x2, y2 = line["bbox"]
        text_center = (x1 + x2) / 2
        page_center = page_width / 2

        dist = abs(text_center - page_center) / page_width

        if dist < 0.05:
            return 1.0
        elif dist < 0.10:
            return 0.6
        else:
            return 0.0


    # leaving layout feature over here as title layout not clearly classified

    DEALER_SUFFIXES = [
        "tractors", "tractor", "motors", "sales", "agencies",
        "co", "co.", "company", "corporation", "ltd", "limited",'agro','tech'
    ]

    def suffix_score(line):
        text = line["text"].lower()

        for suf in DEALER_SUFFIXES:
            if text.endswith(suf):
                return 1.0

        return 0.0

    # repition is because sometimes dealer name appears in bank details also
    def repetition_score(line, all_lines):
        text = line["text"].lower()

        count = 0
        for l in all_lines:
            if text in l["text"].lower() and l is not line:
                count += 1

        if count >= 1:
            return 1.0
        else:
            return 0.0


    WEIGHTS = {
        "top": 0.30,
        "size": 0.25,
        "center": 0.10,
        "suffix": 0.30,
        "repeat": 0.05
    }

    def dealer_score(line, all_lines, page_width, page_height, median_height):

        s_top = top_score(line, page_height)
        s_size = size_score(line, median_height)
        s_center = center_score(line, page_width)
        #s_layout = layout_score(line)
        s_suffix = suffix_score(line)
        s_repeat = repetition_score(line, all_lines)

        score = (
            WEIGHTS["top"]    * s_top +
            WEIGHTS["size"]   * s_size +
            WEIGHTS["center"] * s_center +
            #WEIGHTS["layout"] * s_layout +
            WEIGHTS["suffix"] * s_suffix +
            WEIGHTS["repeat"] * s_repeat
        )

        return score


    scored_lines = []

    for line in ocr_lines:
        score = dealer_score(
            line,
            ocr_lines,
            PAGE_WIDTH,
            PAGE_HEIGHT,
            MEDIAN_TEXT_HEIGHT
        )
        scored_lines.append((score, line))

    scored_lines.sort(reverse=True, key=lambda x: x[0])

    best_score, dealer_line = scored_lines[0]

    return dealer_line['text'],best_score 

#calling function
if dealer_normal_way==1:
    rule_based_dealer,rule_based_dealer_confidence=dealer_extractor()


#------------------------COST EXTRACTION---------------------------

PAGE_HEIGHT, PAGE_WIDTH = img.shape[:2]

def extract_numeric_candidates(ocr_lines):

    

    candidates = []
    for line in ocr_lines:
        text = line["text"].replace(",", "")
        nums = re.findall(r"\b\d{4,7}(?:\.\d{1,2})?\b", text)
        for n in nums:
            candidates.append({
                "value": float(n),
                "text": line["text"],
                "bbox": line["bbox"]
            })
    return candidates

COST_KEYWORDS = [
    "total", "amount", "price", "net", "grand",
    "rs", "inr", "₹",
    "कुल", "राशि", "मूल्य", "कुल राशि"
]

from word2number import w2n

def score_cost_candidate(c, page_height):
    score = 0
    y_center = (c["bbox"][1] + c["bbox"][3]) / 2
    y_ratio = y_center / page_height

    text = c["text"].lower()

    # 1. Lower in page
    if y_ratio > 0.6:
        score += 0.25
    if y_ratio > 0.75:
        score += 0.15

    # 2. Has cost keywords nearby
    if any(k in text for k in COST_KEYWORDS):
        score += 0.30

    # 3. Looks like money
    if 20000 <= c["value"] <= 1000000:
        score += 0.20

    # 4. Right aligned
    if c["bbox"][0] > 0.5 * PAGE_WIDTH:
        score += 0.10

    return score

candidates = extract_numeric_candidates(ocr_lines)

best = None
best_score = 0

for c in candidates:
    s = score_cost_candidate(c, PAGE_HEIGHT)
    if s > best_score:
        best_score = s
        best = c

#calling function
rule_based_cost = best["value"] if best else None
rule_based_cost_confidence = best_score


#----------------Model Extractor-------------------------


BRANDS = [
    "mahindra", "tafe", "sonalika", "john deere",'massey',
    "new holland", "farmtrac", "escort", "eicher", "kubota",
    'valdo','poweetrac','swaraj','massey ferguson','new holland',
    'kartar','vst','solis','captain','same deutz fahr','preet','ace',
    'indo farm','trakstar','autonxt','hav','hindustan','mantra','force',
    'escorts','standard','agri king','cellestial','sukoon','maxgreen','marut'
]

MODEL_KEYWORDS = ["tractor", "model", "di", "4wd", "2wd", "series"]
CONF_THRESHOLD = 0.45



def normalize_text(t):
    return re.sub(r"\s+", " ", t.lower().strip())

def contains_brand(text):
    t = normalize_text(text)
    for b in BRANDS:
        if b in t:
            return b
    return None

def mixed_text_number(text):
    return any(c.isalpha() for c in text) and any(c.isdigit() for c in text)

def extract_core_model(text):
    """
    Extract only model part like:
    mahindra 775 di  → 775 di
    tafe 7515       → 7515
    mf 245 di       → 245 di
    """
    t = normalize_text(text)

    # remove brand
    for b in BRANDS:
        t = t.replace(b, "")

    # remove hp
    t = re.sub(r"\b\d{2,3}\s*(hp|horsepower)\b", "", t)

    # extract best alphanumeric chunk
    candidates = re.findall(r"\b\d{2,4}\s*[a-z]{0,3}\b", t)
    if candidates:
        return candidates[0].strip().upper()

    return None


def extract_table_rows(table_block):
    if "cells" in table_block["res"]:
        cells = table_block["res"]["cells"]
        rows = {}
        for cell in cells:
            rows.setdefault(cell["row"], []).append(cell)
        for r in rows:
            rows[r] = sorted(rows[r], key=lambda c: c["bbox"][0])
        return list(rows.values())
    else:
        x1, y1, x2, y2 = table_block["bbox"]
        return [[{
            "text": table_block["res"].get("html", ""),
            "bbox": [x1, y1, x2, y2]
        }]]


def row_vertical_center(row_cells):
    ys = [(c["bbox"][1] + c["bbox"][3]) / 2 for c in row_cells]
    return sum(ys) / len(ys)

def row_contains_hp(row_cells):
    text = " ".join(normalize_text(c["text"]) for c in row_cells)
    return bool(re.search(r"\b\d{2,3}\s*(hp|horsepower)\b", text))

HEADER_WORDS = ["description", "particulars", "model", "qty", "amount", "price"]

def looks_like_header(text):
    hits = sum(1 for w in HEADER_WORDS if w in text)
    return hits >= 2

def score_row_for_model(row_cells, page_width, page_height):
    score = 0
    texts = [normalize_text(c["text"]) for c in row_cells]
    full_text = " ".join(texts)

    if contains_brand(full_text):
        score += 0.25

    if any(k in full_text for k in MODEL_KEYWORDS):
        score += 0.20

    if mixed_text_number(full_text):
        score += 0.20

    if len(row_cells) >= 3:
        score += 0.15

    first_cell = row_cells[0]
    if first_cell["bbox"][0] < 0.3 * page_width:
        score += 0.10

    yc = row_vertical_center(row_cells)
    y_ratio = yc / page_height
    if 0.30 <= y_ratio <= 0.70:
        score += 0.10

    if row_contains_hp(row_cells):
        score += 0.10

    if looks_like_header(full_text):
        score -= 0.30

    return score



def vertical_neighbors(anchor, ocr_lines, max_dist):
    ay = (anchor["bbox"][1] + anchor["bbox"][3]) / 2
    cluster = []
    for line in ocr_lines:
        ly = (line["bbox"][1] + line["bbox"][3]) / 2
        if abs(ly - ay) < max_dist:
            cluster.append(line)
    return cluster


from bs4 import BeautifulSoup
def clean_html(text):
        if "<html" in text:
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text(" ")
        return text


def model_extractor(img, img_path, ocr_lines):
    PAGE_HEIGHT, PAGE_WIDTH = img.shape[:2]
    heights = [l["bbox"][3] - l["bbox"][1] for l in ocr_lines]
    MEDIAN_TEXT_HEIGHT = np.median(heights)

    table_engine = PPStructure(show_log=False)
    layout = table_engine(img_path)

    best_row = None
    best_score = 0

    # ---- TABLE FIRST ----
    for block in layout:
        if block["type"] == "table":
            rows = extract_table_rows(block)
            for row in rows:
                s = score_row_for_model(row, PAGE_WIDTH, PAGE_HEIGHT)
                if s > best_score:
                    best_score = s
                    best_row = row

    # ---- FALLBACK OCR ----
    if best_row is None:
        for line in ocr_lines:
            fake_row = [line]
            s = score_row_for_model(fake_row, PAGE_WIDTH, PAGE_HEIGHT)
            if s > best_score:
                best_score = s
                best_row = fake_row

    if best_row is None:
        return None, None, 0

    full_model_name = " ".join(clean_html(c["text"]) for c in best_row)
    model_core = extract_core_model(full_model_name)


    


    return model_core, full_model_name.strip(), best_score

#calling function
rule_based_model, rule_based_complete_model, rule_based_model_confidence=model_extractor(img,img_path,ocr_lines)


#----------------------HP Extractor-----------------------------

def hp_extractor():
    HP_KEYWORDS = [
        "hp", "h.p", "h p", "horsepower",
        "एच.पी", "एचपी", "एच पी",
        "हा.पा", "हा पा", "हापी",
    ]

    def normalize(text):
        text = text.lower()
        text = text.replace(".", "")
        text = text.replace(" ", "")
        return text


    def bbox_center(b):
        x1, y1, x2, y2 = b
        return ( (x1+x2)/2, (y1+y2)/2 )


    def distance(b1, b2):
        x1, y1 = bbox_center(b1)
        x2, y2 = bbox_center(b2)
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)


    def extract_any_number(text):
        """
        Extract all numbers from text in range 1–1000
        """
        nums = re.findall(r"\d+", text)
        nums = [int(n) for n in nums if 1 <= int(n) <= 1000]
        return nums


    def extract_hp_loose(ocr_lines):
        candidates = []

        # -------- PASS 1: same-token numbers --------
        for line in ocr_lines:
            text = line["text"]
            norm = normalize(text)

            if "hp" in norm:
                nums = extract_any_number(text)
                if nums:
                    # choose most reasonable (smallest > 10)
                    nums = sorted(nums, key=lambda x: abs(x-50))
                    return nums[0]

        # -------- PASS 2: spatial search --------
        for line in ocr_lines:
            norm = normalize(line["text"])

            if any(k.replace(".", "").replace(" ", "") in norm for k in HP_KEYWORDS):
                hp_box = line["bbox"]

                for other in ocr_lines:
                    nums = extract_any_number(other["text"])
                    for n in nums:
                        d = distance(hp_box, other["bbox"])
                        candidates.append((d, n))

        if candidates:
            candidates.sort(key=lambda x: (x[0], abs(x[1]-50)))
            return candidates[0][1]

        return None
    return extract_hp_loose(ocr_lines)

# calling function
rule_based_hp=hp_extractor()



#---------------------------------------------------
#---------------------QWEN--------------------------
#---------------------------------------------------








from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info



#OFFLINE_PATH="./qwen2_vl_offline"
os.environ["HF_HUB_OFFLINE"] = "1"

def load_qwen():
    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        OFFLINE_PATH, torch_dtype="auto", device_map="auto",local_files_only=True #try changing this  to cuda if sufficient memory available
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-2B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer

    # min_pixels = 256 * 28 * 28     # small images
    # max_pixels = 768 * 28 * 28     # HARD CAP

    # Set lower pixel limits for faster processing
    # 512 tokens is usually plenty for a clear invoice
    min_pixels = 128 * 28 * 28
    max_pixels = 512 * 28 * 28

    processor = AutoProcessor.from_pretrained(OFFLINE_PATH,min_pixels=min_pixels,max_pixels=max_pixels,local_files_only=True)
    return processor,model



def qwen(prompt,img,processor,model):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": prompt,
                    "image": img,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


    


from PIL import Image

def np_to_pil(img_np):
    # OpenCV uses BGR, PIL expects RGB
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)



def clean_value_to_int(text):
    """
    Cleans values like:
    'Rs. 8,01,815.00'
    '7,50,000'
    550000.0
    into -> 801815, 750000, 550000
    """
    if text is None:
        return 0

    # If already numeric, just convert
    if isinstance(text, (int, float)):
        return int(text)

    # If list, take first element
    if isinstance(text, list):
        text = text[0]

    text = str(text)

    # Extract digits
    digits = re.findall(r'\d+', text)
    if not digits:
        return 0

    return int("".join(digits))




def qwen_language(model,processor,body="header"):
    header=np_to_pil(crops[body])
    header_lang_prompt = """Analyze the script used in this header image. 
    If the text is written entirely in English (Latin script), output 1. 
    If there is any Hindi, Marathi, or other Devanagari script present, output 0. 
    Output only the single digit (1 or 0). 
    Do not provide any explanation, labels, or words."""
    output=qwen(header_lang_prompt,header,processor,model)
    return clean_value_to_int(output[0])



def qwen_dealer(model,processor,header="header"):
    header=np_to_pil(crops[header])

    # header_prompt = """Identify the Tractor Dealership/Agency name that issued this invoice. 
    # Ignore the customer name (usually labeled 'To' or 'Buyer').
    # Ignore the tractor manufacturer name (e.g., 'Swaraj' or 'John Deere') unless it is part of the specific dealer name.
    # It is generally the one with largest size in the header.

    # SCRIPT RULE: If the dealer name is in Hindi (Devanagari) or Gujarati script, 
    # you MUST output it in that original script. Do not translate or transliterate it to English.

    # Output the official business name ONLY. No preamble."""

    header_prompt = """Identify the Dealer or Agency name in this image. 
    Focus on the top-center area of the document.
    Find the largest and boldest text, which is typically the business name.
    It may be in English or a regional script (like Hindi, Marathi, or Bengali).
    Output the Dealer Name exactly as it appears in its original script.
    Output the NAME ONLY. No sentences, no preamble, and no translation."""
    # header_prompt = """આ ઇમેજમાંથી ટ્રેક્ટર ડીલર અથવા એજન્સીનું નામ શોધો.

    #     નિયમો:
    #     ૧. માત્ર ગુજરાતી લિપિમાં જ જવાબ આપો (દા.ત. 'શ્રીજી ટ્રેક્ટર્સ').
    #     ૨. અંગ્રેજીમાં અનુવાદ કે લખાણ ન કરો.
    #     ૩. ગ્રાહકનું નામ (ખરીદનાર) અથવા ઉત્પાદક કંપનીનું નામ (Mahindra/Swaraj) ન લખો.
    #     ૪. નામની આગળ 'મે.' કે 'M/s' હોય તો તે પણ ગુજરાતીમાં જ લખો.

    #     ફક્ત ડીલરનું નામ જ લખો. કોઈ વધારાના વાક્યો ન લખો."""


    output= qwen(header_prompt,header,processor,model)
    #print(output[0])
    return output[0]


def qwen_model(model,processor,body="body"):
    body=np_to_pil(crops[body])
    body_prompt = """Find the tractor model identifier in this image (e.g., '575 DI' or 'MU4501'). 
    Exclude the company brand name like Mahindra or Kubota.
    Sometimes the model may have a tick upon it. 
    Output the model text ONLY. No sentences, no preamble."""
    output= qwen(body_prompt,body,processor,model)
    return output[0]

def qwen_cost(model,processor,footer="footer"):
    footer=np_to_pil(crops[footer])
    footer_prompt = """Find the final grand total amount (after all taxes and discounts) in this image.
    Check both numerical values and 'Amount in Words'. 
    Output only the final numerical digits (e.g., 750000). 
    Do not include currency symbols, commas, or words. 
    If multiple totals exist, take the largest and final one."""
    output=qwen(footer_prompt,footer,processor,model)
    return clean_value_to_int(output[0])
    

def qwen_hp(model,processor,body="body"):
    body=np_to_pil(crops[body])
    hp_prompt = """Locate the tractor's engine power or horsepower (HP) rating in this image.
    Look for keywords like 'HP', 'Horse Power', 'BHP', 'Engine Power', or 'HP Category'.
    Output only the final numerical digits (e.g., 45).
    If the value is a decimal, provide it as a float (e.g., 47.5). 
    Value can be only between 25-125.
    Do not include the letters 'HP', units, or any descriptive text.
    If multiple values exist, extract the one labeled as 'HP' or 'Max Power'."""
    output=qwen(hp_prompt,body,processor,model)
    return clean_value_to_int(output[0])



def go_to_qwen(final_dealer,final_model,final_cost,final_hp):
    processor,model=load_qwen()

    if model_to_qwen==1:
        final_model=qwen_model(model,processor)
        #print('model')
    
    if cost_to_qwen==1:
        final_cost=qwen_cost(model,processor)
        #print("cost")
    
    if hp_to_qwen==1:
        final_hp=qwen_hp(model,processor)
        #print("hp")
    
    if dealer_to_qwen==1:
        final_dealer=qwen_dealer(model,processor)
        #print("dealer")
    

    return final_dealer,final_model,final_cost,final_hp



#dealer_to_qwen already sorted above
model_to_qwen=0
cost_to_qwen=1
hp_to_qwen=0

#add some condition for model to forward it to Qwen
if rule_based_model_confidence<0.6:
    model_to_qwen=1


if dealer_normal_way==1 and rule_based_dealer_confidence>0.60:
    dealer_to_qwen=0
    final_dealer=rule_based_dealer


if (int(rule_based_cost)>150000 and int(rule_based_cost)<2500000):
    cost_to_qwen=0
    final_cost=rule_based_cost  


if (rule_based_hp==None) or(rule_based_hp<15 or rule_based_hp>60) :
    hp_to_qwen=1
else:
    final_hp=rule_based_hp

#if all are extracted confidently then give direct output
if (cost_to_qwen + model_to_qwen + dealer_to_qwen + hp_to_qwen)==0:
    #give for output
    final_dealer=rule_based_dealer
    final_cost=rule_based_cost
    #output functionn and stop the rest
else:
    final_dealer,final_model,final_cost,final_hp=go_to_qwen(final_dealer,final_model,final_cost,final_hp)

#---------------------------------------------------
#------------------STAMP AND SIGN-------------------
#---------------------------------------------------


from ultralytics import YOLO


def infer_signature_stamp(model_path, image_path, conf=0.4):
    model = YOLO(model_path)
    results = model(image_path, conf=conf, verbose=False)[0]

    signature_present = False
    signature_bbox = None
    signature_conf = -1.0

    stamp_present = False
    stamp_bbox = None
    stamp_conf = -1.0

    if results.boxes is None:
        return signature_present, signature_bbox, stamp_present, stamp_bbox

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    names = model.names

    for box, cls, score in zip(boxes, classes, confs):
        label = names[int(cls)].lower()

        if label == "signature" and score > signature_conf:
            signature_conf = score
            signature_present = True
            signature_bbox = [int(x) for x in box]

        elif label == "stamp" and score > stamp_conf:
            stamp_conf = score
            stamp_present = True
            stamp_bbox = [int(x) for x in box]

    return signature_present, signature_bbox, stamp_present, stamp_bbox


if __name__ == "__main__":
    #MODEL_PATH = "./best.pt"
    IMAGE_PATH = img_path

    sig_present, sig_bbox, stamp_present, stamp_bbox = infer_signature_stamp(
        MODEL_PATH,
        IMAGE_PATH,
        conf=0.4
    )


#---------------------------------------------------
#---------------------Output------------------------
#---------------------------------------------------


# print(rule_based_dealer_confidence)
# print(rule_based_cost_confidence)
#print(rule_based_model_confidence)

end_time = time.time()
elapsed_time = end_time - start_time



import json

doc_id = doc_id
d_name = final_dealer
m_name = final_model
hp = final_hp
cost = final_cost
sig_present = sig_present
sig_box = sig_bbox
stamp_present = stamp_present
stamp_box = stamp_bbox
conf = rule_based_model_confidence
p_time = elapsed_time
c_est = 0

output_data = {
    "doc_id": doc_id,
    "fields": {
        "dealer_name": d_name,
        "model_name": m_name,
        "horse_power": hp,
        "asset_cost": cost,
        "signature": {
            "present": sig_present,
            "bbox": sig_box
        },
        "stamp": {
            "present": stamp_present,
            "bbox": stamp_box
        }
    },
    "confidence": conf,
    "processing_time_sec": p_time,
    "cost_estimate_usd": c_est
}


json_output = json.dumps(output_data, indent=4)

print()
print()
print()
print(json_output)
print()
print()


from pathlib import Path

# 1. Get the directory of the current script
BASE_DIR = Path(__file__).resolve().parent

# 2. Define the output folder path
output_folder = BASE_DIR / "outputs"

# 3. Create the folder if it doesn't exist (exist_ok=True prevents errors if it's already there)
output_folder.mkdir(exist_ok=True)

# 4. Construct the full file path
file_path = output_folder / f"{doc_id}.json"

# 5. Save the JSON
with open(file_path, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"Result saved to: {file_path}")



