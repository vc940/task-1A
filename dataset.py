import fitz  # PyMuPDF
import pandas as pd
import json
import os
from uuid import uuid4
from matplotlib import pyplot as plt
def extract_blocks_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_blocks = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")  # returns (x0, y0, x1, y1, "text", block_no, ...)
        for block in blocks:
            x0, y0, x1, y1, text = block[:5]
            if text.strip():  # skip empty
                all_blocks.append({
                    "text": text.strip(),
                    "bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "page": page_num + 1
                })
    return all_blocks
result = []
for i in os.listdir("pdfs"):
    PATH = i  
    path2 = PATH.split(".pdf")[0]
    with open(f"solutions/{path2}.json",'r', encoding='utf-8') as file:
        data = json.load(file)
    PATH = f"pdfs/{i}"
    JSONS = extract_blocks_from_pdf(PATH)
    # Loop through each text block in your annotations
    id = str(uuid4())
    for i in range(len(JSONS)):
        text = JSONS[i]["text"].strip()  # Strip leading/trailing spaces
        
        # Go through each outline heading from the TOC
        for j in data["outline"]:
            # Match: if text starts with the outline heading (case-insensitive)
            if text.lower().startswith(j["text"].strip().lower()):
                JSONS[i]["level"] = j["level"]  # Assign heading level (e.g., H1, H2)
                JSONS[i]["id"] = id
                break
        else:
            JSONS[i]["level"] = "para"
            JSONS[i]["id"] = id  # Default label if no match found
    result.extend(JSONS)
    # dataset =(pd.DataFrame(JSONS))
    # dataset["id"] = str(uuid4())
    # print(dataset.columns)
    # print(dataset.level.value_counts())
    # data = pd.read_csv("/home/vc940/Work/ADOBE PDF ANNOTATION/dataset.csv")
    # dataset = pd.concat([data,dataset])
dataset = pd.DataFrame(result)
dataset.to_csv("/home/vc940/Work/ADOBE PDF ANNOTATION/dataset.csv", index=False, escapechar='\\')
print("Dataset created successfully with", len(dataset), "entries.")
