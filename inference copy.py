import fitz  # PyMuPDF
import pandas as pd
import json
import os
from uuid import uuid4


def extract_blocks_from_pdf(pdf_path):
    """Extract text blocks from a PDF file."""
    doc = fitz.open(pdf_path)
    all_blocks = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")  # returns (x0, y0, x1, y1, "text", block_no, ...)
        for block in blocks:
            x0, y0, x1, y1, text = block[:5]
            if text.strip():
                all_blocks.append({
                    "text": text.strip(),
                    "bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "page": page_num + 1
                })
    return all_blocks


def annotate_blocks(blocks, outline):
    """Assign heading levels to blocks based on outline."""
    annotated = []
    doc_id = str(uuid4())
    
    for block in blocks:
        text = block["text"]
        for item in outline:
            if text.lower().startswith(item["text"].strip().lower()):
                block["id"] = doc_id
                break
        else:
            block["id"] = doc_id
        annotated.append(block)
    return annotated


def process_folder(pdf_folder, outline_folder, output_path):
    """Process a folder of PDFs and their corresponding outlines."""
    result = []
    
    for filename in os.listdir(pdf_folder):
        if not filename.endswith(".pdf"):
            continue
        
        base_name = filename.rsplit(".pdf", 1)[0]
        outline_path = os.path.join(outline_folder, f"{base_name}.json")
        pdf_path = os.path.join(pdf_folder, filename)

        if not os.path.exists(outline_path):
            print(f"Warning: Outline for {filename} not found. Skipping.")
            continue
        
        with open(outline_path, 'r', encoding='utf-8') as f:
            outline = json.load(f)["outline"]

        blocks = extract_blocks_from_pdf(pdf_path)
        annotated_blocks = annotate_blocks(blocks, outline)
        result.extend(annotated_blocks)
    
    df = pd.DataFrame(result)
    df.to_csv(output_path, index=False, escapechar='\\')
    print(f"Inference complete. Saved {len(df)} entries to {output_path}.")


if __name__ == "__main__":
    PDF_FOLDER = "/home/vc940/Work/ADOBE PDF ANNOTATION/pdfs/a"
    OUTLINE_FOLDER = "solutions"
    OUTPUT_CSV = "/home/vc940/Work/ADOBE PDF ANNOTATION/datasett.csv"

    process_folder(PDF_FOLDER, OUTLINE_FOLDER, OUTPUT_CSV)
