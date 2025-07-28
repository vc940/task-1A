
import fitz  # PyMuPDF
import pandas as pd
import json
import os
from uuid import uuid4
import matplotlib.pyplot as plt

# --- Configuration ---
PDF_DIR = "pdfs"
SOLUTION_DIR = "solutions"
OUTPUT_CSV = "dataset.csv"

def extract_text_blocks(pdf_path: str) -> list[dict]:
    """
    Extracts all non-empty text blocks from a given PDF file.

    Args:
        pdf_path: The file path to the PDF.

    Returns:
        A list of dictionaries, where each dictionary represents a
        text block containing its text, bounding box, and page number.
        Returns an empty list if the PDF cannot be opened.
    """
    all_blocks = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error: Could not open or process PDF '{pdf_path}'. Reason: {e}")
        return all_blocks

    for page_num, page in enumerate(doc):
        # Extract blocks with detailed information
        blocks = page.get_text("blocks")  # Format: (x0, y0, x1, y1, "text", block_no, block_type)
        for block in blocks:
            x0, y0, x1, y1, text = block[:5]
            clean_text = text.strip()
            if clean_text:  # Only process blocks with actual text
                all_blocks.append({
                    "text": clean_text,
                    "bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "page": page_num + 1,
                })
    doc.close()
    return all_blocks

def classify_blocks(blocks: list[dict], solution_data: dict) -> pd.DataFrame:
    """
    Classifies text blocks as headings or paragraphs based on solution data.

    Args:
        blocks: A list of text block dictionaries from `extract_text_blocks`.
        solution_data: A dictionary loaded from a JSON file, containing an 'outline'.

    Returns:
        A pandas DataFrame with the classified blocks, including a 'level' column.
    """
    if "outline" not in solution_data:
        print("Warning: 'outline' key not found in solution data. All blocks will be 'para'.")
        solution_headings = []
    else:
        solution_headings = solution_data["outline"]

    for block in blocks:
        block_text_lower = block["text"].lower()
        # Assume 'para' by default
        block["level"] = "para"
        for heading in solution_headings:
            heading_text = heading.get("text", "").strip().lower()
            if heading_text and block_text_lower.startswith(heading_text):
                block["level"] = heading.get("level", "unknown_heading")
                break  # Stop after finding the first match
    
    return pd.DataFrame(blocks)


def main():
    """
    Main function to process all PDFs, classify their content,
    and append the results to a CSV file.
    """
    # --- 1. Process all PDFs and collect data in memory ---
    all_new_data = []
    processed_files = os.listdir(PDF_DIR)
    
    if not processed_files:
        print(f"No PDF files found in the '{PDF_DIR}' directory. Exiting.")
        return

    for pdf_filename in processed_files:
        if not pdf_filename.lower().endswith(".pdf"):
            continue

        base_name = os.path.splitext(pdf_filename)[0]
        pdf_path = os.path.join(PDF_DIR, pdf_filename)
        json_path = os.path.join(SOLUTION_DIR, f"{base_name}.json")

        print(f"Processing: {pdf_path}")

        # Check for corresponding solution file
        if not os.path.exists(json_path):
            print(f"  - Warning: Skipping. No solution file found at '{json_path}'")
            continue

        # Load solution data
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                solution = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  - Warning: Skipping. Could not read or parse JSON '{json_path}'. Reason: {e}")
            continue

        # Extract and classify blocks
        blocks = extract_text_blocks(pdf_path)
        if not blocks:
            print(f"  - Warning: Skipping. No text blocks extracted from '{pdf_path}'")
            continue
            
        pdf_df = classify_blocks(blocks, solution)
        
        # Add a unique ID for this specific PDF
        pdf_df["pdf_id"] = str(uuid4())
        all_new_data.append(pdf_df)

    if not all_new_data:
        print("No new data was generated. Check warnings for details.")
        return

    # --- 2. Combine all new data ---
    new_dataset = pd.concat(all_new_data, ignore_index=True)
    print("\n--- Classification Summary (New Data) ---")
    print(new_dataset['level'].value_counts())
    print("-----------------------------------------\n")

    # --- 3. Update master CSV file ---
    try:
        # Read existing data if the file exists
        if os.path.exists(OUTPUT_CSV):
            print(f"Appending to existing dataset: '{OUTPUT_CSV}'")
            existing_data = pd.read_csv(OUTPUT_CSV)
            combined_dataset = pd.concat([existing_data, new_dataset], ignore_index=True)
        else:
            print(f"Creating new dataset: '{OUTPUT_CSV}'")
            combined_dataset = new_dataset
        
        # Save the final dataset
        combined_dataset.to_csv(OUTPUT_CSV, index=False)
        print("Dataset saved successfully.")

    except Exception as e:
        print(f"Error: Failed to write to CSV '{OUTPUT_CSV}'. Reason: {e}")
        return

    # --- 4. Perform analysis on the last processed PDF ---
    # This section demonstrates analysis on one of the newly added files.
    if all_new_data:
        last_pdf_df = all_new_data[-1]
        last_pdf_id = last_pdf_df["pdf_id"].iloc[0]
        
        print(f"\n--- Analysis of last processed PDF (ID: {last_pdf_id}) ---")
        last_pdf_df["text_length"] = last_pdf_df["text"].apply(len)
        
        plt.figure(figsize=(10, 6))
        last_pdf_df["text_length"].hist(bins=50)
        plt.title(f"Histogram of Text Block Length for '{processed_files[-1]}'")
        plt.xlabel("Number of Characters per Text Block")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75)
        plt.show()


if __name__ == "__main__":
    main()
