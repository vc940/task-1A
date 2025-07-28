import os
import json
import PyPDF2
import google.generativeai as genai
import shutil
import time
# Set your Gemini API key
GOOGLE_API_KEY = "AIzaSyBrtSWzsmHJkWZZ5VGYCR70JNMyB6fihe0"
genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def get_headings_from_gemini(text):
    prompt = (
        "Extract all headings (H1, H2, H3, H4) and their texts from the following document. "
        "Return the result as a JSON array with objects containing 'level' (H1/H2/H3/H4) and 'text'. "
        "Do not include page numbers. Example output: "
        "[{\"level\": \"H1\", \"text\": \"INTRODUCTION\"}, ...]\n\n"
        "Document:\n"
        f"{text}"
    )
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    print("Sending request to Gemini model...")
    print("Prompt length:", len(prompt))
      # Truncate prompt for debugging
    response = model.generate_content(prompt)
    if len(prompt)> 100000:
        time.sleep(20)  
    # Try to extract JSON from the response
    try:
        json_start = response.text.find('[')
        json_end = response.text.rfind(']') + 1
        headings_json = response.text[json_start:json_end]
        return json.loads(headings_json)
    except Exception as e:
        print("Error parsing Gemini response:", e)
        print("Raw response:", response.text)
        return []

def main(pdf_path, output_json):
    text = extract_text_from_pdf(pdf_path)
    headings = get_headings_from_gemini(text)
    result = {
        "title": os.path.basename(pdf_path),
        "outline": headings
    }
    with open(output_json, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Extracted outline saved to {output_json}")

if __name__ == "__main__":
    
    for pdf_file in os.listdir("downloaded_pdfs"):
        try:
            if not pdf_file.lower().endswith(".pdf"):
                continue
            pdf_path = f"downloaded_pdfs/{pdf_file}"  # Replace with your PDF file path
            pdf_file = pdf_file.split(".pdf")[0]
            output_json = f"solutions/{pdf_file}.json"  # Replace with your desired output JSON file path
            main(pdf_path,output_json)
            shutil.move(pdf_path ,"pdfs")
            time.sleep(2)
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            continue