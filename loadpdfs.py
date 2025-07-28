import os
import requests
from tqdm import tqdm

def download_pdfs(url_list, save_folder):
    """
    Downloads PDF files from the list of URLs into the specified folder.

    Args:
        url_list (list of str): List of PDF URLs to download.
        save_folder (str): Folder path where PDFs will be saved.

    Returns:
        None
    """
    os.makedirs(save_folder, exist_ok=True)

    for url in tqdm(url_list, desc="Downloading PDFs"):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Extract filename from URL or generate a unique name
            filename = os.path.basename(url)
            if not filename.lower().endswith(".pdf"):
                filename += ".pdf"
            save_path = os.path.join(save_folder, filename)

            # If filename already exists, add suffix to avoid overwrite
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(save_path):
                filename = f"{base}_{counter}{ext}"
                save_path = os.path.join(save_folder, filename)
                counter += 1

            # Write content to file in chunks
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    # Example list of PDF URLs - replace with your own URLs
    pdf_urls = [f"https://arxiv.org/pdf/2106.076{str(i).zfill(2)}" for i in range(1, 100)]
    print(pdf_urls)
    save_folder = "downloaded_pdfs"

    download_pdfs(pdf_urls, save_folder)
    print(f"All done! PDFs saved to folder: {save_folder}")
