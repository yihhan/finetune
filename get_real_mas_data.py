"""
Script to help download real MAS documents from official sources

This script provides instructions and tools to download actual MAS documents
for your fine-tuning project.
"""

import requests
import json
from pathlib import Path
import time

def download_mas_document(url: str, filename: str, output_dir: Path = Path("data/mas_official")):
    """Download a document from MAS website"""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename
    
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"✓ Downloaded: {file_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

def get_mas_document_links():
    """Get important MAS document URLs"""
    return {
        "MAS_Notice_626_AML_CFT.html": "https://www.mas.gov.sg/regulation/notices/notice-626",
        "MAS_Notice_637_Capital_Adequacy.html": "https://www.mas.gov.sg/regulation/notices/notice-637", 
        "MAS_Guidelines_AI_Advisory.html": "https://www.mas.gov.sg/publications/guidelines/guidelines-on-automated-investment-management-services",
        "MAS_Technology_Risk_Management.html": "https://www.mas.gov.sg/publications/guidelines/technology-risk-management-guidelines",
        "MAS_Payment_Services_Act.html": "https://www.mas.gov.sg/regulation/acts/payment-services-act",
        "MAS_PDPA_Guidelines.html": "https://www.mas.gov.sg/publications/guidelines/personal-data-protection-act-guidelines",
        "MAS_Cybersecurity_Requirements.html": "https://www.mas.gov.sg/publications/guidelines/cybersecurity-requirements",
    }

def main():
    """Download real MAS documents"""
    print("🚀 Downloading Real MAS Documents...")
    print("Note: This will download actual HTML pages from MAS website")
    
    # Get document links
    documents = get_mas_document_links()
    
    # Create output directory
    output_dir = Path("data/mas_official")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download documents
    success_count = 0
    for filename, url in documents.items():
        if download_mas_document(url, filename, output_dir):
            success_count += 1
        time.sleep(2)  # Be respectful to the server
    
    print(f"\n✅ Download completed!")
    print(f"📁 Files saved to: {output_dir}")
    print(f"📊 Successfully downloaded: {success_count}/{len(documents)} documents")
    
    # Create processing instructions
    instructions = """
# Processing Downloaded MAS Documents

The downloaded HTML files need to be processed to extract Q&A pairs.

## Manual Processing Steps:

1. Open each HTML file in a web browser
2. Copy the relevant text content
3. Create structured Q&A pairs
4. Save as .txt files in the data/ directory

## Automated Processing (Advanced):

You can create a script to parse the HTML and extract:
- Headings as questions
- Content as answers
- Regulatory requirements
- Compliance guidelines

## Example Q&A Format:

Q: What is the purpose of MAS Notice 626?
A: MAS Notice 626 outlines requirements for prevention of money laundering and countering the financing of terrorism.

Q: Who must comply with these requirements?
A: All licensed financial institutions operating in Singapore must comply.

## Next Steps:

1. Process the downloaded HTML files
2. Run: python dataset_prep.py
3. Run: python train.py
4. Run: python eval.py
"""
    
    with open(output_dir / "PROCESSING_INSTRUCTIONS.txt", 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"\n📝 Processing instructions saved to: {output_dir}/PROCESSING_INSTRUCTIONS.txt")

if __name__ == "__main__":
    main()
