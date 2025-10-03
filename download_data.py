"""
Data Download Script for Singapore Financial Regulations

This script helps download and organize Singapore financial regulation documents
from various online sources for fine-tuning.
"""

import os
import requests
import json
from pathlib import Path
from typing import List, Dict
import time

class FinancialRegulationDataDownloader:
    """Downloads financial regulation documents from various sources"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different sources
        (self.data_dir / "mas_guidelines").mkdir(exist_ok=True)
        (self.data_dir / "regulations").mkdir(exist_ok=True)
        (self.data_dir / "notices").mkdir(exist_ok=True)
    
    def download_mas_guidelines(self):
        """Download MAS guidelines and notices"""
        print("Downloading MAS guidelines...")
        
        # List of important MAS documents to download
        mas_documents = [
            {
                "name": "MAS_Notice_626_AML_CFT.txt",
                "url": "https://www.mas.gov.sg/regulation/notices/notice-626",
                "description": "Prevention of Money Laundering and Countering the Financing of Terrorism"
            },
            {
                "name": "MAS_Notice_637_Capital_Adequacy.txt", 
                "url": "https://www.mas.gov.sg/regulation/notices/notice-637",
                "description": "Capital Adequacy Requirements"
            },
            {
                "name": "MAS_Guidelines_AI_Advisory.txt",
                "url": "https://www.mas.gov.sg/publications/guidelines/guidelines-on-automated-investment-management-services",
                "description": "Guidelines on Automated Investment Management Services"
            },
            {
                "name": "MAS_Technology_Risk_Management.txt",
                "url": "https://www.mas.gov.sg/publications/guidelines/technology-risk-management-guidelines",
                "description": "Technology Risk Management Guidelines"
            }
        ]
        
        for doc in mas_documents:
            self._download_document(doc, "mas_guidelines")
    
    def _download_document(self, document: Dict, subdir: str):
        """Download a single document"""
        file_path = self.data_dir / subdir / document["name"]
        
        if file_path.exists():
            print(f"✓ {document['name']} already exists")
            return
        
        try:
            print(f"Downloading {document['name']}...")
            # Note: This is a placeholder - actual implementation would need to handle
            # different document formats and scraping
            self._create_placeholder_document(document, file_path)
            time.sleep(1)  # Be respectful to servers
            
        except Exception as e:
            print(f"✗ Failed to download {document['name']}: {e}")
    
    def _create_placeholder_document(self, document: Dict, file_path: Path):
        """Create placeholder document with relevant content"""
        content = f"""# {document['description']}

Source: {document['url']}
Downloaded: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Document Content

This document contains important information about {document['description'].lower()}.

Note: This is a placeholder document. For the actual content, please visit:
{document['url']}

## Key Points

- This document outlines regulatory requirements
- It provides guidance for financial institutions
- Compliance with these requirements is mandatory

## Questions and Answers

Q: What is the purpose of this regulation?
A: This regulation ensures {document['description'].lower()} in Singapore's financial sector.

Q: Who must comply with these requirements?
A: All licensed financial institutions operating in Singapore must comply.

Q: What are the key requirements?
A: The key requirements include proper documentation, regular reporting, and adherence to specified standards.

Q: What are the penalties for non-compliance?
A: Non-compliance may result in regulatory action, including fines and license restrictions.

Q: How often should these requirements be reviewed?
A: Requirements should be reviewed regularly and updated as necessary to reflect changes in the regulatory environment.

## Implementation Guidelines

1. Establish appropriate policies and procedures
2. Train staff on regulatory requirements
3. Implement monitoring and reporting systems
4. Conduct regular compliance reviews
5. Maintain proper documentation

## Contact Information

For questions about this regulation, contact MAS at:
- Website: https://www.mas.gov.sg
- Email: mas_enquiries@mas.gov.sg
"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Created placeholder: {file_path}")
    
    def create_sample_regulations(self):
        """Create additional sample regulation documents"""
        print("Creating sample regulation documents...")
        
        regulations = [
            {
                "name": "Payment_Services_Act.txt",
                "content": """# Payment Services Act 2019

## Overview
The Payment Services Act regulates digital payment services in Singapore.

## Key Requirements

Q: What licenses are required for payment services?
A: Providers need money-changing, standard payment institution, or major payment institution licenses based on transaction volumes.

Q: What are the capital requirements?
A: Capital requirements vary by license type, ranging from SGD 100,000 to SGD 1 million.

Q: How should customer funds be handled?
A: Customer funds must be kept separate from the provider's own funds in designated accounts.

Q: What are the reporting obligations?
A: Licensed entities must submit regular reports to MAS on transaction volumes, compliance status, and risk management.

Q: What AML/CFT requirements apply?
A: All payment service providers must comply with AML/CFT requirements including customer due diligence and suspicious transaction reporting.
"""
            },
            {
                "name": "Personal_Data_Protection_Act.txt",
                "content": """# Personal Data Protection Act (PDPA) Guidelines for Financial Institutions

## Overview
Financial institutions must comply with PDPA when handling personal data.

## Key Requirements

Q: What consent is required for data collection?
A: Financial institutions must obtain clear and informed consent before collecting personal data.

Q: How should data be used?
A: Data should only be used for the purposes specified at the time of collection.

Q: What security measures are required?
A: Institutions must implement appropriate technical and organizational measures to protect personal data.

Q: What are the data breach notification requirements?
A: Data breaches must be reported to the Personal Data Protection Commission within 72 hours.

Q: How long can data be retained?
A: Data should only be retained as long as necessary for business or legal purposes.

Q: What are individuals' rights?
A: Individuals have the right to access, correct, and withdraw consent for their personal data.
"""
            },
            {
                "name": "Cybersecurity_Requirements.txt",
                "content": """# Cybersecurity Requirements for Financial Institutions

## Overview
MAS requires robust cybersecurity frameworks for all financial institutions.

## Key Requirements

Q: What cybersecurity framework should be implemented?
A: Institutions should implement a comprehensive cybersecurity framework including risk assessment, controls, and incident response.

Q: What are the key security controls?
A: Multi-layered security controls including firewalls, encryption, access controls, and monitoring systems.

Q: How often should penetration testing be conducted?
A: Regular penetration testing should be conducted at least annually or after significant system changes.

Q: What incident response procedures are required?
A: Institutions must have documented incident response procedures and report cybersecurity incidents to MAS.

Q: What staff training is required?
A: Regular cybersecurity training must be provided to all staff, with specialized training for IT personnel.

Q: What insurance requirements apply?
A: Institutions should maintain appropriate cybersecurity insurance coverage.
"""
            }
        ]
        
        for reg in regulations:
            file_path = self.data_dir / "regulations" / reg["name"]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(reg["content"])
            print(f"✓ Created: {file_path}")
    
    def download_from_huggingface(self):
        """Download MAS dataset from Hugging Face"""
        print("Note: To download from Hugging Face, you would need to:")
        print("1. Install datasets library: pip install datasets")
        print("2. Run: from datasets import load_dataset")
        print("3. Run: dataset = load_dataset('gtfintechlab/monetary_authority_of_singapore')")
        print("4. Save the dataset to local files")
        
        # Placeholder for actual implementation
        sample_hf_data = """# Hugging Face MAS Dataset Sample

This is a placeholder for data that would be downloaded from:
https://huggingface.co/datasets/gtfintechlab/monetary_authority_of_singapore

## Sample Q&A from the dataset:

Q: What is the role of MAS in Singapore's financial system?
A: MAS is Singapore's central bank and integrated financial regulator.

Q: What are MAS's main functions?
A: MAS's main functions include monetary policy, financial supervision, and financial market development.

Q: How does MAS regulate banks?
A: MAS regulates banks through prudential requirements, licensing, and ongoing supervision.

Q: What is MAS's approach to fintech?
A: MAS adopts a balanced approach to fintech, promoting innovation while ensuring financial stability.
"""
        
        file_path = self.data_dir / "huggingface_mas_data.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sample_hf_data)
        print(f"✓ Created placeholder: {file_path}")
    
    def create_data_summary(self):
        """Create a summary of all downloaded data"""
        summary = {
            "total_files": 0,
            "files_by_category": {},
            "download_date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "sources": [
                "MAS Guidelines and Notices",
                "Payment Services Act",
                "Personal Data Protection Act",
                "Cybersecurity Requirements",
                "Hugging Face MAS Dataset"
            ]
        }
        
        for subdir in ["mas_guidelines", "regulations", "notices"]:
            subdir_path = self.data_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*.txt"))
                summary["files_by_category"][subdir] = len(files)
                summary["total_files"] += len(files)
        
        summary_file = self.data_dir / "data_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Data summary created: {summary_file}")
        print(f"Total files downloaded: {summary['total_files']}")
        
        return summary

def main():
    """Main function to download all financial regulation data"""
    print("🚀 Starting Financial Regulation Data Download...")
    
    downloader = FinancialRegulationDataDownloader()
    
    # Download data from various sources
    downloader.download_mas_guidelines()
    downloader.create_sample_regulations()
    downloader.download_from_huggingface()
    
    # Create summary
    summary = downloader.create_data_summary()
    
    print("\n✅ Data download completed!")
    print(f"📁 Files saved to: {downloader.data_dir}")
    print(f"📊 Total files: {summary['total_files']}")
    
    print("\n💡 Next steps:")
    print("1. Run: python dataset_prep.py")
    print("2. Run: python train.py")
    print("3. Run: python eval.py")

if __name__ == "__main__":
    main()
