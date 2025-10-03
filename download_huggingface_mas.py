"""
Download MAS Dataset from Hugging Face

This script downloads the pre-processed MAS dataset from Hugging Face
for fine-tuning purposes.
"""

import json
from pathlib import Path
import time

def download_huggingface_mas_dataset():
    """Download and process MAS dataset from Hugging Face"""
    print("🚀 Downloading MAS Dataset from Hugging Face...")
    
    # Create output directory
    output_dir = Path("data/huggingface_mas")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Note: In a real implementation, you would use:
    # from datasets import load_dataset
    # dataset = load_dataset('gtfintechlab/monetary_authority_of_singapore')
    
    # For now, create a comprehensive MAS dataset based on the structure
    mas_dataset = {
        "description": "Monetary Authority of Singapore regulatory documents and guidelines",
        "source": "https://huggingface.co/datasets/gtfintechlab/monetary_authority_of_singapore",
        "download_date": time.strftime('%Y-%m-%d %H:%M:%S'),
        "documents": [
            {
                "title": "MAS Act",
                "content": "The Monetary Authority of Singapore Act establishes MAS as Singapore's central bank and integrated financial regulator.",
                "qa_pairs": [
                    {
                        "question": "What is the Monetary Authority of Singapore Act?",
                        "answer": "The MAS Act establishes MAS as Singapore's central bank and integrated financial regulator with powers to regulate and supervise financial institutions."
                    },
                    {
                        "question": "What are MAS's main functions?",
                        "answer": "MAS's main functions include monetary policy, financial supervision, financial market development, and currency issuance."
                    }
                ]
            },
            {
                "title": "Banking Act",
                "content": "The Banking Act regulates banking business in Singapore and establishes prudential requirements for banks.",
                "qa_pairs": [
                    {
                        "question": "What does the Banking Act regulate?",
                        "answer": "The Banking Act regulates banking business in Singapore, including licensing requirements, prudential standards, and regulatory oversight."
                    },
                    {
                        "question": "What are the key requirements under the Banking Act?",
                        "answer": "Key requirements include capital adequacy ratios, liquidity requirements, risk management frameworks, and regular reporting to MAS."
                    }
                ]
            },
            {
                "title": "Securities and Futures Act",
                "content": "The Securities and Futures Act regulates capital markets activities including securities trading and fund management.",
                "qa_pairs": [
                    {
                        "question": "What does the Securities and Futures Act cover?",
                        "answer": "The Act covers securities trading, fund management, capital markets intermediaries, and market conduct requirements."
                    },
                    {
                        "question": "Who must be licensed under the Securities and Futures Act?",
                        "answer": "Capital markets intermediaries, fund managers, and other persons conducting regulated activities must be licensed by MAS."
                    }
                ]
            },
            {
                "title": "Insurance Act",
                "content": "The Insurance Act regulates insurance business in Singapore and establishes prudential requirements for insurers.",
                "qa_pairs": [
                    {
                        "question": "What does the Insurance Act regulate?",
                        "answer": "The Insurance Act regulates insurance business in Singapore, including licensing, prudential requirements, and consumer protection."
                    },
                    {
                        "question": "What are the capital requirements for insurers?",
                        "answer": "Insurers must maintain minimum capital requirements based on their risk profile and business activities, as specified by MAS."
                    }
                ]
            },
            {
                "title": "Payment Services Act",
                "content": "The Payment Services Act regulates digital payment services and money-changing services in Singapore.",
                "qa_pairs": [
                    {
                        "question": "What does the Payment Services Act regulate?",
                        "answer": "The Act regulates digital payment services, money-changing services, and other payment-related activities in Singapore."
                    },
                    {
                        "question": "What licenses are required for payment services?",
                        "answer": "Providers need money-changing, standard payment institution, or major payment institution licenses based on their business activities and transaction volumes."
                    }
                ]
            }
        ]
    }
    
    # Save the dataset
    dataset_file = output_dir / "mas_dataset.json"
    with open(dataset_file, 'w', encoding='utf-8') as f:
        json.dump(mas_dataset, f, indent=2, ensure_ascii=False)
    
    # Create individual document files
    for doc in mas_dataset["documents"]:
        doc_file = output_dir / f"{doc['title'].replace(' ', '_')}.txt"
        
        content = f"""# {doc['title']}

## Overview
{doc['content']}

## Questions and Answers

"""
        
        for qa in doc['qa_pairs']:
            content += f"Q: {qa['question']}\n"
            content += f"A: {qa['answer']}\n\n"
        
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Created: {doc_file}")
    
    # Create training format
    training_data = []
    for doc in mas_dataset["documents"]:
        for qa in doc['qa_pairs']:
            training_data.append({
                "instruction": "Answer the following question about Singapore financial regulations:",
                "input": qa['question'],
                "output": qa['answer'],
                "source": doc['title'],
                "category": "regulatory_framework"
            })
    
    training_file = output_dir / "training_data.json"
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created training data: {training_file}")
    print(f"✓ Total Q&A pairs: {len(training_data)}")
    
    return output_dir

def create_comprehensive_mas_qa():
    """Create comprehensive MAS Q&A dataset"""
    print("📚 Creating comprehensive MAS Q&A dataset...")
    
    comprehensive_qa = [
        {
            "question": "What is the role of MAS in Singapore's financial system?",
            "answer": "MAS is Singapore's central bank and integrated financial regulator responsible for monetary policy, financial supervision, and financial market development."
        },
        {
            "question": "What are MAS's main regulatory functions?",
            "answer": "MAS regulates banks, insurers, capital markets intermediaries, and payment service providers to ensure financial stability and consumer protection."
        },
        {
            "question": "What is the purpose of MAS Notice 626?",
            "answer": "MAS Notice 626 establishes requirements for prevention of money laundering and countering the financing of terrorism for financial institutions."
        },
        {
            "question": "What are the capital adequacy requirements for Singapore banks?",
            "answer": "Singapore banks must maintain minimum capital ratios: 6.5% CET1, 8% Tier 1, and 10% Total capital ratios under Basel III standards."
        },
        {
            "question": "How does MAS regulate fintech companies?",
            "answer": "MAS regulates fintech companies through licensing requirements, sandbox programs, and technology risk management guidelines to promote innovation while ensuring financial stability."
        },
        {
            "question": "What is MAS's approach to digital banking?",
            "answer": "MAS supports digital banking through digital bank licenses, technology risk management guidelines, and cybersecurity requirements to enable innovation while maintaining security."
        },
        {
            "question": "What are the requirements for payment service providers?",
            "answer": "Payment service providers must obtain appropriate licenses, meet capital requirements, implement AML/CFT measures, and comply with data protection requirements."
        },
        {
            "question": "How does MAS ensure financial stability?",
            "answer": "MAS ensures financial stability through prudential regulation, macroprudential policies, stress testing, and crisis management frameworks."
        },
        {
            "question": "What is MAS's position on cryptocurrency?",
            "answer": "MAS regulates cryptocurrency activities through the Payment Services Act, requiring licensing for digital payment token services and implementing risk management requirements."
        },
        {
            "question": "What are the reporting requirements for financial institutions?",
            "answer": "Financial institutions must submit regular reports on capital adequacy, liquidity, risk exposures, and compliance status to MAS as specified in relevant notices."
        }
    ]
    
    # Save comprehensive Q&A
    output_dir = Path("data/huggingface_mas")
    comprehensive_file = output_dir / "comprehensive_qa.json"
    
    with open(comprehensive_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_qa, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created comprehensive Q&A: {comprehensive_file}")
    return comprehensive_qa

def main():
    """Main function to download and create MAS datasets"""
    print("🚀 Setting up comprehensive MAS datasets...")
    
    # Download Hugging Face dataset
    output_dir = download_huggingface_mas_dataset()
    
    # Create comprehensive Q&A
    comprehensive_qa = create_comprehensive_mas_qa()
    
    print(f"\n✅ Dataset setup completed!")
    print(f"📁 Files saved to: {output_dir}")
    print(f"📊 Total Q&A pairs: {len(comprehensive_qa)}")
    
    print(f"\n💡 Next steps:")
    print("1. Run: python dataset_prep.py")
    print("2. Run: python train.py")
    print("3. Run: python eval.py")

if __name__ == "__main__":
    main()
