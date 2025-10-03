"""
Dataset Preparation Script for Singapore Financial Regulation Q&A

This script converts financial regulation documents into Q&A pairs suitable for fine-tuning.
It processes various document formats and generates training data in the required format.
"""

import json
import os
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

@dataclass
class QAPair:
    """Data class for Question-Answer pairs"""
    question: str
    answer: str
    source: str
    category: str

class FinancialRegulationDatasetPrep:
    """Main class for preparing financial regulation Q&A dataset"""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "processed_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Common financial regulation categories
        self.categories = {
            "capital_requirements": "Capital Adequacy Requirements",
            "risk_management": "Risk Management",
            "compliance": "Compliance and Reporting",
            "ai_advisory": "AI in Financial Advisory",
            "data_protection": "Data Protection and Privacy",
            "anti_money_laundering": "Anti-Money Laundering",
            "cybersecurity": "Cybersecurity",
            "digital_banking": "Digital Banking Services"
        }
    
    def extract_qa_from_text(self, text: str, source: str, category: str) -> List[QAPair]:
        """Extract Q&A pairs from structured text"""
        qa_pairs = []
        
        # Pattern to identify potential Q&A sections
        qa_patterns = [
            r"Q:\s*(.+?)\nA:\s*(.+?)(?=\nQ:|$)",
            r"Question:\s*(.+?)\nAnswer:\s*(.+?)(?=\nQuestion:|$)",
            r"What\s+is\s+(.+?)\?\s*(.+?)(?=\n[A-Z]|$)",
            r"How\s+does\s+(.+?)\s+work\?\s*(.+?)(?=\n[A-Z]|$)",
        ]
        
        for pattern in qa_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    question, answer = match
                    question = question.strip()
                    answer = answer.strip()
                    
                    if len(question) > 10 and len(answer) > 20:
                        qa_pairs.append(QAPair(
                            question=question,
                            answer=answer,
                            source=source,
                            category=category
                        ))
        
        return qa_pairs
    
    def generate_questions_from_content(self, content: str, source: str, category: str) -> List[QAPair]:
        """Generate questions from regulatory content using templates"""
        qa_pairs = []
        
        # Question templates for different types of content
        templates = [
            ("What is the {entity} requirement for {topic}?", "definition"),
            ("How does {entity} regulate {topic}?", "regulation"),
            ("What are the key requirements for {topic} under {entity} guidelines?", "requirements"),
            ("What are the penalties for non-compliance with {topic}?", "penalties"),
            ("How should financial institutions implement {topic}?", "implementation"),
            ("What is {entity}'s position on {topic}?", "position"),
            ("What are the reporting requirements for {topic}?", "reporting"),
            ("How does {entity} define {topic}?", "definition"),
        ]
        
        # Extract key terms and entities
        key_terms = self._extract_key_terms(content)
        
        for template, q_type in templates:
            for term in key_terms[:5]:  # Limit to avoid too many questions
                question = template.format(entity="MAS", topic=term)
                qa_pairs.append(QAPair(
                    question=question,
                    answer=content[:500] + "...",  # Truncated answer
                    source=source,
                    category=category
                ))
        
        return qa_pairs
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key financial terms from text"""
        # Simple keyword extraction (can be enhanced with NLP libraries)
        financial_terms = [
            "capital adequacy", "risk management", "compliance", "reporting",
            "anti-money laundering", "cybersecurity", "data protection",
            "digital banking", "fintech", "cryptocurrency", "AI advisory",
            "robo-advisory", "digital assets", "payment services",
            "remittance", "money laundering", "terrorist financing",
            "customer due diligence", "KYC", "AML", "CTF"
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in financial_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    def create_sample_dataset(self) -> List[QAPair]:
        """Create a sample dataset with Singapore financial regulation Q&A pairs"""
        sample_data = [
            {
                "question": "What is MAS's position on the use of artificial intelligence in financial advisory services?",
                "answer": "MAS supports the responsible use of AI in financial advisory services while ensuring adequate safeguards. Financial institutions must ensure that AI systems used in advisory services are fair, transparent, and accountable. They should have robust governance frameworks, regular model validation, and human oversight mechanisms. MAS expects institutions to clearly disclose the use of AI to customers and ensure that AI-driven recommendations are explainable and aligned with customers' best interests.",
                "source": "MAS Guidelines on AI in Financial Advisory",
                "category": "ai_advisory"
            },
            {
                "question": "What are the capital adequacy requirements for banks in Singapore?",
                "answer": "Singapore banks are required to maintain a minimum Common Equity Tier 1 (CET1) capital ratio of 6.5%, Tier 1 capital ratio of 8%, and Total capital ratio of 10%. These requirements are based on Basel III standards. MAS also requires banks to maintain a capital conservation buffer of 2.5% and a countercyclical capital buffer that can range from 0% to 2.5% depending on economic conditions. Banks must also meet leverage ratio requirements and undergo regular stress testing.",
                "source": "MAS Notice 637 - Capital Adequacy Requirements",
                "category": "capital_requirements"
            },
            {
                "question": "How should financial institutions implement anti-money laundering measures?",
                "answer": "Financial institutions must implement comprehensive AML measures including customer due diligence (CDD), ongoing monitoring, and suspicious transaction reporting. They should conduct enhanced due diligence for high-risk customers and politically exposed persons (PEPs). Institutions must maintain transaction records for at least 5 years and implement risk-based AML programs that are regularly reviewed and updated. They should also provide regular AML training to staff and appoint a designated AML officer.",
                "source": "MAS Notice 626 - Prevention of Money Laundering and Countering the Financing of Terrorism",
                "category": "anti_money_laundering"
            },
            {
                "question": "What are the data protection requirements for financial institutions under the PDPA?",
                "answer": "Financial institutions must comply with the Personal Data Protection Act (PDPA) which requires obtaining consent before collecting personal data, using data only for specified purposes, and implementing appropriate security measures. They must allow individuals to access and correct their personal data, and provide withdrawal of consent mechanisms. Institutions must also implement data breach notification procedures and appoint a Data Protection Officer. Data should be retained only as long as necessary for business or legal purposes.",
                "source": "PDPA Guidelines for Financial Institutions",
                "category": "data_protection"
            },
            {
                "question": "What cybersecurity requirements must financial institutions meet?",
                "answer": "Financial institutions must implement robust cybersecurity frameworks including regular risk assessments, multi-layered security controls, and incident response procedures. They should conduct regular penetration testing, maintain up-to-date security patches, and implement access controls with multi-factor authentication. Institutions must have cyber resilience measures, regular staff training, and maintain cybersecurity insurance. They should also participate in information sharing initiatives and report cybersecurity incidents to MAS within prescribed timeframes.",
                "source": "MAS Technology Risk Management Guidelines",
                "category": "cybersecurity"
            },
            {
                "question": "How does MAS regulate digital payment services?",
                "answer": "MAS regulates digital payment services under the Payment Services Act (PSA). Providers must obtain appropriate licenses (money-changing, standard payment institution, or major payment institution licenses) based on their business activities and transaction volumes. Licensed entities must meet capital requirements, maintain customer funds separately, and implement robust risk management systems. They must also comply with AML/CFT requirements, data protection laws, and reporting obligations to MAS.",
                "source": "Payment Services Act 2019",
                "category": "digital_banking"
            },
            {
                "question": "What are the key requirements for robo-advisory services in Singapore?",
                "answer": "Robo-advisory services must be provided by licensed financial advisers and comply with MAS guidelines on automated investment management services. Providers must ensure adequate risk profiling of clients, maintain human oversight capabilities, and provide clear disclosures about algorithm limitations. They should implement safeguards against algorithm bias, maintain audit trails, and ensure clients understand the automated nature of services. Providers must also have contingency plans for system failures and maintain appropriate professional indemnity insurance.",
                "source": "MAS Guidelines on Automated Investment Management Services",
                "category": "ai_advisory"
            },
            {
                "question": "What compliance reporting requirements do banks have under MAS regulations?",
                "answer": "Banks must submit regular regulatory returns including monthly balance sheet returns, quarterly profit and loss statements, and annual audited financial statements. They must report on capital adequacy ratios, liquidity positions, and large exposures. Banks also need to submit suspicious transaction reports, AML/CFT compliance reports, and technology risk management reports. MAS requires immediate notification of material events, operational incidents, and changes in key personnel. All reports must be submitted through MAS' electronic submission system within prescribed deadlines.",
                "source": "MAS Notice 1015 - Reporting Requirements for Banks",
                "category": "compliance"
            }
        ]
        
        return [QAPair(**item) for item in sample_data]
    
    def save_dataset(self, qa_pairs: List[QAPair], filename: str = "financial_regulation_qa.json"):
        """Save Q&A pairs to JSON file"""
        output_file = self.output_dir / filename
        
        data = []
        for qa in qa_pairs:
            data.append({
                "question": qa.question,
                "answer": qa.answer,
                "source": qa.source,
                "category": qa.category
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {output_file}")
        print(f"Total Q&A pairs: {len(data)}")
        
        # Also save as CSV for easy viewing
        csv_file = self.output_dir / filename.replace('.json', '.csv')
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        print(f"CSV version saved to {csv_file}")
        
        return output_file
    
    def create_training_format(self, qa_pairs: List[QAPair]) -> List[Dict]:
        """Convert Q&A pairs to training format for fine-tuning"""
        training_data = []
        
        for qa in qa_pairs:
            # Create instruction-following format
            instruction = f"Answer the following question about Singapore financial regulations:"
            input_text = qa.question
            output_text = qa.answer
            
            training_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "source": qa.source,
                "category": qa.category
            })
        
        return training_data

def main():
    """Main function to run dataset preparation"""
    print("Starting Financial Regulation Dataset Preparation...")
    
    # Initialize dataset preparer
    prep = FinancialRegulationDatasetPrep()
    
    # Create sample dataset
    print("Creating sample financial regulation Q&A dataset...")
    qa_pairs = prep.create_sample_dataset()
    
    # Save dataset
    prep.save_dataset(qa_pairs)
    
    # Create training format
    training_data = prep.create_training_format(qa_pairs)
    
    # Save training format
    training_file = prep.output_dir / "training_data.json"
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"Training data saved to {training_file}")
    print(f"Training samples: {len(training_data)}")
    
    # Print sample
    print("\nSample Q&A pair:")
    sample = training_data[0]
    print(f"Instruction: {sample['instruction']}")
    print(f"Input: {sample['input']}")
    print(f"Output: {sample['output'][:200]}...")
    print(f"Category: {sample['category']}")

if __name__ == "__main__":
    main()
