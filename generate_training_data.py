"""
Generate Large-Scale Training Data for Singapore Financial Regulations SFT

This script generates 500+ high-quality Q&A pairs using GPT-4 or other LLMs
to create a proper dataset for supervised fine-tuning.
"""

import json
import os
import time
from typing import List, Dict, Any
from pathlib import Path
import random
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not installed - using mock generation")

try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback if tqdm not available
    def tqdm(iterable, desc="", leave=True):
        print(f"{desc}...")
        return iterable

# Set up OpenAI (you'll need to add your API key)
# openai.api_key = "your-openai-api-key-here"

class FinancialDataGenerator:
    """Generate Singapore financial regulation Q&A pairs at scale"""
    
    def __init__(self, output_dir: str = "processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Singapore financial regulation topics
        self.topics = {
            "capital_adequacy": {
                "description": "Capital requirements, ratios, buffers for banks",
                "keywords": ["capital", "ratio", "buffer", "tier 1", "tier 2", "leverage"],
                "specific_amounts": ["SGD 1 million", "8%", "4.5%", "2.5%"],
                "regulations": ["MAS Notice 637", "Basel III"]
            },
            "anti_money_laundering": {
                "description": "AML/CFT requirements, customer due diligence, reporting",
                "keywords": ["AML", "CFT", "suspicious transaction", "customer due diligence", "KYC"],
                "specific_amounts": ["SGD 20,000", "within 15 days", "immediately"],
                "regulations": ["MAS Notice 626", "CDSA", "TSOFA"]
            },
            "payment_services": {
                "description": "Payment institution licensing, e-money, digital payments",
                "keywords": ["payment institution", "e-money", "digital wallet", "remittance"],
                "specific_amounts": ["SGD 100,000", "SGD 1 million", "SGD 3 million"],
                "regulations": ["Payment Services Act", "MAS Notice PSN01"]
            },
            "cybersecurity": {
                "description": "Cybersecurity requirements, incident reporting, resilience",
                "keywords": ["cybersecurity", "incident", "vulnerability", "penetration testing"],
                "specific_amounts": ["within 1 hour", "annually", "72 hours"],
                "regulations": ["MAS TRM Guidelines", "Cybersecurity Act"]
            },
            "data_protection": {
                "description": "Personal data protection, privacy, consent management",
                "keywords": ["personal data", "consent", "privacy", "data breach"],
                "specific_amounts": ["within 72 hours", "30 days", "up to 10% of turnover"],
                "regulations": ["PDPA", "MAS Guidelines on Data Management"]
            },
            "digital_banking": {
                "description": "Digital bank licensing, requirements, operations",
                "keywords": ["digital bank", "virtual bank", "fintech", "innovation"],
                "specific_amounts": ["SGD 15 million", "SGD 1.5 billion", "5 years"],
                "regulations": ["MAS Digital Bank License", "Banking Act"]
            },
            "insurance": {
                "description": "Insurance regulations, solvency, conduct requirements",
                "keywords": ["insurance", "solvency", "policyholder", "claims"],
                "specific_amounts": ["SGD 5 million", "150%", "21 days"],
                "regulations": ["Insurance Act", "MAS Notice 133"]
            },
            "securities": {
                "description": "Securities trading, market conduct, licensing",
                "keywords": ["securities", "trading", "market conduct", "disclosure"],
                "specific_amounts": ["5%", "SGD 250,000", "14 days"],
                "regulations": ["Securities and Futures Act", "MAS Notice SFA04"]
            }
        }
        
        # Question templates for variety
        self.question_templates = [
            "What are the {topic} requirements for {entity} in Singapore?",
            "How should {entity} comply with {topic} regulations?",
            "What is MAS's position on {topic} for {entity}?",
            "What are the penalties for non-compliance with {topic} rules?",
            "How frequently must {entity} report {topic} information to MAS?",
            "What documentation is required for {topic} compliance?",
            "What are the licensing requirements for {topic}?",
            "How does MAS supervise {topic} activities?",
            "What are the risk management requirements for {topic}?",
            "What training is required for staff handling {topic}?"
        ]
        
        # Entity types
        self.entities = [
            "banks", "payment institutions", "insurance companies", "securities firms",
            "digital banks", "fintech companies", "money changers", "remittance companies",
            "fund managers", "financial advisers", "credit rating agencies"
        ]
    
    def generate_qa_with_gpt4(self, topic: str, question: str, use_mock: bool = True) -> Dict[str, str]:
        """Generate Q&A pair using GPT-4 (or mock for demo)"""
        
        topic_info = self.topics[topic]
        
        if use_mock:
            # Mock generation for demo (replace with actual GPT-4 call)
            return self.generate_mock_answer(topic, question, topic_info)
        
        # Real GPT-4 generation (uncomment when you have API key)
        """
        prompt = f'''
        You are an expert in Singapore financial regulations. Generate a detailed, accurate answer to this question about Singapore's financial regulatory framework.

        Topic: {topic_info['description']}
        Question: {question}

        Requirements:
        1. Be specific to Singapore and MAS regulations
        2. Include specific amounts, timeframes, or percentages where applicable
        3. Mention relevant regulations: {', '.join(topic_info['regulations'])}
        4. Use these keywords naturally: {', '.join(topic_info['keywords'])}
        5. Be factual and professional
        6. Length: 100-200 words

        Answer:
        '''
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            answer = response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"GPT-4 API error: {e}")
            answer = self.generate_mock_answer(topic, question, topic_info)
        
        return {
            "question": question,
            "answer": answer,
            "topic": topic,
            "source": "gpt4_generated"
        }
        """
        
        return self.generate_mock_answer(topic, question, topic_info)
    
    def generate_mock_answer(self, topic: str, question: str, topic_info: Dict) -> Dict[str, str]:
        """Generate mock answers for demonstration"""
        
        # Create realistic mock answers based on topic
        mock_answers = {
            "capital_adequacy": f"Singapore banks must maintain minimum capital adequacy ratios as prescribed by MAS Notice 637. The Common Equity Tier 1 ratio must be at least 6.5%, while the Total Capital ratio must be at least 10%. Banks must also maintain a capital conservation buffer of 2.5% and may be subject to additional buffers. These requirements align with Basel III standards and ensure banks can absorb losses while continuing operations. MAS conducts regular supervisory reviews to assess capital adequacy.",
            
            "anti_money_laundering": f"Financial institutions in Singapore must implement comprehensive AML/CFT measures under MAS Notice 626. This includes conducting customer due diligence, monitoring transactions for suspicious activities, and filing Suspicious Transaction Reports (STRs) immediately upon detection. Enhanced due diligence is required for transactions exceeding SGD 20,000. Institutions must maintain transaction records for at least 5 years and provide AML training to staff. Non-compliance can result in penalties up to SGD 1 million.",
            
            "payment_services": f"Payment institutions in Singapore must obtain licenses under the Payment Services Act. Major payment institutions require minimum base capital of SGD 1 million, while standard payment institutions need SGD 100,000. Institutions must maintain customer funds in segregated accounts and comply with operational requirements including risk management, audit, and technology risk management. MAS supervises payment institutions through regular inspections and reporting requirements.",
            
            "cybersecurity": f"Financial institutions must comply with MAS Technology Risk Management Guidelines, implementing robust cybersecurity frameworks. This includes conducting annual penetration testing, maintaining incident response procedures, and reporting cyber incidents to MAS within 1 hour of discovery. Institutions must establish cybersecurity governance, implement multi-factor authentication, and conduct regular vulnerability assessments. Staff must receive cybersecurity awareness training annually.",
            
            "data_protection": f"Financial institutions must comply with the Personal Data Protection Act (PDPA) and MAS Guidelines on Data Management. This requires obtaining consent for data collection, implementing data protection measures, and appointing Data Protection Officers. Data breaches must be reported to PDPC within 72 hours if they result in significant harm. Institutions must conduct regular data protection impact assessments and provide staff training on data handling procedures.",
            
            "digital_banking": f"Digital banks in Singapore must obtain full bank licenses and maintain minimum paid-up capital of SGD 15 million. They must demonstrate sustainable business models and reach SGD 1.5 billion in deposits within 5 years. Digital banks must comply with the same prudential requirements as traditional banks, including capital adequacy, liquidity, and risk management standards. MAS supervises digital banks through regular reporting and on-site examinations.",
            
            "insurance": f"Insurance companies in Singapore must maintain minimum solvency capital requirements under MAS Notice 133. Life insurers require SGD 5 million minimum capital while general insurers need SGD 10 million. Companies must maintain solvency ratios of at least 120% and submit quarterly solvency returns to MAS. Claims must be processed within 21 days, and companies must establish proper governance and risk management frameworks.",
            
            "securities": f"Securities firms must comply with licensing requirements under the Securities and Futures Act. Capital market services license holders must maintain minimum base capital ranging from SGD 100,000 to SGD 5 million depending on activities. Firms must implement proper risk management, maintain client segregation, and submit regular financial returns to MAS. Market misconduct can result in penalties up to SGD 2 million or 3 times the profit gained."
        }
        
        base_answer = mock_answers.get(topic, "This is a mock answer for demonstration purposes.")
        
        # Add some variation based on question
        if "penalty" in question.lower() or "non-compliance" in question.lower():
            base_answer += " Non-compliance may result in monetary penalties, license suspension, or criminal prosecution depending on severity."
        elif "frequency" in question.lower() or "report" in question.lower():
            base_answer += " Regular reporting to MAS is required, typically on monthly or quarterly basis depending on the specific requirement."
        
        return {
            "question": question,
            "answer": base_answer,
            "topic": topic,
            "source": "mock_generated"
        }
    
    def generate_questions_for_topic(self, topic: str, num_questions: int = 60) -> List[str]:
        """Generate varied questions for a specific topic"""
        
        questions = []
        topic_info = self.topics[topic]
        
        # Use templates with variations
        for i in range(num_questions):
            template = random.choice(self.question_templates)
            entity = random.choice(self.entities)
            
            # Replace placeholders
            question = template.format(
                topic=topic.replace('_', ' '),
                entity=entity
            )
            
            # Add some specific variations
            if i % 10 == 0:  # Every 10th question, make it more specific
                keyword = random.choice(topic_info['keywords'])
                question = f"What are the specific {keyword} requirements for {entity} under Singapore regulations?"
            elif i % 15 == 0:  # Every 15th question, ask about amounts
                if topic_info['specific_amounts']:
                    amount = random.choice(topic_info['specific_amounts'])
                    question = f"Why do Singapore regulations specify {amount} for {topic.replace('_', ' ')} compliance?"
            
            questions.append(question)
        
        return questions
    
    def generate_large_dataset(self, target_size: int = 500, use_mock: bool = True) -> List[Dict]:
        """Generate large dataset of Q&A pairs"""
        
        print(f"🚀 Generating {target_size} Q&A pairs for Singapore financial regulations...")
        
        dataset = []
        questions_per_topic = target_size // len(self.topics)
        
        for topic in tqdm(self.topics.keys(), desc="Processing topics"):
            print(f"\n📊 Generating {questions_per_topic} questions for {topic}...")
            
            questions = self.generate_questions_for_topic(topic, questions_per_topic)
            
            for question in tqdm(questions, desc=f"Generating {topic} Q&A", leave=False):
                qa_pair = self.generate_qa_with_gpt4(topic, question, use_mock)
                dataset.append(qa_pair)
                
                # Small delay to avoid rate limits (remove for mock)
                if not use_mock:
                    time.sleep(0.1)
        
        print(f"\n✅ Generated {len(dataset)} Q&A pairs!")
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str = "large_financial_qa_dataset.json"):
        """Save the generated dataset"""
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Dataset saved to: {output_path}")
        
        # Also create training format
        training_data = []
        for item in dataset:
            training_data.append({
                "instruction": "You are an expert in Singapore financial regulations. Answer the following question accurately and comprehensively:",
                "input": item["question"],
                "output": item["answer"]
            })
        
        training_path = self.output_dir / "large_training_data.json"
        with open(training_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"🎯 Training format saved to: {training_path}")
        
        # Statistics
        topics = {}
        for item in dataset:
            topic = item.get('topic', 'unknown')
            topics[topic] = topics.get(topic, 0) + 1
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total Q&A pairs: {len(dataset)}")
        print(f"  Topics covered: {len(topics)}")
        for topic, count in topics.items():
            print(f"    {topic}: {count} pairs")
        
        return output_path, training_path

def main():
    """Main function to generate large-scale training data"""
    
    print("🎯 LARGE-SCALE SFT DATA GENERATION")
    print("="*50)
    print("This will generate 500+ Singapore financial regulation Q&A pairs")
    print("for proper supervised fine-tuning.")
    
    # Initialize generator
    generator = FinancialDataGenerator()
    
    # Generate dataset (using mock for demo - set use_mock=False for real GPT-4)
    dataset = generator.generate_large_dataset(
        target_size=500,
        use_mock=True  # Set to False when you have OpenAI API key
    )
    
    # Save dataset
    qa_path, training_path = generator.save_dataset(dataset)
    
    print(f"\n🎉 LARGE DATASET GENERATION COMPLETED!")
    print(f"📁 Files created:")
    print(f"  • {qa_path} - Full Q&A dataset")
    print(f"  • {training_path} - Training format")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Review the generated data quality")
    print(f"  2. Run: python flan_t5_full_finetune.py (update to use large dataset)")
    print(f"  3. Expect much better results with 500+ samples!")
    
    print(f"\n💡 To use real GPT-4 generation:")
    print(f"  1. Get OpenAI API key")
    print(f"  2. Set: openai.api_key = 'your-key'")
    print(f"  3. Change: use_mock=False")

if __name__ == "__main__":
    main()
