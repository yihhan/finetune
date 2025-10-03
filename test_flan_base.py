"""
Test Flan-T5-Base (larger model) to see if it works better
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings("ignore")

def test_flan_t5_base():
    """Test the larger Flan-T5-base model"""
    
    print("🤖 Loading Flan-T5-BASE (larger model)...")
    
    try:
        # Load the larger base model
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        
        print("✅ Model loaded successfully!")
        
        # Test questions
        test_questions = [
            "What is 2+2?",
            "What are capital requirements for banks?",
            "Define financial regulation.",
            "What is MAS in Singapore?",
        ]
        
        print("\n🧪 Testing Flan-T5-BASE responses:")
        print("="*60)
        
        for question in test_questions:
            print(f"\nQ: {question}")
            
            # Try different input formats
            formats = [
                question,  # Direct question
                f"Answer: {question}",  # Simple format
                f"Answer this question: {question}",  # Explicit format
            ]
            
            for i, input_text in enumerate(formats):
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        num_beams=3,
                        do_sample=False,
                        early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"  Format {i+1}: {response}")
            
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ Error loading Flan-T5-base: {e}")
        print("Flan-T5-base might be too large for this environment")

def test_t5_small():
    """Test regular T5-small as backup"""
    
    print("\n🔄 Testing regular T5-small as backup...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        
        print("✅ T5-small loaded successfully!")
        
        # T5 needs specific format
        question = "question: What is 2+2?"
        
        inputs = tokenizer(question, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                num_beams=2,
                do_sample=False,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"T5-small response: {response}")
        
    except Exception as e:
        print(f"❌ Error with T5-small: {e}")

if __name__ == "__main__":
    test_flan_t5_base()
    test_t5_small()
