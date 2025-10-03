"""
Debug Flan-T5 Inference with Better Generation Parameters
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def test_flan_t5_generation():
    """Test Flan-T5 with different generation parameters"""
    
    print("🤖 Loading Flan-T5 models...")
    
    # Load base model
    base_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    
    # Try to load fine-tuned model
    try:
        lora_path = Path("flan_t5_financial_model/lora_adapters")
        if lora_path.exists():
            print("Loading fine-tuned model with LoRA...")
            ft_model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            print("Fine-tuned model not found, using base model")
            ft_model = base_model
    except:
        print("Error loading fine-tuned model, using base model")
        ft_model = base_model
    
    # Test question
    question = "What are the capital adequacy requirements for banks in Singapore?"
    input_text = f"Answer this financial regulation question: {question}"
    
    print(f"\n📝 Question: {question}")
    print(f"📝 Input prompt: {input_text}")
    print("="*80)
    
    # Test different generation strategies
    strategies = [
        {
            "name": "Conservative",
            "params": {
                "max_new_tokens": 100,
                "temperature": 0.1,
                "do_sample": False,  # Greedy decoding
                "num_beams": 1,
            }
        },
        {
            "name": "Beam Search",
            "params": {
                "max_new_tokens": 100,
                "num_beams": 4,
                "do_sample": False,
                "early_stopping": True,
            }
        },
        {
            "name": "Sampling",
            "params": {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            }
        }
    ]
    
    for model_name, model in [("Base Flan-T5", base_model), ("Fine-tuned", ft_model)]:
        print(f"\n🔍 Testing {model_name}:")
        print("-" * 40)
        
        # Tokenize input
        inputs = base_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
        
        for strategy in strategies:
            print(f"\n{strategy['name']} Strategy:")
            
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        **strategy['params'],
                        pad_token_id=base_tokenizer.pad_token_id,
                        eos_token_id=base_tokenizer.eos_token_id,
                    )
                
                response = base_tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"  Response: {response}")
                print(f"  Length: {len(response)} chars")
                
            except Exception as e:
                print(f"  Error: {e}")
        
        print("="*80)

def test_simple_prompts():
    """Test with very simple prompts"""
    
    print("\n🧪 Testing with simple prompts...")
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    
    simple_tests = [
        "What is 2+2?",
        "Define capital adequacy.",
        "What is MAS?",
        "Explain banking regulations.",
    ]
    
    for test in simple_tests:
        print(f"\nQ: {test}")
        
        inputs = tokenizer(test, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=3,
                do_sample=False,
                early_stopping=True,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"A: {response}")

if __name__ == "__main__":
    test_flan_t5_generation()
    test_simple_prompts()
