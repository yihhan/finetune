"""
Merge LoRA Adapters - Convert LoRA back to full model for inference

Sometimes LoRA adapters don't work properly during inference. 
This script merges them back into the base model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from pathlib import Path

def merge_lora_adapters():
    """Merge LoRA adapters back into base model"""
    
    print("🔄 MERGING LORA ADAPTERS")
    print("=" * 50)
    
    # Check if LoRA model exists
    model_path = Path("flan_t5_large_dataset_model")
    lora_path = model_path / "lora_adapters"
    merged_path = model_path / "merged_model"
    
    if not lora_path.exists():
        print(f"❌ LoRA adapters not found at: {lora_path}")
        return False
    
    print(f"✅ LoRA adapters found at: {lora_path}")
    
    try:
        # Load base model and LoRA adapters
        print("1. Loading base model...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        
        print("2. Loading LoRA adapters...")
        lora_model = PeftModel.from_pretrained(base_model, lora_path)
        
        print("3. Merging LoRA adapters into base model...")
        merged_model = lora_model.merge_and_unload()
        
        print("4. Saving merged model...")
        merged_path.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(merged_path)
        
        # Also save tokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        tokenizer.save_pretrained(merged_path)
        
        print(f"✅ Merged model saved to: {merged_path}")
        
        # Test the merged model
        print("\n5. Testing merged model...")
        test_input = "Answer this Singapore financial regulation question: What are capital requirements?"
        
        # Load merged model for testing
        merged_model_test = AutoModelForSeq2SeqLM.from_pretrained(merged_path)
        tokenizer_test = AutoTokenizer.from_pretrained(merged_path)
        
        inputs = tokenizer_test(test_input, return_tensors="pt")
        with torch.no_grad():
            outputs = merged_model_test.generate(**inputs, max_new_tokens=50, num_beams=3)
        response = tokenizer_test.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Merged model response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error merging LoRA adapters: {e}")
        return False

def test_merged_vs_base():
    """Compare merged model vs base model"""
    
    print("\n🧪 TESTING MERGED VS BASE")
    print("=" * 50)
    
    merged_path = Path("flan_t5_large_dataset_model/merged_model")
    
    if not merged_path.exists():
        print("❌ Merged model not found. Run merge first.")
        return
    
    # Load models
    base_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    
    merged_tokenizer = AutoTokenizer.from_pretrained(merged_path)
    merged_model = AutoModelForSeq2SeqLM.from_pretrained(merged_path)
    
    # Test questions
    test_questions = [
        "What are capital requirements for banks in Singapore?",
        "How should financial institutions implement AML measures?",
        "What are MAS cybersecurity requirements?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        
        # Base model
        base_inputs = base_tokenizer(f"Answer this Singapore financial regulation question: {question}", return_tensors="pt")
        with torch.no_grad():
            base_outputs = base_model.generate(**base_inputs, max_new_tokens=80, num_beams=3)
        base_response = base_tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        
        # Merged model
        merged_inputs = merged_tokenizer(f"Answer this Singapore financial regulation question: {question}", return_tensors="pt")
        with torch.no_grad():
            merged_outputs = merged_model.generate(**merged_inputs, max_new_tokens=80, num_beams=3)
        merged_response = merged_tokenizer.decode(merged_outputs[0], skip_special_tokens=True)
        
        print(f"   Base:   {base_response}")
        print(f"   Merged: {merged_response}")
        
        if base_response != merged_response:
            print("   ✅ SUCCESS: Different responses!")
        else:
            print("   ❌ PROBLEM: Still identical")

if __name__ == "__main__":
    success = merge_lora_adapters()
    if success:
        test_merged_vs_base()
    else:
        print("❌ Merging failed, cannot test")
