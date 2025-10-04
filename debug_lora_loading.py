"""
Debug LoRA Loading - Check if LoRA adapters are actually being applied
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from pathlib import Path
import json

def test_lora_loading():
    """Test if LoRA adapters are actually being applied"""
    
    print("🔍 DEBUGGING LORA LOADING")
    print("=" * 50)
    
    # Load base model
    print("1. Loading base Flan-T5-base...")
    base_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    
    # Check if LoRA model exists
    model_path = Path("flan_t5_large_dataset_model")
    lora_path = model_path / "lora_adapters"
    
    if not lora_path.exists():
        print(f"❌ LoRA adapters not found at: {lora_path}")
        print("❌ Training probably failed or incomplete")
        return
    
    print(f"✅ LoRA adapters found at: {lora_path}")
    
    # List LoRA files
    print("\n2. LoRA adapter files:")
    for file in lora_path.iterdir():
        print(f"   - {file.name}")
    
    # Load LoRA model
    print("\n3. Loading LoRA model...")
    try:
        base_model_copy = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        lora_model = PeftModel.from_pretrained(base_model_copy, lora_path)
        print("✅ LoRA model loaded successfully")
        
        # Check if models are actually different
        print("\n4. Checking model parameters...")
        
        # Get a parameter from base model
        base_param = None
        lora_param = None
        
        for name, param in base_model.named_parameters():
            if 'encoder.block.0.layer.0.SelfAttention.q.weight' in name:
                base_param = param.clone()
                break
        
        for name, param in lora_model.named_parameters():
            if 'encoder.block.0.layer.0.SelfAttention.q.weight' in name:
                lora_param = param.clone()
                break
        
        if base_param is not None and lora_param is not None:
            if torch.equal(base_param, lora_param):
                print("❌ WARNING: Base and LoRA parameters are identical!")
                print("❌ This suggests LoRA adapters aren't being applied")
            else:
                print("✅ Base and LoRA parameters are different")
                print("✅ LoRA adapters are being applied")
        
        # Test generation
        print("\n5. Testing generation...")
        test_input = "Answer this Singapore financial regulation question: What are capital requirements?"
        
        # Base model response
        base_inputs = base_tokenizer(test_input, return_tensors="pt")
        with torch.no_grad():
            base_outputs = base_model.generate(**base_inputs, max_new_tokens=50, num_beams=3)
        base_response = base_tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        
        # LoRA model response  
        lora_inputs = base_tokenizer(test_input, return_tensors="pt")
        with torch.no_grad():
            lora_outputs = lora_model.generate(**lora_inputs, max_new_tokens=50, num_beams=3)
        lora_response = base_tokenizer.decode(lora_outputs[0], skip_special_tokens=True)
        
        print(f"\nBase response: {base_response}")
        print(f"LoRA response: {lora_response}")
        
        if base_response == lora_response:
            print("❌ PROBLEM: Responses are identical!")
            print("❌ LoRA adapters may not be working properly")
        else:
            print("✅ SUCCESS: Responses are different!")
            print("✅ LoRA adapters are working")
            
    except Exception as e:
        print(f"❌ Error loading LoRA model: {e}")
    
    print("\n" + "=" * 50)
    print("🔍 LoRA debugging completed!")

if __name__ == "__main__":
    test_lora_loading()
