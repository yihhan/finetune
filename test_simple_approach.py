"""
Test Simple Approach - Minimal validation script

This tests the absolute basics:
1. Can we fine-tune Flan-T5 at all?
2. Do we get different responses?
3. What's the minimum that works?
"""

def test_basic_concept():
    """Test if the basic concept works with minimal complexity"""
    
    print("🧪 TESTING BASIC FINE-TUNING CONCEPT")
    print("=" * 50)
    
    # Test 1: Can we load Flan-T5?
    print("1. Testing Flan-T5 loading...")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        
        print("   ✅ Flan-T5-small loads successfully")
        
        # Test basic generation
        test_input = "What is Singapore?"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        import torch
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"   ✅ Basic generation works: '{response}'")
        
    except Exception as e:
        print(f"   ❌ Failed to load Flan-T5: {e}")
        return False
    
    # Test 2: Can we set up LoRA?
    print("\n2. Testing LoRA setup...")
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            r=4,  # Minimal rank
            lora_alpha=8,
            target_modules=["q", "v"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        
        lora_model = get_peft_model(model, lora_config)
        lora_model.print_trainable_parameters()
        
        print("   ✅ LoRA setup successful")
        
    except Exception as e:
        print(f"   ❌ Failed to setup LoRA: {e}")
        return False
    
    # Test 3: Can we create a simple dataset?
    print("\n3. Testing dataset creation...")
    try:
        from datasets import Dataset
        
        simple_data = [
            {"input": "What is MAS?", "output": "MAS is the Monetary Authority of Singapore."},
            {"input": "What is SGD?", "output": "SGD is the Singapore Dollar currency."},
            {"input": "Singapore capital?", "output": "Singapore City is the capital of Singapore."}
        ]
        
        dataset = Dataset.from_list(simple_data)
        print(f"   ✅ Dataset created: {len(dataset)} examples")
        
    except Exception as e:
        print(f"   ❌ Failed to create dataset: {e}")
        return False
    
    # Test 4: Can we tokenize?
    print("\n4. Testing tokenization...")
    try:
        def tokenize_function(examples):
            inputs = tokenizer(examples["input"], truncation=True, padding=True, max_length=64)
            targets = tokenizer(examples["output"], truncation=True, padding=True, max_length=64)
            inputs["labels"] = targets["input_ids"]
            return inputs
        
        tokenized = dataset.map(tokenize_function, batched=True)
        print("   ✅ Tokenization successful")
        
    except Exception as e:
        print(f"   ❌ Failed to tokenize: {e}")
        return False
    
    print("\n🎯 BASIC CONCEPT TEST RESULTS:")
    print("✅ All basic components work!")
    print("✅ Ready to try minimal fine-tuning")
    print("\n💡 Next: Run simple_flan_t5_finetune.py")
    
    return True

def check_environment():
    """Check if we have the right environment"""
    
    print("🔍 CHECKING ENVIRONMENT")
    print("=" * 30)
    
    # Check Python packages
    required_packages = [
        "torch", "transformers", "datasets", "peft", "accelerate"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("\n✅ All packages available")
        return True

if __name__ == "__main__":
    if check_environment():
        test_basic_concept()
    else:
        print("❌ Environment check failed - fix dependencies first")
