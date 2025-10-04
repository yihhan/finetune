"""
SUPER SIMPLE Debug - Test if ANYTHING works

Let's test the most basic concepts:
1. Can we load a model that gives sensible responses?
2. Can we change ANY weights at all?
3. What's the minimum that works?
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test_different_models():
    """Test different models to find one that works"""
    
    print("🔍 TESTING DIFFERENT MODELS")
    print("=" * 40)
    
    models_to_test = [
        "google/flan-t5-small",
        "google/flan-t5-base", 
        "t5-small",
        "google/t5-efficient-tiny"
    ]
    
    test_question = "What currency does Singapore use?"
    
    for model_name in models_to_test:
        print(f"\n📊 Testing: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            inputs = tokenizer(test_question, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=20, num_beams=2)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"   Response: '{response}'")
            
            # Check if response makes sense
            if "singapore" in response.lower() or "sgd" in response.lower() or "dollar" in response.lower():
                print("   ✅ SENSIBLE response!")
            else:
                print("   ❌ Nonsense response")
                
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")

def test_manual_weight_change():
    """Test if we can manually change model weights"""
    
    print("\n🔧 TESTING MANUAL WEIGHT CHANGES")
    print("=" * 40)
    
    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        
        test_input = "What is Singapore?"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        # Get original response
        with torch.no_grad():
            original_outputs = model.generate(**inputs, max_new_tokens=10, num_beams=2)
        original_response = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
        print(f"Original response: '{original_response}'")
        
        # Manually modify a weight (just to see if anything changes)
        print("\nManually changing model weights...")
        with torch.no_grad():
            # Find first parameter and add some noise
            for name, param in model.named_parameters():
                if param.requires_grad and len(param.shape) > 1:
                    print(f"Modifying: {name}")
                    param.data += torch.randn_like(param.data) * 0.01  # Small random noise
                    break
        
        # Get new response
        with torch.no_grad():
            new_outputs = model.generate(**inputs, max_new_tokens=10, num_beams=2)
        new_response = tokenizer.decode(new_outputs[0], skip_special_tokens=True)
        print(f"Modified response: '{new_response}'")
        
        if original_response != new_response:
            print("✅ SUCCESS: Manual weight change affected output!")
            return True
        else:
            print("❌ PROBLEM: Manual weight change had no effect")
            return False
            
    except Exception as e:
        print(f"❌ Manual weight test failed: {e}")
        return False

def test_tiny_training():
    """Test the tiniest possible training"""
    
    print("\n🎯 TESTING TINY TRAINING")
    print("=" * 40)
    
    try:
        from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
        from datasets import Dataset
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        
        # Create ONE training example
        data = [{"input_text": "What is MAS?", "target_text": "MAS is the Monetary Authority of Singapore"}]
        dataset = Dataset.from_list(data)
        
        def preprocess(examples):
            inputs = tokenizer(examples["input_text"], truncation=True, padding=True, max_length=64)
            targets = tokenizer(examples["target_text"], truncation=True, padding=True, max_length=64)
            inputs["labels"] = targets["input_ids"]
            return inputs
        
        tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
        
        # Training args - VERY simple
        training_args = TrainingArguments(
            output_dir="tiny_test",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            learning_rate=1e-2,  # High LR
            logging_steps=1,
            save_steps=1000,  # Don't save
            report_to=None,
            remove_unused_columns=False,
        )
        
        # Train for just 1 step
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model),
        )
        
        # Test before training
        test_input = "What is MAS?"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            before_outputs = model.generate(**inputs, max_new_tokens=10)
        before_response = tokenizer.decode(before_outputs[0], skip_special_tokens=True)
        print(f"Before training: '{before_response}'")
        
        # Train
        print("Training for 1 epoch...")
        trainer.train()
        
        # Test after training
        with torch.no_grad():
            after_outputs = model.generate(**inputs, max_new_tokens=10)
        after_response = tokenizer.decode(after_outputs[0], skip_special_tokens=True)
        print(f"After training: '{after_response}'")
        
        if before_response != after_response:
            print("✅ SUCCESS: Training changed the response!")
            return True
        else:
            print("❌ PROBLEM: Training had no effect")
            return False
            
    except Exception as e:
        print(f"❌ Tiny training failed: {e}")
        return False

def main():
    """Run all basic tests"""
    
    print("🚨 SUPER SIMPLE DEBUG TESTS")
    print("=" * 50)
    print("Testing the most basic concepts to find what works...")
    
    # Test 1: Different models
    test_different_models()
    
    # Test 2: Manual weight changes
    manual_works = test_manual_weight_change()
    
    # Test 3: Tiny training (only if manual works)
    if manual_works:
        training_works = test_tiny_training()
    else:
        print("\n⚠️ Skipping training test - manual changes don't work")
        training_works = False
    
    print("\n" + "=" * 50)
    print("🎯 DIAGNOSTIC SUMMARY:")
    print(f"   Manual weight changes work: {manual_works}")
    print(f"   Tiny training works: {training_works}")
    
    if not manual_works:
        print("\n❌ FUNDAMENTAL ISSUE: Can't even change model manually")
        print("❌ Problem is with model loading or generation")
    elif not training_works:
        print("\n❌ TRAINING ISSUE: Manual works but training doesn't")
        print("❌ Problem is with training setup or LoRA")
    else:
        print("\n✅ BASIC CONCEPTS WORK: Can proceed with real fine-tuning")

if __name__ == "__main__":
    main()
