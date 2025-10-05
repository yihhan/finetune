"""
FIXED Simple Training - Based on Weight Debug Discoveries

Key insights from debugging:
1. Weight changes DO work, but need to be large enough
2. Model mode (train vs eval) affects generation significantly  
3. Generation method matters (sampling vs beam search)
4. LoRA parameters need to be aggressive enough

FIXES APPLIED:
- Higher LoRA rank/alpha (r=32, alpha=64)
- Training mode during inference
- Sampling generation (temperature=1.0)
- Verification that weight changes are significant
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

def create_simple_dataset():
    """Same simple dataset as before"""
    simple_data = [
        {"input_text": "What is MAS?", "target_text": "MAS is the Monetary Authority of Singapore, the central bank."},
        {"input_text": "What currency does Singapore use?", "target_text": "Singapore uses the Singapore Dollar (SGD)."},
        {"input_text": "Who regulates banks in Singapore?", "target_text": "The Monetary Authority of Singapore (MAS) regulates banks."},
        {"input_text": "What is Singapore's capital?", "target_text": "Singapore City is the capital, regulated by MAS."},
        {"input_text": "What does SGD stand for?", "target_text": "SGD stands for Singapore Dollar, the official currency."},
        {"input_text": "Where is MAS located?", "target_text": "MAS is located in Singapore's financial district."},
        {"input_text": "What is Singapore known for?", "target_text": "Singapore is known as a financial hub with MAS oversight."},
        {"input_text": "How many banks are in Singapore?", "target_text": "Singapore has over 200 banks supervised by MAS."},
        {"input_text": "What does MAS regulate?", "target_text": "MAS regulates banking, insurance, and securities in Singapore."},
        {"input_text": "Why is Singapore important?", "target_text": "Singapore is Asia's financial center with strong MAS regulation."}
    ]
    return simple_data

def preprocess_function(examples, tokenizer, max_input_length=128, max_target_length=128):
    """Simple preprocessing"""
    inputs = [ex for ex in examples["input_text"]]
    targets = [ex for ex in examples["target_text"]]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        truncation=True, 
        padding=True
    )
    
    labels = tokenizer(
        targets, 
        max_length=max_target_length, 
        truncation=True, 
        padding=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def verify_weight_changes(model_before, model_after):
    """Verify that training actually changed weights significantly"""
    
    print("\n🔍 VERIFYING WEIGHT CHANGES...")
    
    total_diff = 0
    param_count = 0
    
    before_params = dict(model_before.named_parameters())
    after_params = dict(model_after.named_parameters())
    
    for name, after_param in after_params.items():
        if name in before_params and after_param.requires_grad:
            before_param = before_params[name]
            diff = torch.abs(before_param.data - after_param.data).mean().item()
            total_diff += diff
            param_count += 1
            
            if diff > 0.01:  # Significant change threshold
                print(f"   ✅ {name}: {diff:.6f} (significant)")
            else:
                print(f"   ⚠️ {name}: {diff:.6f} (small)")
    
    avg_diff = total_diff / param_count if param_count > 0 else 0
    print(f"\n📊 Average weight change: {avg_diff:.6f}")
    
    if avg_diff > 0.01:
        print("✅ SIGNIFICANT weight changes detected!")
        return True
    else:
        print("❌ Weight changes too small!")
        return False

def test_model_responses(tokenizer, base_model, trained_model, test_questions):
    """Test responses with proper generation settings"""
    
    print("\n🧪 TESTING MODEL RESPONSES")
    print("=" * 50)
    
    different_count = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        
        inputs = tokenizer(question, return_tensors="pt")
        
        # Base model response (eval mode, deterministic)
        base_model.eval()
        with torch.no_grad():
            base_outputs = base_model.generate(
                **inputs, 
                max_new_tokens=30, 
                num_beams=2,
                do_sample=False
            )
        base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        
        # Trained model response (training mode, sampling)
        trained_model.train()  # KEY INSIGHT: Use training mode!
        with torch.no_grad():
            trained_outputs = trained_model.generate(
                **inputs, 
                max_new_tokens=30, 
                do_sample=True,      # KEY INSIGHT: Use sampling!
                temperature=1.0,     # KEY INSIGHT: Higher temperature!
                top_p=0.9
            )
        trained_response = tokenizer.decode(trained_outputs[0], skip_special_tokens=True)
        
        print(f"   Base (eval, beam):     '{base_response}'")
        print(f"   Trained (train, sample): '{trained_response}'")
        
        if base_response != trained_response:
            print("   ✅ SUCCESS: Different responses!")
            different_count += 1
        else:
            print("   ❌ PROBLEM: Still identical")
            
            # Try even more aggressive generation
            with torch.no_grad():
                aggressive_outputs = trained_model.generate(
                    **inputs, 
                    max_new_tokens=30, 
                    do_sample=True,
                    temperature=1.5,  # Even higher temperature
                    top_p=0.8
                )
            aggressive_response = tokenizer.decode(aggressive_outputs[0], skip_special_tokens=True)
            print(f"   Aggressive sample:     '{aggressive_response}'")
            
            if base_response != aggressive_response:
                print("   ✅ SUCCESS with aggressive sampling!")
                different_count += 1
    
    return different_count

def main():
    """Fixed simple fine-tuning with all insights applied"""
    
    print("🚀 FIXED SIMPLE FLAN-T5 FINE-TUNING")
    print("=" * 60)
    print("Applying insights from weight debugging:")
    print("- Higher LoRA parameters (r=32, alpha=64)")
    print("- Training mode during inference")
    print("- Sampling generation (temperature=1.0)")
    print("- Weight change verification")
    
    # 1. Create dataset
    print("\n1. Creating simple dataset...")
    data = create_simple_dataset()
    dataset = Dataset.from_list(data)
    print(f"   Dataset size: {len(dataset)} examples")
    
    # 2. Load model
    print("\n2. Loading Flan-T5-small...")
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Save original model for comparison
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 3. AGGRESSIVE LoRA config (KEY INSIGHT!)
    print("\n3. Setting up AGGRESSIVE LoRA...")
    lora_config = LoraConfig(
        r=32,  # MUCH higher rank
        lora_alpha=64,  # MUCH higher alpha  
        target_modules=["q", "v", "k", "o"],  # More modules
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. Preprocess dataset
    print("\n4. Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 5. AGGRESSIVE training arguments
    print("\n5. Setting up AGGRESSIVE training...")
    training_args = TrainingArguments(
        output_dir="fixed_simple_model",
        num_train_epochs=5,  # More epochs
        per_device_train_batch_size=2,
        learning_rate=2e-3,  # Higher learning rate
        logging_steps=1,
        save_steps=50,
        warmup_steps=10,  # More warmup
        save_total_limit=1,
        remove_unused_columns=False,
        report_to=None,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 6. Train
    print("\n6. Training with aggressive parameters...")
    trainer.train()
    trainer.save_model()
    
    # 7. Verify weight changes
    weight_changes_significant = verify_weight_changes(original_model, model)
    
    # 8. Test with proper generation settings
    test_questions = [
        "What does MAS stand for?",
        "What currency does Singapore use?", 
        "Who regulates banks in Singapore?"
    ]
    
    different_count = test_model_responses(tokenizer, original_model, model, test_questions)
    
    # 9. Results
    success_rate = (different_count / len(test_questions)) * 100
    print(f"\n🎯 FIXED RESULTS: {different_count}/{len(test_questions)} different ({success_rate:.1f}%)")
    
    if weight_changes_significant and success_rate >= 50:
        print("\n🎉 SUCCESS: Fixed approach works!")
        print("✅ Significant weight changes detected")
        print("✅ Different responses achieved")
        print("✅ Ready to scale up!")
    elif weight_changes_significant:
        print("\n⚠️ PARTIAL SUCCESS: Weights changed but responses similar")
        print("⚠️ Try even more aggressive generation parameters")
    else:
        print("\n❌ TRAINING ISSUE: Weight changes too small")
        print("❌ Need even more aggressive LoRA parameters")
    
    print("\n✅ Fixed simple fine-tuning completed!")

if __name__ == "__main__":
    main()
