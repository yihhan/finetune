"""
SIMPLE Flan-T5 Fine-tuning - Back to Basics

This is a minimal, proven approach based on working examples.
No complexity, just the essentials that actually work.
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
import os

def create_simple_dataset():
    """Create a tiny, simple dataset to test if training works at all"""
    
    # Just 10 simple examples - if this doesn't work, nothing will
    simple_data = [
        {
            "input_text": "What is the capital of Singapore?",
            "target_text": "The capital of Singapore is Singapore City, and MAS is the central bank."
        },
        {
            "input_text": "What does MAS stand for?", 
            "target_text": "MAS stands for Monetary Authority of Singapore, the central bank and financial regulator."
        },
        {
            "input_text": "What currency does Singapore use?",
            "target_text": "Singapore uses the Singapore Dollar (SGD) as its official currency."
        },
        {
            "input_text": "Who regulates banks in Singapore?",
            "target_text": "The Monetary Authority of Singapore (MAS) regulates all banks and financial institutions."
        },
        {
            "input_text": "What is Singapore's financial center?",
            "target_text": "Singapore is a major financial hub in Asia, regulated by MAS with strict banking standards."
        },
        {
            "input_text": "How many banks are in Singapore?",
            "target_text": "Singapore has over 200 banks and financial institutions, all supervised by MAS."
        },
        {
            "input_text": "What is Singapore known for?",
            "target_text": "Singapore is known as a global financial center with world-class banking regulations by MAS."
        },
        {
            "input_text": "Where is MAS located?",
            "target_text": "MAS (Monetary Authority of Singapore) is located in Singapore's central business district."
        },
        {
            "input_text": "What does Singapore regulate?",
            "target_text": "Singapore regulates banking, insurance, securities and payment systems through MAS."
        },
        {
            "input_text": "Why is Singapore important?",
            "target_text": "Singapore is important as Asia's financial hub with strong MAS oversight and SGD stability."
        }
    ]
    
    return simple_data

def preprocess_function(examples, tokenizer, max_input_length=128, max_target_length=128):
    """Simple preprocessing - just tokenize input and target"""
    
    # Tokenize inputs
    inputs = [ex for ex in examples["input_text"]]
    targets = [ex for ex in examples["target_text"]]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        truncation=True, 
        padding=True
    )
    
    # Tokenize targets
    labels = tokenizer(
        targets, 
        max_length=max_target_length, 
        truncation=True, 
        padding=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    """Simple fine-tuning that should actually work"""
    
    print("🚀 SIMPLE FLAN-T5 FINE-TUNING")
    print("=" * 50)
    
    # 1. Create simple dataset
    print("1. Creating simple dataset...")
    data = create_simple_dataset()
    dataset = Dataset.from_list(data)
    print(f"   Dataset size: {len(dataset)} examples")
    
    # 2. Load model and tokenizer
    print("2. Loading Flan-T5-small...")
    model_name = "google/flan-t5-small"  # Start small
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 3. Simple LoRA config
    print("3. Setting up simple LoRA...")
    lora_config = LoraConfig(
        r=8,  # Very small rank
        lora_alpha=16,  # 2x rank
        target_modules=["q", "v"],  # Just attention
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. Preprocess dataset
    print("4. Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 5. Simple training arguments
    print("5. Setting up training...")
    training_args = TrainingArguments(
        output_dir="simple_flan_t5_model",
        num_train_epochs=3,  # Just 3 epochs
        per_device_train_batch_size=2,  # Small batch
        learning_rate=1e-3,  # Higher learning rate
        logging_steps=1,
        save_steps=50,
        eval_steps=50,
        warmup_steps=10,
        save_total_limit=1,
        remove_unused_columns=False,
        report_to=None,
    )
    
    # 6. Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # 7. Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 8. Train
    print("6. Starting training...")
    trainer.train()
    
    # 9. Save model
    print("7. Saving model...")
    trainer.save_model()
    
    # 10. Test the model
    print("8. Testing model...")
    test_model()
    
    print("✅ Simple fine-tuning completed!")

def test_model():
    """Test if the simple model actually learned something"""
    
    print("\n🧪 TESTING SIMPLE MODEL")
    print("=" * 30)
    
    # Load the trained model
    model_path = "simple_flan_t5_model"
    if not os.path.exists(model_path):
        print("❌ Model not found!")
        return
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    
    # Load base model for comparison
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    
    # Load fine-tuned model
    from peft import PeftModel
    fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    fine_tuned_model = PeftModel.from_pretrained(fine_tuned_model, model_path)
    
    # Test questions
    test_questions = [
        "What does MAS stand for?",
        "What currency does Singapore use?",
        "Who regulates banks in Singapore?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        
        # Tokenize
        inputs = tokenizer(question, return_tensors="pt")
        
        # Base model response
        with torch.no_grad():
            base_outputs = base_model.generate(**inputs, max_new_tokens=50, num_beams=2)
        base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        
        # Fine-tuned model response
        with torch.no_grad():
            ft_outputs = fine_tuned_model.generate(**inputs, max_new_tokens=50, num_beams=2)
        ft_response = tokenizer.decode(ft_outputs[0], skip_special_tokens=True)
        
        print(f"   Base:       {base_response}")
        print(f"   Fine-tuned: {ft_response}")
        
        if base_response != ft_response:
            print("   ✅ SUCCESS: Different responses!")
        else:
            print("   ❌ PROBLEM: Still identical")

if __name__ == "__main__":
    main()
