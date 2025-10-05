#!/usr/bin/env python3
"""
Quick test to see if different model architectures can learn basic Singapore Q&A
This will determine if the issue is Flan-T5 specific or fine-tuning in general
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset

def test_gpt2():
    """Test GPT-2 with simple Q&A learning"""
    print("🤖 Testing GPT-2 (Causal LM)")
    print("=" * 40)
    
    # Load GPT-2
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    original_model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Simple Q&A data in prompt-completion format
    training_data = [
        "Q: What does MAS stand for? A: MAS stands for Monetary Authority of Singapore.",
        "Q: What currency does Singapore use? A: Singapore uses the Singapore Dollar (SGD).",
        "Q: Who regulates banks in Singapore? A: The Monetary Authority of Singapore regulates banks."
    ]
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)
    
    dataset = Dataset.from_dict({'text': training_data})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # LoRA for GPT-2
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training
    training_args = TrainingArguments(
        output_dir="gpt2_test",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        learning_rate=1e-3,
        logging_steps=1,
        save_steps=100,
        report_to="none"
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    print("🚀 Training GPT-2...")
    trainer.train()
    
    # Test
    test_prompt = "Q: What does MAS stand for? A:"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    original_model = original_model.to(device)
    
    # Base response
    original_model.eval()
    with torch.no_grad():
        base_outputs = original_model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
    
    # Fine-tuned response
    model.eval()
    with torch.no_grad():
        ft_outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    ft_response = tokenizer.decode(ft_outputs[0], skip_special_tokens=True)
    
    print(f"\n📊 GPT-2 Results:")
    print(f"   Base:       '{base_response}'")
    print(f"   Fine-tuned: '{ft_response}'")
    
    # Check for success
    if 'monetary authority' in ft_response.lower() or 'singapore' in ft_response.lower():
        print("   ✅ SUCCESS: GPT-2 learned Singapore content!")
        return True
    elif base_response != ft_response:
        print("   ⚠️ PARTIAL: Different response but no Singapore content")
        return False
    else:
        print("   ❌ FAILED: Identical responses")
        return False

def test_original_t5():
    """Test original T5 (not Flan-T5)"""
    print("\n🤖 Testing Original T5-small")
    print("=" * 40)
    
    # Load original T5
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    original_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    
    # Simple Q&A data
    training_data = [
        {'input': 'What does MAS stand for?', 'output': 'Monetary Authority of Singapore'},
        {'input': 'What currency does Singapore use?', 'output': 'Singapore Dollar'},
        {'input': 'Who regulates banks in Singapore?', 'output': 'Monetary Authority of Singapore'}
    ]
    
    def preprocess_function(examples):
        inputs = examples['input']
        targets = examples['output']
        model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding=True)
        labels = tokenizer(targets, max_length=64, truncation=True, padding=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    dataset = Dataset.from_list(training_data)
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    
    # LoRA for T5
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["q", "v", "k", "o", "wi", "wo"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training
    training_args = TrainingArguments(
        output_dir="t5_original_test",
        num_train_epochs=10,
        per_device_train_batch_size=1,
        learning_rate=1e-3,
        logging_steps=1,
        save_steps=100,
        report_to="none"
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    print("🚀 Training original T5...")
    trainer.train()
    
    # Test
    test_input = "What does MAS stand for?"
    inputs = tokenizer(test_input, return_tensors="pt")
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    original_model = original_model.to(device)
    
    # Base response
    original_model.eval()
    with torch.no_grad():
        base_outputs = original_model.generate(**inputs, max_new_tokens=15)
    base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
    
    # Fine-tuned response
    model.eval()
    with torch.no_grad():
        ft_outputs = model.generate(**inputs, max_new_tokens=15)
    ft_response = tokenizer.decode(ft_outputs[0], skip_special_tokens=True)
    
    print(f"\n📊 Original T5 Results:")
    print(f"   Base:       '{base_response}'")
    print(f"   Fine-tuned: '{ft_response}'")
    
    # Check for success
    if 'monetary authority' in ft_response.lower() or 'singapore' in ft_response.lower():
        print("   ✅ SUCCESS: Original T5 learned Singapore content!")
        return True
    elif base_response != ft_response:
        print("   ⚠️ PARTIAL: Different response but no Singapore content")
        return False
    else:
        print("   ❌ FAILED: Identical responses")
        return False

def main():
    """Run architecture compatibility tests"""
    print("🔄 ARCHITECTURE COMPATIBILITY TEST")
    print("=" * 60)
    print("Testing if ANY model can learn basic Singapore Q&A")
    
    results = {}
    
    # Test GPT-2
    try:
        results['GPT-2'] = test_gpt2()
    except Exception as e:
        print(f"❌ GPT-2 test failed with error: {e}")
        results['GPT-2'] = False
    
    # Test original T5
    try:
        results['T5-original'] = test_original_t5()
    except Exception as e:
        print(f"❌ T5-original test failed with error: {e}")
        results['T5-original'] = False
    
    # Final analysis
    print(f"\n🎯 FINAL ARCHITECTURE ANALYSIS")
    print("=" * 60)
    
    all_results = {
        'Flan-T5 (LoRA limited)': False,
        'Flan-T5 (LoRA comprehensive)': False, 
        'Flan-T5 (Full fine-tuning)': False,
        'GPT-2 (LoRA)': results.get('GPT-2', False),
        'T5-original (LoRA)': results.get('T5-original', False)
    }
    
    print("📊 COMPLETE RESULTS:")
    for approach, success in all_results.items():
        status = '✅ SUCCESS' if success else '❌ FAILED'
        print(f"   {approach}: {status}")
    
    successful_approaches = [k for k, v in all_results.items() if v]
    
    if successful_approaches:
        print(f"\n🎉 BREAKTHROUGH FOUND!")
        print(f"✅ Working approaches: {', '.join(successful_approaches)}")
        print("\n🚀 Next Steps:")
        print("   • Scale up successful approach with larger dataset")
        print("   • Implement full Singapore financial Q&A system")
        print("   • Compare performance vs GPT-4 RAG baseline")
    else:
        print("\n💥 COMPLETE FINE-TUNING FAILURE")
        print("❌ ALL architectures failed to learn basic Q&A")
        print("\n🔍 This means:")
        print("   • Fine-tuning approach is fundamentally flawed")
        print("   • Need completely different methodology")
        print("\n💡 Alternative Approaches:")
        print("   • RAG with embedding-based retrieval")
        print("   • Few-shot learning with GPT-4")
        print("   • Prompt engineering with base models")
        print("   • Custom retrieval + generation pipeline")

if __name__ == "__main__":
    main()
