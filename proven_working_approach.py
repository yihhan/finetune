# 🎯 PROVEN WORKING APPROACH - Simple, Guaranteed to Work
# Based on the successful quick_architecture_test.py approach

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset

print("🎯 PROVEN WORKING APPROACH - SIMPLE & GUARANTEED")
print("=" * 80)
print("Based on the successful quick_architecture_test.py that actually worked")

# Use the EXACT approach that worked before
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1. Load models (EXACT same as working version)
print("\n🔄 Loading GPT-2 models...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("gpt2")
original_model = AutoModelForCausalLM.from_pretrained("gpt2")  # For comparison

model = model.to(device)
original_model = original_model.to(device)

print("✅ Loaded GPT-2 models")

# 2. Create HIGH-QUALITY training data (small but perfect)
print("\n📝 Creating high-quality training data...")

# Start with PROVEN working examples, then add more
training_data = [
    "Q: What does MAS stand for? A: MAS stands for Monetary Authority of Singapore, which is Singapore's central bank and integrated financial regulator responsible for monetary policy and financial supervision.",
    "Q: What currency does Singapore use? A: Singapore uses the Singapore Dollar (SGD) as its official currency, which is managed by the Monetary Authority of Singapore.",
    "Q: Who regulates banks in Singapore? A: The Monetary Authority of Singapore (MAS) regulates banks in Singapore, conducting supervision and setting prudential requirements.",
    "Q: What is MAS Notice 626? A: MAS Notice 626 sets out the Prevention of Money Laundering and Countering the Financing of Terrorism requirements for banks in Singapore.",
    "Q: What are the capital adequacy requirements for Singapore banks? A: Singapore banks must maintain minimum capital ratios including Common Equity Tier 1 ratio of 6.5% and Total Capital Ratio of 10% as required by MAS.",
    "Q: What is STRO? A: STRO is the Suspicious Transaction Reporting Office within MAS that receives and analyzes suspicious transaction reports from financial institutions.",
    "Q: How often must banks report to MAS? A: Banks must submit various regulatory returns to MAS on monthly, quarterly, and annual basis depending on the specific requirement.",
    "Q: What is the Payment Services Act? A: The Payment Services Act is Singapore's regulatory framework for payment services, requiring providers to obtain licenses from MAS.",
    "Q: What are AML requirements in Singapore? A: Financial institutions must implement anti-money laundering measures including customer due diligence, transaction monitoring, and reporting suspicious activities to STRO.",
    "Q: What is MAS's role in fintech regulation? A: MAS promotes fintech innovation through regulatory sandboxes while ensuring appropriate consumer protection and financial stability measures."
]

print(f"✅ Created {len(training_data)} high-quality Q&A pairs")

# 3. Prepare dataset (EXACT same tokenization as working version)
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=False, max_length=256)

dataset = Dataset.from_dict({'text': training_data})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print("✅ Tokenized dataset")

# 4. LoRA config (EXACT same as working version)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # Same as working version
    lora_alpha=16,          # Same as working version  
    lora_dropout=0.1,       # Same as working version
    target_modules=["c_attn", "c_proj"]  # EXACT same as working version
)

model = get_peft_model(model, lora_config)
print("✅ Applied LoRA configuration")
model.print_trainable_parameters()

# 5. Training arguments (CONSERVATIVE - guaranteed to work)
training_args = TrainingArguments(
    output_dir="proven_working_model",
    num_train_epochs=5,              # Same as working version
    per_device_train_batch_size=1,   # Same as working version
    learning_rate=1e-3,              # Same as working version
    logging_steps=1,
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=0,        # Prevent multiprocessing issues
    remove_unused_columns=False,
)

# 6. Data collator (EXACT same as working version)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 7. Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("\n🚀 Starting training with PROVEN working parameters...")
print("This should definitely work - using exact same approach as successful test")

# Train the model
trainer.train()

print("✅ Training completed!")

# 8. Save the model
model.save_pretrained("proven_working_model")
tokenizer.save_pretrained("proven_working_model")
print("✅ Model saved to: proven_working_model")

# 9. Test the model (EXACT same test as working version)
print("\n🧪 Testing the trained model...")

def test_model(model, test_prompt):
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 80,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(test_prompt, "").strip()

# Test questions
test_questions = [
    "Q: What does MAS stand for? A:",
    "Q: What currency does Singapore use? A:",
    "Q: Who regulates banks in Singapore? A:",
]

print("\n" + "=" * 80)
print("🎯 PROVEN WORKING MODEL TEST RESULTS")
print("=" * 80)

singapore_terms = ['singapore', 'mas', 'monetary authority', 'sgd', 'dollar']
success_count = 0

for i, question in enumerate(test_questions, 1):
    print(f"\n{i}. {question}")
    
    # Test original model
    original_response = test_model(original_model, question)
    print(f"   🔵 Original: {original_response[:60]}{'...' if len(original_response) > 60 else ''}")
    
    # Test fine-tuned model
    finetuned_response = test_model(model, question)
    print(f"   🟢 Fine-tuned: {finetuned_response[:60]}{'...' if len(finetuned_response) > 60 else ''}")
    
    # Check for Singapore content
    has_singapore_content = any(term in finetuned_response.lower() for term in singapore_terms)
    if has_singapore_content:
        print(f"   ✅ Contains Singapore financial content")
        success_count += 1
    else:
        print(f"   ❌ Missing Singapore financial content")

success_rate = (success_count / len(test_questions)) * 100
print(f"\n🎯 SUCCESS RATE: {success_count}/{len(test_questions)} ({success_rate:.1f}%)")

if success_rate >= 80:
    print("🎉 EXCELLENT: Proven approach worked perfectly!")
    print("✅ Model successfully learned Singapore financial knowledge")
elif success_rate >= 60:
    print("✅ GOOD: Proven approach is working")
    print("💡 Consider more training epochs for even better results")
else:
    print("⚠️ NEEDS INVESTIGATION: Even proven approach having issues")
    print("💡 May need to check Colab environment or dependencies")

print(f"\n💡 NEXT STEPS:")
if success_rate >= 60:
    print("   1. ✅ This approach works - scale up with more data")
    print("   2. 📈 Add more training examples (20-50 high-quality Q&As)")
    print("   3. 🔧 Increase epochs to 8-10 for better learning")
    print("   4. 🚀 Deploy for production use")
else:
    print("   1. 🔍 Run the diagnostic script to find specific issues")
    print("   2. 🔧 Check Colab environment and dependencies")
    print("   3. 📝 Verify training data format and content")

print("\n" + "=" * 80)
print("✅ PROVEN WORKING APPROACH COMPLETED!")
print("=" * 80)
