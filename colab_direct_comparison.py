# 🔄 DIRECT COLAB COMPARISON - Copy this entire cell into Colab
# This avoids any caching issues with downloaded files

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time

print("🔄 BASE vs FINE-TUNED MODEL COMPARISON")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load models
print("\n🔄 Loading models...")
try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    base_model = base_model.to(device)
    print(f"✅ Loaded BASE GPT-2 model")
    
    # Load another copy for fine-tuned version
    base_for_peft = AutoModelForCausalLM.from_pretrained("gpt2")
    base_for_peft = base_for_peft.to(device)
    
    # Load fine-tuned model - CORRECT PATH
    model_path = "gpt2_comprehensive_singapore_model/checkpoint-1080"
    print(f"🔍 Trying to load from: {model_path}")
    finetuned_model = PeftModel.from_pretrained(base_for_peft, model_path)
    print(f"✅ Loaded FINE-TUNED model from: {model_path}")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("\n🔍 Let's check what checkpoints exist:")
    import os
    if os.path.exists("gpt2_comprehensive_singapore_model"):
        checkpoints = [d for d in os.listdir("gpt2_comprehensive_singapore_model") if d.startswith("checkpoint-")]
        print(f"Available checkpoints: {checkpoints}")
        if checkpoints:
            # Use the latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            model_path = f"gpt2_comprehensive_singapore_model/{latest_checkpoint}"
            print(f"🔄 Trying latest checkpoint: {model_path}")
            try:
                finetuned_model = PeftModel.from_pretrained(base_for_peft, model_path)
                print(f"✅ Loaded FINE-TUNED model from: {model_path}")
            except Exception as e2:
                print(f"❌ Still failed: {e2}")
                exit()
        else:
            print("❌ No checkpoints found")
            exit()
    else:
        print("❌ Model directory not found")
        exit()

# Quick test questions
test_questions = [
    "What does MAS stand for?",
    "What currency does Singapore use?", 
    "Who regulates banks in Singapore?",
    "What are the capital adequacy requirements for Singapore banks?",
    "What is MAS Notice 626?"
]

def generate_response(model, question, max_length=80):
    """Generate response from model"""
    prompt = f"Q: {question} A:"
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    
    except Exception as e:
        return f"Error: {str(e)}"

# Singapore financial terms
singapore_terms = ['mas', 'singapore', 'monetary authority', 'sgd', 'dollar', 'notice', 'regulation', 'financial', 'bank', 'capital']

def has_singapore_content(text):
    """Check if response contains Singapore financial content"""
    if not text:
        return False
    text_lower = text.lower()
    return any(term in text_lower for term in singapore_terms)

# Run quick comparison
print(f"\n🚀 QUICK COMPARISON TEST...")
print("=" * 80)

base_singapore_count = 0
ft_singapore_count = 0

for i, question in enumerate(test_questions, 1):
    print(f"\n📝 QUESTION {i}: {question}")
    print("-" * 60)
    
    # Generate responses
    base_response = generate_response(base_model, question)
    ft_response = generate_response(finetuned_model, question)
    
    # Check Singapore content
    base_has_sg = has_singapore_content(base_response)
    ft_has_sg = has_singapore_content(ft_response)
    
    if base_has_sg:
        base_singapore_count += 1
    if ft_has_sg:
        ft_singapore_count += 1
    
    # Print comparison
    print(f"🔵 BASE: {base_response[:70]}{'...' if len(base_response) > 70 else ''}")
    print(f"   Singapore content: {'✅' if base_has_sg else '❌'}")
    
    print(f"🟢 FINE-TUNED: {ft_response[:70]}{'...' if len(ft_response) > 70 else ''}")
    print(f"   Singapore content: {'✅' if ft_has_sg else '❌'}")
    
    # Quick verdict
    if ft_has_sg and not base_has_sg:
        print(f"   🎯 VERDICT: ✅ FINE-TUNED BETTER")
    elif base_has_sg and not ft_has_sg:
        print(f"   🎯 VERDICT: ❌ BASE BETTER")
    elif ft_has_sg and base_has_sg:
        print(f"   🎯 VERDICT: 🤔 BOTH HAVE SINGAPORE CONTENT")
    else:
        print(f"   🎯 VERDICT: ⚠️ NEITHER HAS SINGAPORE CONTENT")

# Final results
print("\n" + "=" * 80)
print("🎯 QUICK COMPARISON RESULTS")
print("=" * 80)

base_rate = (base_singapore_count / len(test_questions)) * 100
ft_rate = (ft_singapore_count / len(test_questions)) * 100

print(f"📊 SINGAPORE CONTENT RATE:")
print(f"   🔵 Base Model: {base_singapore_count}/{len(test_questions)} ({base_rate:.1f}%)")
print(f"   🟢 Fine-tuned: {ft_singapore_count}/{len(test_questions)} ({ft_rate:.1f}%)")

if ft_rate > base_rate + 20:
    print(f"\n🎉 EXCELLENT: Fine-tuning significantly improved Singapore knowledge!")
    print(f"   ✅ {ft_rate - base_rate:.1f}% improvement in Singapore content")
elif ft_rate > base_rate:
    print(f"\n✅ GOOD: Fine-tuning improved Singapore knowledge")
    print(f"   📈 {ft_rate - base_rate:.1f}% improvement in Singapore content")
elif ft_rate == base_rate:
    print(f"\n⚠️ NO CHANGE: Fine-tuning didn't improve Singapore knowledge")
else:
    print(f"\n❌ WORSE: Fine-tuning actually reduced Singapore knowledge")
    print(f"   📉 {base_rate - ft_rate:.1f}% decrease in Singapore content")

print(f"\n💡 RECOMMENDATION:")
if ft_rate >= 80:
    print("   🎉 Excellent! Your fine-tuning worked very well!")
elif ft_rate >= 60:
    print("   ✅ Good results! Fine-tuning is working.")
elif ft_rate >= 40:
    print("   ⚠️ Moderate results. Consider more training.")
else:
    print("   ❌ Poor results. Need to troubleshoot training approach.")

print("\n" + "=" * 80)
print("✅ QUICK COMPARISON COMPLETED!")
print("=" * 80)
