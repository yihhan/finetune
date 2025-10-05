# 🔄 BASE vs FINE-TUNED MODEL COMPARISON
# Compare responses side-by-side to see if fine-tuning actually worked

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time
import re

print("🔄 BASE vs FINE-TUNED MODEL COMPARISON")
print("=" * 80)
print("Side-by-side comparison to see if fine-tuning actually improved responses")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
print("\n🔄 Loading both base and fine-tuned models...")
try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model (clean GPT-2)
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    base_model = base_model.to(device)
    print(f"✅ Loaded BASE GPT-2 model")
    
    # Load another copy for fine-tuned version
    base_for_peft = AutoModelForCausalLM.from_pretrained("gpt2")
    base_for_peft = base_for_peft.to(device)
    
    # Load fine-tuned model
    model_path = "gpt2_comprehensive_singapore_model/checkpoint-1080"
    finetuned_model = PeftModel.from_pretrained(base_for_peft, model_path)
    print(f"✅ Loaded FINE-TUNED model from: {model_path}")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# Test questions - mix of training and new questions
test_questions = [
    "What does MAS stand for?",
    "What currency does Singapore use?", 
    "Who regulates banks in Singapore?",
    "What are the capital adequacy requirements for Singapore banks?",
    "What is MAS Notice 626?",
    "What is STRO?",
    "What are AML requirements in Singapore?",
    "How does MAS regulate fintech companies?",
    "What is the Payment Services Act?",
    "What are cybersecurity requirements for banks?"
]

def generate_response(model, question, max_new_tokens=128):
    """Generate response from model"""
    prompt = f"Q: {question} A:"
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            end_time = time.time()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response, end_time - start_time
    
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Singapore financial terms for evaluation
singapore_terms = [
    'mas', 'singapore', 'monetary authority', 'sgd', 'dollar', 'notice', 
    'regulation', 'financial', 'bank', 'capital', 'aml', 'compliance',
    'requirement', 'guideline', 'institution', 'risk', 'management'
]

def evaluate_response(response):
    """Evaluate response quality"""
    if not response or response.startswith("Error:"):
        return {"singapore_content": False, "length": 0, "quality": "error"}
    
    # Check Singapore content
    singapore_content = any(term in response.lower() for term in singapore_terms)
    
    # Check response length and quality
    length = len(response.strip())
    
    if length < 10:
        quality = "too_short"
    elif response.startswith("Q:") or "I don't know" in response.lower():
        quality = "poor"
    elif singapore_content and length > 30:
        quality = "good"
    elif length > 20:
        quality = "moderate"
    else:
        quality = "poor"
    
    return {
        "singapore_content": singapore_content,
        "length": length,
        "quality": quality
    }

# Run comparison test
print(f"\n🚀 RUNNING SIDE-BY-SIDE COMPARISON...")
print("=" * 80)

base_stats = {"singapore": 0, "good_quality": 0, "total_length": 0}
finetuned_stats = {"singapore": 0, "good_quality": 0, "total_length": 0}

for i, question in enumerate(test_questions, 1):
    print(f"\n📝 QUESTION {i}: {question}")
    print("-" * 60)
    
    # Generate responses from both models
    base_response, base_time = generate_response(base_model, question)
    finetuned_response, ft_time = generate_response(finetuned_model, question)
    
    # Evaluate responses
    base_eval = evaluate_response(base_response)
    ft_eval = evaluate_response(finetuned_response)
    
    # Update statistics
    if base_eval["singapore_content"]:
        base_stats["singapore"] += 1
    if base_eval["quality"] in ["good", "moderate"]:
        base_stats["good_quality"] += 1
    base_stats["total_length"] += base_eval["length"]
    
    if ft_eval["singapore_content"]:
        finetuned_stats["singapore"] += 1
    if ft_eval["quality"] in ["good", "moderate"]:
        finetuned_stats["good_quality"] += 1
    finetuned_stats["total_length"] += ft_eval["length"]
    
    # Print comparison
    print(f"🔵 BASE MODEL:")
    print(f"   Response: {base_response[:100]}{'...' if len(base_response) > 100 else ''}")
    print(f"   Quality: {base_eval['quality']} | Singapore: {'✅' if base_eval['singapore_content'] else '❌'} | Length: {base_eval['length']} | Time: {base_time:.2f}s")
    
    print(f"\n🟢 FINE-TUNED MODEL:")
    print(f"   Response: {finetuned_response[:100]}{'...' if len(finetuned_response) > 100 else ''}")
    print(f"   Quality: {ft_eval['quality']} | Singapore: {'✅' if ft_eval['singapore_content'] else '❌'} | Length: {ft_eval['length']} | Time: {ft_time:.2f}s")
    
    # Comparison verdict
    if ft_eval["quality"] == "good" and base_eval["quality"] in ["poor", "moderate"]:
        verdict = "🎉 FINE-TUNED BETTER"
    elif ft_eval["singapore_content"] and not base_eval["singapore_content"]:
        verdict = "✅ FINE-TUNED BETTER (Singapore content)"
    elif ft_eval["quality"] == base_eval["quality"] and ft_eval["singapore_content"] == base_eval["singapore_content"]:
        verdict = "⚠️ NO SIGNIFICANT DIFFERENCE"
    elif base_eval["quality"] == "good" and ft_eval["quality"] in ["poor", "moderate"]:
        verdict = "❌ BASE MODEL BETTER"
    else:
        verdict = "🤔 MIXED RESULTS"
    
    print(f"\n   🎯 VERDICT: {verdict}")

# Calculate final statistics
num_questions = len(test_questions)
base_singapore_rate = (base_stats["singapore"] / num_questions) * 100
ft_singapore_rate = (finetuned_stats["singapore"] / num_questions) * 100
base_quality_rate = (base_stats["good_quality"] / num_questions) * 100
ft_quality_rate = (finetuned_stats["good_quality"] / num_questions) * 100
base_avg_length = base_stats["total_length"] / num_questions
ft_avg_length = finetuned_stats["total_length"] / num_questions

# Print comprehensive comparison
print("\n" + "=" * 80)
print("🎯 COMPREHENSIVE COMPARISON RESULTS")
print("=" * 80)

print(f"📊 SINGAPORE CONTENT:")
print(f"   🔵 Base Model: {base_stats['singapore']}/{num_questions} ({base_singapore_rate:.1f}%)")
print(f"   🟢 Fine-tuned: {finetuned_stats['singapore']}/{num_questions} ({ft_singapore_rate:.1f}%)")
if ft_singapore_rate > base_singapore_rate:
    print(f"   ✅ Fine-tuned is {ft_singapore_rate - base_singapore_rate:.1f}% better at Singapore content")
elif ft_singapore_rate < base_singapore_rate:
    print(f"   ❌ Fine-tuned is {base_singapore_rate - ft_singapore_rate:.1f}% worse at Singapore content")
else:
    print(f"   ⚠️ No difference in Singapore content")

print(f"\n📊 RESPONSE QUALITY:")
print(f"   🔵 Base Model: {base_stats['good_quality']}/{num_questions} ({base_quality_rate:.1f}%)")
print(f"   🟢 Fine-tuned: {finetuned_stats['good_quality']}/{num_questions} ({ft_quality_rate:.1f}%)")
if ft_quality_rate > base_quality_rate:
    print(f"   ✅ Fine-tuned is {ft_quality_rate - base_quality_rate:.1f}% better quality")
elif ft_quality_rate < base_quality_rate:
    print(f"   ❌ Fine-tuned is {base_quality_rate - ft_quality_rate:.1f}% worse quality")
else:
    print(f"   ⚠️ No difference in quality")

print(f"\n📊 RESPONSE LENGTH:")
print(f"   🔵 Base Model: {base_avg_length:.1f} characters average")
print(f"   🟢 Fine-tuned: {ft_avg_length:.1f} characters average")

print(f"\n🎯 OVERALL ASSESSMENT:")
improvement_score = 0
if ft_singapore_rate > base_singapore_rate + 10:
    improvement_score += 2
elif ft_singapore_rate > base_singapore_rate:
    improvement_score += 1

if ft_quality_rate > base_quality_rate + 10:
    improvement_score += 2
elif ft_quality_rate > base_quality_rate:
    improvement_score += 1

if improvement_score >= 3:
    print("   🎉 EXCELLENT: Fine-tuning significantly improved the model!")
    print("   ✅ Your fine-tuning was successful!")
elif improvement_score >= 2:
    print("   ✅ GOOD: Fine-tuning improved the model moderately")
    print("   💡 Consider more training epochs for better results")
elif improvement_score >= 1:
    print("   ⚠️ MINIMAL: Fine-tuning showed slight improvement")
    print("   💡 May need more training data or different parameters")
else:
    print("   ❌ NO IMPROVEMENT: Fine-tuning did not improve the model")
    print("   💡 Possible issues:")
    print("      • Training data quality")
    print("      • Learning rate too high/low")
    print("      • Insufficient training epochs")
    print("      • LoRA parameters not optimal")

print(f"\n💡 RECOMMENDATIONS:")
if improvement_score < 2:
    print("   • Try more training epochs (5-10)")
    print("   • Lower learning rate (1e-5 instead of 5e-4)")
    print("   • Increase LoRA rank (r=16 instead of r=8)")
    print("   • Add more diverse training examples")
else:
    print("   • Fine-tuning is working! Consider:")
    print("   • More training epochs for further improvement")
    print("   • Larger dataset for broader knowledge")
    print("   • Deploy for production use")

print("\n" + "=" * 80)
print("✅ COMPARISON COMPLETED!")
print("=" * 80)
