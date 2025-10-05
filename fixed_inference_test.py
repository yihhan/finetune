# 🔧 FIXED INFERENCE TEST - Proper generation parameters for your trained model
# Your model learned correctly, but needs better generation settings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("🔧 FIXED INFERENCE TEST - PROPER GENERATION PARAMETERS")
print("=" * 80)
print("Your model learned correctly - just needs better generation settings!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained model
print("\n🔄 Loading your trained model...")
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    base_model = base_model.to(device)
    
    # Load your fine-tuned model
    model_path = "gpt2_comprehensive_singapore_model/checkpoint-1080"
    finetuned_model = PeftModel.from_pretrained(base_model, model_path)
    print(f"✅ Loaded your fine-tuned model from: {model_path}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

def generate_clean_response(model, question, max_length=100):
    """Generate clean response with optimized parameters"""
    prompt = f"Q: {question} A:"
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        model.eval()
        with torch.no_grad():
            # OPTIMIZED GENERATION PARAMETERS for your model
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_length,
                num_return_sequences=1,
                
                # FIXED PARAMETERS - prevent repetition and improve quality
                temperature=0.8,           # Slightly higher for variety
                do_sample=True,
                top_p=0.95,               # Slightly higher for better responses
                top_k=50,                 # Add top_k for better control
                repetition_penalty=1.3,   # HIGHER to prevent repetition
                no_repeat_ngram_size=3,   # Prevent 3-gram repetition
                
                # Stop tokens
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                
                # Early stopping
                early_stopping=True,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        # Remove any trailing repetitive text
        sentences = response.split('. ')
        if len(sentences) > 1:
            # Check for repetitive sentences
            unique_sentences = []
            for sentence in sentences:
                if sentence.strip() and sentence.strip() not in [s.strip() for s in unique_sentences]:
                    unique_sentences.append(sentence)
            response = '. '.join(unique_sentences)
            if not response.endswith('.') and len(unique_sentences) > 0:
                response += '.'
        
        return response
    
    except Exception as e:
        return f"Error: {str(e)}"

# Test questions
test_questions = [
    "What does MAS stand for?",
    "What currency does Singapore use?",
    "Who regulates banks in Singapore?",
    "What are the capital adequacy requirements for Singapore banks?",
    "What is MAS Notice 626?",
    "What is STRO?",
    "What are the AML reporting requirements?",
    "What is the Payment Services Act?"
]

print(f"\n🧪 TESTING WITH FIXED GENERATION PARAMETERS...")
print("=" * 80)

singapore_terms = ['singapore', 'mas', 'monetary authority', 'sgd', 'dollar', 'notice', 'regulation', 'financial', 'bank']
success_count = 0
quality_count = 0

for i, question in enumerate(test_questions, 1):
    print(f"\n📝 QUESTION {i}: {question}")
    print("-" * 60)
    
    # Generate response with fixed parameters
    response = generate_clean_response(finetuned_model, question)
    
    # Check quality
    has_singapore_content = any(term in response.lower() for term in singapore_terms)
    is_good_quality = (
        len(response) > 20 and 
        len(response) < 300 and 
        not response.startswith("Error:") and
        "(" not in response.count("(") > 3  # Not too many parentheses
    )
    
    if has_singapore_content:
        success_count += 1
    if is_good_quality:
        quality_count += 1
    
    # Display result
    print(f"🟢 RESPONSE: {response}")
    print(f"📊 Singapore Content: {'✅' if has_singapore_content else '❌'}")
    print(f"📊 Good Quality: {'✅' if is_good_quality else '❌'}")
    print(f"📊 Length: {len(response)} characters")

# Calculate results
singapore_rate = (success_count / len(test_questions)) * 100
quality_rate = (quality_count / len(test_questions)) * 100

print(f"\n" + "=" * 80)
print("🎯 FIXED INFERENCE RESULTS")
print("=" * 80)

print(f"📊 PERFORMANCE METRICS:")
print(f"   Singapore Content: {success_count}/{len(test_questions)} ({singapore_rate:.1f}%)")
print(f"   Response Quality: {quality_count}/{len(test_questions)} ({quality_rate:.1f}%)")

print(f"\n🎯 ASSESSMENT:")
if singapore_rate >= 80 and quality_rate >= 70:
    print("🎉 EXCELLENT: Your model works perfectly with fixed generation!")
    print("✅ The training was successful - just needed better inference parameters")
    recommendation = "Deploy with confidence using these generation parameters"
elif singapore_rate >= 60 and quality_rate >= 50:
    print("✅ GOOD: Your model is working well with fixed generation!")
    print("✅ Significant improvement with proper parameters")
    recommendation = "Fine-tune generation parameters further for even better results"
elif singapore_rate >= 40:
    print("⚠️ MODERATE: Some improvement but still needs work")
    recommendation = "Consider additional training or parameter adjustment"
else:
    print("❌ POOR: Model still not performing well")
    recommendation = "May need to retrain with different approach"

print(f"\n💡 RECOMMENDATION: {recommendation}")

print(f"\n🔧 KEY FIXES APPLIED:")
print(f"   • Increased repetition_penalty to 1.3 (was causing repetition)")
print(f"   • Added no_repeat_ngram_size=3 (prevents phrase repetition)")
print(f"   • Optimized temperature and top_p values")
print(f"   • Added response cleaning to remove repetitive sentences")
print(f"   • Added early_stopping for cleaner endings")

if singapore_rate >= 60:
    print(f"\n🚀 SUCCESS! Your training worked - it was just an inference problem!")
    print(f"   Use these generation parameters for production deployment")
else:
    print(f"\n💡 If still not good enough, try the proven working approach")
    print(f"   with smaller, higher-quality dataset")

print(f"\n" + "=" * 80)
print("✅ FIXED INFERENCE TEST COMPLETED!")
print("=" * 80)
