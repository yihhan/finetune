# 🔍 TRAINING DIAGNOSTIC - Find out why fine-tuning isn't working
# Comprehensive analysis of training data, model, and parameters

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

print("🔍 TRAINING DIAGNOSTIC - FINDING THE PROBLEM")
print("=" * 80)

# 1. Check training data quality
print("\n📊 STEP 1: ANALYZING TRAINING DATA QUALITY")
print("-" * 50)

try:
    # Load the training data
    with open('gpt2_maximum_training_data.json', 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    print(f"✅ Loaded {len(training_data)} training examples")
    
    # Analyze data format
    print(f"\n📝 TRAINING DATA ANALYSIS:")
    
    valid_qa_count = 0
    empty_answers = 0
    short_answers = 0
    long_answers = 0
    singapore_content = 0
    
    for i, example in enumerate(training_data[:10]):  # Check first 10
        print(f"\n{i+1}. {example[:100]}{'...' if len(example) > 100 else ''}")
        
        if isinstance(example, str) and "Q: " in example and " A: " in example:
            valid_qa_count += 1
            parts = example.split(" A: ", 1)
            if len(parts) == 2:
                question = parts[0].replace("Q: ", "").strip()
                answer = parts[1].strip()
                
                if not answer:
                    empty_answers += 1
                elif len(answer) < 20:
                    short_answers += 1
                elif len(answer) > 200:
                    long_answers += 1
                
                if any(term in example.lower() for term in ['singapore', 'mas', 'monetary authority']):
                    singapore_content += 1
    
    print(f"\n📊 DATA QUALITY METRICS:")
    print(f"   Valid Q&A format: {valid_qa_count}/10")
    print(f"   Empty answers: {empty_answers}/10")
    print(f"   Short answers (<20 chars): {short_answers}/10")
    print(f"   Long answers (>200 chars): {long_answers}/10")
    print(f"   Singapore content: {singapore_content}/10")
    
    if valid_qa_count < 8:
        print("   ❌ PROBLEM: Poor data format - many examples not in Q: A: format")
    if empty_answers > 2:
        print("   ❌ PROBLEM: Too many empty answers")
    if short_answers > 5:
        print("   ❌ PROBLEM: Answers too short - not enough content to learn")
    if singapore_content < 7:
        print("   ❌ PROBLEM: Not enough Singapore financial content")
        
except Exception as e:
    print(f"❌ Error loading training data: {e}")

# 2. Check model and checkpoint
print(f"\n🤖 STEP 2: ANALYZING MODEL AND CHECKPOINT")
print("-" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # Check if model directory exists
    model_dir = "gpt2_comprehensive_singapore_model"
    if os.path.exists(model_dir):
        checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
        print(f"✅ Found {len(checkpoints)} checkpoints: {checkpoints}")
        
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            checkpoint_path = f"{model_dir}/{latest_checkpoint}"
            print(f"📁 Latest checkpoint: {checkpoint_path}")
            
            # Check checkpoint contents
            checkpoint_files = os.listdir(checkpoint_path)
            print(f"📄 Checkpoint files: {checkpoint_files}")
            
            required_files = ['adapter_config.json', 'adapter_model.safetensors']
            missing_files = [f for f in required_files if f not in checkpoint_files]
            if missing_files:
                print(f"❌ PROBLEM: Missing required files: {missing_files}")
            else:
                print(f"✅ All required PEFT files present")
                
                # Try to load the model
                try:
                    tokenizer = AutoTokenizer.from_pretrained("gpt2")
                    tokenizer.pad_token = tokenizer.eos_token
                    
                    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
                    base_model = base_model.to(device)
                    
                    finetuned_model = PeftModel.from_pretrained(base_model, checkpoint_path)
                    print(f"✅ Successfully loaded fine-tuned model")
                    
                    # Quick test
                    test_prompt = "Q: What does MAS stand for? A:"
                    inputs = tokenizer(test_prompt, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = finetuned_model.generate(
                            **inputs,
                            max_length=inputs['input_ids'].shape[1] + 50,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response.replace(test_prompt, "").strip()
                    
                    print(f"🧪 Quick test response: '{response}'")
                    
                    if 'singapore' in response.lower() or 'monetary authority' in response.lower():
                        print(f"✅ Model shows Singapore knowledge")
                    else:
                        print(f"❌ PROBLEM: Model doesn't show Singapore knowledge")
                        
                except Exception as e:
                    print(f"❌ PROBLEM: Cannot load or test model: {e}")
        else:
            print(f"❌ PROBLEM: No checkpoints found in model directory")
    else:
        print(f"❌ PROBLEM: Model directory doesn't exist")
        
except Exception as e:
    print(f"❌ Error analyzing model: {e}")

# 3. Check training parameters (if available)
print(f"\n⚙️ STEP 3: ANALYZING TRAINING PARAMETERS")
print("-" * 50)

try:
    # Look for training args
    if os.path.exists(f"{model_dir}/{latest_checkpoint}/training_args.bin"):
        print(f"✅ Found training arguments file")
        # Could load and analyze training args here
    
    # Check adapter config
    if os.path.exists(f"{model_dir}/{latest_checkpoint}/adapter_config.json"):
        with open(f"{model_dir}/{latest_checkpoint}/adapter_config.json", 'r') as f:
            adapter_config = json.load(f)
        
        print(f"📊 LORA CONFIGURATION:")
        print(f"   r (rank): {adapter_config.get('r', 'unknown')}")
        print(f"   lora_alpha: {adapter_config.get('lora_alpha', 'unknown')}")
        print(f"   lora_dropout: {adapter_config.get('lora_dropout', 'unknown')}")
        print(f"   target_modules: {adapter_config.get('target_modules', 'unknown')}")
        
        # Check if parameters are reasonable
        r_value = adapter_config.get('r', 0)
        if r_value < 4:
            print(f"   ⚠️ WARNING: r={r_value} might be too low")
        elif r_value > 64:
            print(f"   ⚠️ WARNING: r={r_value} might be too high")
        else:
            print(f"   ✅ r={r_value} looks reasonable")
            
except Exception as e:
    print(f"❌ Error analyzing training parameters: {e}")

# 4. Recommendations
print(f"\n💡 STEP 4: DIAGNOSTIC RECOMMENDATIONS")
print("-" * 50)

print(f"🔧 POTENTIAL ISSUES AND FIXES:")

print(f"\n1. TRAINING DATA ISSUES:")
print(f"   • If data format is wrong: Reformat to proper 'Q: ... A: ...' structure")
print(f"   • If answers too short: Generate longer, more detailed answers")
print(f"   • If not enough Singapore content: Add more MAS-specific examples")

print(f"\n2. MODEL LOADING ISSUES:")
print(f"   • If model won't load: Check PEFT installation and file permissions")
print(f"   • If no checkpoints: Training may have failed - check logs")

print(f"\n3. TRAINING PARAMETER ISSUES:")
print(f"   • Learning rate too high: Try 1e-5 instead of 5e-4")
print(f"   • Not enough epochs: Try 5-10 epochs instead of 3")
print(f"   • LoRA rank too low: Try r=16 instead of r=8")
print(f"   • Wrong target modules: Ensure ['c_attn', 'c_proj'] for GPT-2")

print(f"\n4. GENERATION ISSUES:")
print(f"   • Poor responses: Adjust temperature, top_p, repetition_penalty")
print(f"   • Repetitive text: Increase repetition_penalty to 1.2-1.5")
print(f"   • Off-topic responses: Model may not have learned properly")

print(f"\n🎯 NEXT STEPS:")
print(f"   1. Run this diagnostic to identify specific issues")
print(f"   2. Fix identified problems (data, parameters, etc.)")
print(f"   3. Retrain with corrected approach")
print(f"   4. Test with ground truth comparison")

print(f"\n" + "=" * 80)
print("✅ DIAGNOSTIC COMPLETED!")
print("Share the results and I'll help you fix the specific issues found.")
print("=" * 80)
