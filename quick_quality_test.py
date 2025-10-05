#!/usr/bin/env python3
"""
Quick test to check if the full fine-tuning produced quality responses
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test_quality_responses():
    """Test if the fine-tuned model produces quality Singapore responses"""
    
    print("🧪 QUICK QUALITY TEST - Full Fine-Tuned Model")
    print("=" * 60)
    
    # Load models
    model_name = "google/flan-t5-small"
    model_path = "quality_finetuned_model"
    
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Load fine-tuned model
    try:
        finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        print("✅ Fine-tuned model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading fine-tuned model: {e}")
        return
    
    # Test questions
    test_questions = [
        "What does MAS stand for?",
        "What currency does Singapore use?",
        "Who regulates banks in Singapore?"
    ]
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    finetuned_model.to(device)
    
    quality_count = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        
        # Prepare input
        input_text = f"Answer this Singapore financial regulation question: {question}"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Base model response
        base_model.eval()
        with torch.no_grad():
            base_outputs = base_model.generate(
                **inputs, 
                max_new_tokens=50, 
                num_beams=3,
                do_sample=False
            )
        base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        
        # Fine-tuned model response
        finetuned_model.eval()
        with torch.no_grad():
            ft_outputs = finetuned_model.generate(
                **inputs, 
                max_new_tokens=50, 
                num_beams=3,
                do_sample=False
            )
        ft_response = tokenizer.decode(ft_outputs[0], skip_special_tokens=True)
        
        print(f"   Base:        '{base_response}'")
        print(f"   Fine-tuned:  '{ft_response}'")
        
        # Check quality
        singapore_keywords = [
            'monetary authority of singapore', 'mas', 'singapore dollar', 'sgd',
            'singapore', 'regulates', 'banks'
        ]
        
        if any(keyword in ft_response.lower() for keyword in singapore_keywords):
            print("   ✅ QUALITY: Contains Singapore financial content!")
            quality_count += 1
        else:
            print("   ❌ POOR: No Singapore financial content detected")
    
    # Final assessment
    quality_rate = (quality_count / len(test_questions)) * 100
    print(f"\n" + "=" * 60)
    print(f"🎯 QUALITY ASSESSMENT: {quality_count}/{len(test_questions)} ({quality_rate:.1f}%)")
    
    if quality_rate >= 80:
        print("🎉 EXCELLENT: High-quality Singapore responses achieved!")
        print("✅ Full fine-tuning SUCCESS!")
    elif quality_rate >= 50:
        print("✅ GOOD: Some quality improvement detected")
        print("⚠️ May need more training or better data")
    else:
        print("❌ POOR: Still low quality responses")
        print("❌ Full fine-tuning approach needs adjustment")
    
    # Check for training issues
    print(f"\n🔍 TRAINING DIAGNOSTICS:")
    print("⚠️ Training loss = 0.000000 suggests possible overfitting")
    print("⚠️ Validation loss = nan indicates data/loss computation issues")
    print("💡 Consider: Lower LR, fewer epochs, or data preprocessing fixes")

if __name__ == "__main__":
    test_quality_responses()
