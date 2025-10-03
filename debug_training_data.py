"""
Debug Training Data Quality

Check if our training data actually contains Singapore-specific information
that should make responses different from the base model.
"""

import json
from pathlib import Path

def analyze_training_data():
    """Analyze the training data to see if it contains Singapore-specific info"""
    
    print("🔍 ANALYZING TRAINING DATA QUALITY")
    print("="*60)
    
    # Load training data
    data_path = "processed_data/enhanced_training_data.json"
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Training data not found: {data_path}")
        return
    
    print(f"📊 Total training samples: {len(data)}")
    
    # Analyze content
    singapore_keywords = ['singapore', 'mas', 'sgd', 'monetary authority', 'mas requirements']
    specific_numbers = ['million', 'sgd', '$', 'percent', '%', 'days', 'hours', 'monthly', 'annually']
    
    singapore_count = 0
    specific_count = 0
    
    print(f"\n📝 Sample training data:")
    print("-" * 60)
    
    for i, item in enumerate(data[:5]):  # Show first 5 samples
        input_text = item['input']
        output_text = item['output']
        
        print(f"\n{i+1}. INPUT: {input_text}")
        print(f"   OUTPUT: {output_text[:200]}...")
        
        # Check for Singapore-specific content
        output_lower = output_text.lower()
        has_singapore = any(keyword in output_lower for keyword in singapore_keywords)
        has_specific = any(keyword in output_lower for keyword in specific_numbers)
        
        if has_singapore:
            singapore_count += 1
            print(f"   ✅ Contains Singapore-specific content")
        else:
            print(f"   ⚠️ Generic content (no Singapore keywords)")
            
        if has_specific:
            specific_count += 1
            print(f"   ✅ Contains specific numbers/details")
        else:
            print(f"   ⚠️ Vague content (no specific details)")
        
        print("-" * 60)
    
    # Overall analysis
    print(f"\n📊 CONTENT ANALYSIS:")
    print(f"  Singapore-specific samples: {singapore_count}/{len(data)} ({100*singapore_count/len(data):.1f}%)")
    print(f"  Samples with specific details: {specific_count}/{len(data)} ({100*specific_count/len(data):.1f}%)")
    
    # Check if training data is actually different from what base model would say
    print(f"\n🎯 KEY QUESTIONS:")
    print(f"  1. Does training data contain Singapore-specific info? {'✅ YES' if singapore_count > len(data)//2 else '❌ NO'}")
    print(f"  2. Does training data have specific details? {'✅ YES' if specific_count > len(data)//2 else '❌ NO'}")
    print(f"  3. Would base model give different answers? {'🤔 MAYBE' if singapore_count > 0 else '❌ UNLIKELY'}")
    
    # Test specific examples
    print(f"\n🧪 TESTING SPECIFIC EXAMPLES:")
    
    test_cases = [
        {
            "question": "What are the capital requirements for banks in Singapore?",
            "expected_keywords": ["sgd", "million", "mas", "singapore"],
            "base_model_likely": "generic capital requirements"
        },
        {
            "question": "How frequently must banks submit returns to MAS?",
            "expected_keywords": ["monthly", "mas", "singapore"],
            "base_model_likely": "quarterly or annually"
        }
    ]
    
    for test in test_cases:
        print(f"\nQ: {test['question']}")
        
        # Find matching training sample
        matching_sample = None
        for item in data:
            if any(word in item['input'].lower() for word in test['question'].lower().split()[:3]):
                matching_sample = item
                break
        
        if matching_sample:
            print(f"Training Answer: {matching_sample['output'][:150]}...")
            
            # Check if it contains expected keywords
            answer_lower = matching_sample['output'].lower()
            found_keywords = [kw for kw in test['expected_keywords'] if kw in answer_lower]
            
            print(f"Singapore Keywords Found: {found_keywords}")
            print(f"Base Model Would Say: {test['base_model_likely']}")
            
            if found_keywords:
                print("✅ Training data IS different from base model")
            else:
                print("❌ Training data might be too generic")
        else:
            print("❌ No matching training sample found")
    
    print(f"\n🎯 CONCLUSION:")
    if singapore_count > len(data)//2 and specific_count > len(data)//2:
        print("✅ Training data looks good - should produce different responses")
        print("❓ Problem might be with LoRA parameters or training process")
    else:
        print("❌ Training data is too generic - won't produce different responses")
        print("💡 Need more Singapore-specific, detailed training data")

if __name__ == "__main__":
    analyze_training_data()
