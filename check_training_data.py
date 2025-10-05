# 🔍 CHECK TRAINING DATA QUALITY - Find the real problem
# Analyze the 429-example dataset that's causing broken responses

import json
import re

print("🔍 CHECKING TRAINING DATA QUALITY - FINDING THE REAL PROBLEM")
print("=" * 80)

try:
    # Load the training data
    with open('gpt2_maximum_training_data.json', 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    print(f"✅ Loaded {len(training_data)} training examples")
    
    # Detailed analysis of problematic patterns
    print(f"\n📝 DETAILED TRAINING DATA ANALYSIS:")
    
    problematic_examples = []
    repetitive_phrases = {}
    url_examples = []
    broken_examples = []
    
    for i, example in enumerate(training_data):
        if isinstance(example, str):
            # Check for URLs
            if 'http' in example or 'www.' in example or '.sg' in example:
                url_examples.append((i, example[:100]))
            
            # Check for repetitive phrases
            if 'Implementation should be proportionate' in example:
                if 'Implementation should be proportionate' not in repetitive_phrases:
                    repetitive_phrases['Implementation should be proportionate'] = []
                repetitive_phrases['Implementation should be proportionate'].append(i)
            
            if 'MAS Notice 1015' in example:
                if 'MAS Notice 1015' not in repetitive_phrases:
                    repetitive_phrases['MAS Notice 1015'] = []
                repetitive_phrases['MAS Notice 1015'].append(i)
            
            # Check for broken Q&A format
            if not (example.startswith('Q: ') and ' A: ' in example):
                broken_examples.append((i, example[:100]))
            
            # Check for very long answers (might be corrupted)
            if ' A: ' in example:
                answer_part = example.split(' A: ', 1)[1]
                if len(answer_part) > 500:  # Very long answers
                    problematic_examples.append((i, len(answer_part), example[:150]))
    
    print(f"\n🚨 PROBLEMS FOUND:")
    print(f"   Examples with URLs: {len(url_examples)}")
    print(f"   Broken Q&A format: {len(broken_examples)}")
    print(f"   Very long answers (>500 chars): {len(problematic_examples)}")
    
    print(f"\n📊 REPETITIVE PHRASES:")
    for phrase, indices in repetitive_phrases.items():
        print(f"   '{phrase}': appears in {len(indices)} examples")
        if len(indices) > 50:  # Too many repetitions
            print(f"      ❌ PROBLEM: This phrase appears {len(indices)} times - causing repetition!")
    
    if url_examples:
        print(f"\n🌐 EXAMPLES WITH URLs (first 5):")
        for i, (idx, example) in enumerate(url_examples[:5]):
            print(f"   {idx}: {example}...")
    
    if broken_examples:
        print(f"\n💥 BROKEN FORMAT EXAMPLES (first 5):")
        for i, (idx, example) in enumerate(broken_examples[:5]):
            print(f"   {idx}: {example}...")
    
    if problematic_examples:
        print(f"\n📏 VERY LONG ANSWERS (first 5):")
        for i, (idx, length, example) in enumerate(problematic_examples[:5]):
            print(f"   {idx} ({length} chars): {example}...")
    
    # Sample some random examples to see quality
    print(f"\n📋 RANDOM SAMPLE OF TRAINING DATA (5 examples):")
    import random
    sample_indices = random.sample(range(len(training_data)), min(5, len(training_data)))
    
    for i, idx in enumerate(sample_indices):
        example = training_data[idx]
        print(f"\n{i+1}. Example {idx}:")
        print(f"   {example}")
        
        # Analyze this example
        if ' A: ' in example:
            question = example.split(' A: ')[0].replace('Q: ', '')
            answer = example.split(' A: ')[1]
            print(f"   📝 Q: {question[:50]}{'...' if len(question) > 50 else ''}")
            print(f"   💬 A: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"   📊 Answer length: {len(answer)} chars")
            
            # Check for quality issues
            issues = []
            if len(answer) > 400:
                issues.append("Very long")
            if 'http' in answer or 'www.' in answer:
                issues.append("Contains URLs")
            if answer.count('Implementation should be proportionate') > 0:
                issues.append("Repetitive phrase")
            if len(answer.split('.')) > 10:
                issues.append("Too many sentences")
            
            if issues:
                print(f"   ⚠️ Issues: {', '.join(issues)}")
            else:
                print(f"   ✅ Looks good")

except Exception as e:
    print(f"❌ Error loading training data: {e}")

print(f"\n" + "=" * 80)
print("🎯 DIAGNOSIS:")
print("=" * 80)

print(f"Based on the broken model responses, the training data likely has:")
print(f"1. 🚨 Too many repetitive phrases (causing model to repeat)")
print(f"2. 🌐 URLs and broken links (causing nonsensical URLs in responses)")
print(f"3. 📏 Very long, rambling answers (causing incoherent responses)")
print(f"4. 💥 Poor quality content mixed with good content")

print(f"\n💡 SOLUTION:")
print(f"The 429-example dataset is corrupted/low-quality.")
print(f"We need to go back to a SMALL, HIGH-QUALITY dataset that actually works.")

print(f"\n🎯 RECOMMENDATION:")
print(f"1. Abandon the 429-example dataset")
print(f"2. Use the proven working approach with 10-20 perfect examples")
print(f"3. Train with conservative parameters that we know work")
print(f"4. Scale up only after confirming it works")

print(f"\n" + "=" * 80)
print("✅ TRAINING DATA ANALYSIS COMPLETED!")
print("=" * 80)
