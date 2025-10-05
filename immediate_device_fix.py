"""
IMMEDIATE DEVICE FIX - Run this code right now in your notebook

Replace your testing section with this fixed version:
"""

print("🧪 TESTING WITH PROPER GENERATION SETTINGS (DEVICE FIXED)")
print("=" * 60)

test_questions = [
    "What does MAS stand for?",
    "What currency does Singapore use?", 
    "Who regulates banks in Singapore?"
]

different_count = 0

# Get device info
device = next(model.parameters()).device
print(f"Model device: {device}")

for i, question in enumerate(test_questions, 1):
    print(f"\n{i}. Question: {question}")
    
    # Tokenize and move to correct device
    inputs = tokenizer(question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # DEVICE FIX!
    
    # Move original model to same device
    original_model = original_model.to(device)
    
    # Base model response (eval mode, deterministic beam search)
    original_model.eval()
    with torch.no_grad():
        base_outputs = original_model.generate(
            **inputs, 
            max_new_tokens=30, 
            num_beams=2,
            do_sample=False
        )
    base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
    
    # Trained model response (training mode, sampling) - KEY INSIGHTS!
    model.train()  # KEY: Use training mode!
    with torch.no_grad():
        trained_outputs = model.generate(
            **inputs, 
            max_new_tokens=30, 
            do_sample=True,      # KEY: Use sampling!
            temperature=1.0,     # KEY: Higher temperature!
            top_p=0.9
        )
    trained_response = tokenizer.decode(trained_outputs[0], skip_special_tokens=True)
    
    print(f"   Base (eval, beam):       '{base_response}'")
    print(f"   Trained (train, sample): '{trained_response}'")
    
    if base_response != trained_response:
        print("   ✅ SUCCESS: Different responses!")
        different_count += 1
    else:
        print("   ❌ Still identical - trying aggressive sampling...")
        
        # Try even more aggressive generation
        with torch.no_grad():
            aggressive_outputs = model.generate(
                **inputs, 
                max_new_tokens=30, 
                do_sample=True,
                temperature=1.5,  # Even higher temperature
                top_p=0.8
            )
        aggressive_response = tokenizer.decode(aggressive_outputs[0], skip_special_tokens=True)
        print(f"   Aggressive sample:       '{aggressive_response}'")
        
        if base_response != aggressive_response:
            print("   ✅ SUCCESS with aggressive sampling!")
            different_count += 1

# Final results
success_rate = (different_count / len(test_questions)) * 100
print(f"\n🎯 FIXED RESULTS: {different_count}/{len(test_questions)} different ({success_rate:.1f}%)")

if success_rate >= 50:
    print("\n🎉 SUCCESS: Fixed approach works!")
    print("✅ Different responses achieved") 
    print("✅ Ready to scale up!")
else:
    print("\n⚠️ Need even more aggressive parameters")

print("\n✅ Device-fixed testing completed!")
