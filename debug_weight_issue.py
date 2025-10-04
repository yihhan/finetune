"""
Debug Weight Issue - Why don't weight changes affect output?

This is the core problem: manual weight changes have zero effect.
Let's test different theories about why this happens.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test_weight_change_theories():
    """Test different theories about why weight changes don't work"""
    
    print("🔧 DEBUGGING WEIGHT CHANGE ISSUE")
    print("=" * 50)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    
    test_input = "What is Singapore?"
    inputs = tokenizer(test_input, return_tensors="pt")
    
    print("🧪 THEORY 1: Generation parameters too deterministic")
    print("-" * 30)
    
    # Test different generation strategies
    generation_configs = [
        {"max_new_tokens": 10, "num_beams": 1, "do_sample": False, "name": "Greedy"},
        {"max_new_tokens": 10, "num_beams": 2, "do_sample": False, "name": "Beam-2"},
        {"max_new_tokens": 10, "do_sample": True, "temperature": 0.7, "name": "Sampling"},
        {"max_new_tokens": 10, "do_sample": True, "temperature": 1.5, "name": "High-temp"},
    ]
    
    for config in generation_configs:
        name = config.pop("name")
        with torch.no_grad():
            outputs = model.generate(**inputs, **config)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   {name}: '{response}'")
    
    print("\n🧪 THEORY 2: Weight change verification")
    print("-" * 30)
    
    # Get original weight value
    target_param = None
    target_name = None
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) > 1:
            target_param = param
            target_name = name
            break
    
    if target_param is not None:
        original_weight = target_param.data.clone()
        print(f"Original weight sample: {original_weight.flatten()[:5]}")
        
        # Modify weight
        target_param.data += torch.randn_like(target_param.data) * 0.1  # Larger change
        modified_weight = target_param.data.clone()
        print(f"Modified weight sample: {modified_weight.flatten()[:5]}")
        
        # Verify change happened
        weight_diff = torch.abs(original_weight - modified_weight).mean()
        print(f"Weight difference: {weight_diff:.6f}")
        
        if weight_diff > 0.001:
            print("✅ Weight change verified")
        else:
            print("❌ Weight change too small")
    
    print("\n🧪 THEORY 3: Model mode and gradients")
    print("-" * 30)
    
    print(f"Model training mode: {model.training}")
    print(f"Requires grad: {target_param.requires_grad if target_param is not None else 'N/A'}")
    
    # Force training mode
    model.train()
    print(f"After model.train(): {model.training}")
    
    # Test generation in training mode
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1.0)
    response_train = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Training mode response: '{response_train}'")
    
    # Back to eval mode
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1.0)
    response_eval = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Eval mode response: '{response_eval}'")
    
    print("\n🧪 THEORY 4: Massive weight change")
    print("-" * 30)
    
    # Make a HUGE change that should definitely affect output
    if target_param is not None:
        # Zero out the entire parameter
        original_data = target_param.data.clone()
        target_param.data.zero_()
        
        print("Zeroed out entire weight matrix...")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1.0)
        response_zero = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Zero weights response: '{response_zero}'")
        
        # Restore original weights
        target_param.data = original_data
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1.0)
        response_restored = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Restored weights response: '{response_restored}'")
        
        if response_zero != response_restored:
            print("✅ MASSIVE change affected output!")
            return True
        else:
            print("❌ Even zeroing weights had no effect!")
            return False
    
    return False

def test_simple_forward_pass():
    """Test if we can affect the forward pass directly"""
    
    print("\n🧪 THEORY 5: Direct forward pass test")
    print("-" * 30)
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    
    test_input = "What is Singapore?"
    inputs = tokenizer(test_input, return_tensors="pt")
    
    # Get original logits
    with torch.no_grad():
        original_outputs = model(**inputs)
        original_logits = original_outputs.logits
    
    print(f"Original logits shape: {original_logits.shape}")
    print(f"Original logits sample: {original_logits[0, 0, :5]}")
    
    # Modify a weight
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) > 1:
            param.data += torch.randn_like(param.data) * 0.1
            print(f"Modified: {name}")
            break
    
    # Get new logits
    with torch.no_grad():
        new_outputs = model(**inputs)
        new_logits = new_outputs.logits
    
    print(f"New logits sample: {new_logits[0, 0, :5]}")
    
    # Check if logits changed
    logits_diff = torch.abs(original_logits - new_logits).mean()
    print(f"Logits difference: {logits_diff:.6f}")
    
    if logits_diff > 0.001:
        print("✅ Weight change affected forward pass!")
        return True
    else:
        print("❌ Weight change did NOT affect forward pass!")
        return False

if __name__ == "__main__":
    weight_affects_generation = test_weight_change_theories()
    weight_affects_forward = test_simple_forward_pass()
    
    print("\n" + "=" * 50)
    print("🎯 DIAGNOSTIC SUMMARY:")
    print(f"   Weight changes affect generation: {weight_affects_generation}")
    print(f"   Weight changes affect forward pass: {weight_affects_forward}")
    
    if not weight_affects_forward:
        print("\n❌ FUNDAMENTAL ISSUE: Weights don't affect computation at all!")
        print("❌ This suggests model loading or architecture problem")
    elif not weight_affects_generation:
        print("\n⚠️ GENERATION ISSUE: Forward pass works but generation doesn't")
        print("⚠️ This suggests generation caching or determinism problem")
    else:
        print("\n✅ WEIGHTS WORK: Problem must be elsewhere")
