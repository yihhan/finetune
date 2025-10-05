"""
Quick Device Fix - Move inputs to same device as model

The error shows model is on CUDA but inputs are on CPU.
Simple fix: move inputs to model's device.
"""

# Add this to your testing section:

# Get model device
device = next(model.parameters()).device
print(f"Model device: {device}")

# Move inputs to same device as model
inputs = tokenizer(question, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}  # KEY FIX!

# Then proceed with generation as normal
with torch.no_grad():
    trained_outputs = model.generate(
        **inputs, 
        max_new_tokens=30, 
        do_sample=True,
        temperature=1.0,
        top_p=0.9
    )
