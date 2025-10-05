# 🏋️ DEFINITIVE TRAINING FIX - Complete training setup that works
print("🏋️ Setting up DEFINITIVE training configuration...")

# Calculate optimal training parameters based on dataset size
dataset_size = len(tokenized_dataset)
optimal_epochs = max(3, min(8, 200 // dataset_size))  # Scale epochs based on dataset size
optimal_batch_size = 2 if dataset_size > 50 else 1   # Larger batch for bigger datasets

print(f"📊 Dataset size: {dataset_size} examples")
print(f"🔧 Optimal epochs: {optimal_epochs}")
print(f"🔧 Optimal batch size: {optimal_batch_size}")

# CRITICAL: Proper DataCollator setup
print("🔧 Setting up DataCollatorForLanguageModeling...")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False,  # We're doing causal LM, not masked LM
    pad_to_multiple_of=8,  # Efficient padding
    return_tensors="pt"  # DataCollator handles tensor conversion
)

# Enhanced Training Arguments for 429 Q&A dataset
training_args = TrainingArguments(
    output_dir="gpt2_maximum_singapore_model",
    num_train_epochs=optimal_epochs,          # Scaled based on dataset size
    per_device_train_batch_size=optimal_batch_size,  # Optimized for dataset
    learning_rate=5e-4,                       # Slightly more conservative for larger dataset
    warmup_steps=min(50, dataset_size // 4), # Warmup based on dataset size
    logging_steps=max(1, dataset_size // 10), # Log every 10% of dataset
    save_steps=max(50, dataset_size),         # Save at end of each epoch
    eval_strategy="no",                       # Focus on training (fixed parameter name)
    save_total_limit=2,                       # Keep last 2 checkpoints
    load_best_model_at_end=False,            # Use final model
    report_to="none",                         # No external logging
    gradient_accumulation_steps=2,            # Effective batch size = batch_size * 2
    fp16=True,                               # Mixed precision for efficiency
    dataloader_pin_memory=True,              # Faster data loading
    remove_unused_columns=False,             # Keep all columns
    prediction_loss_only=True,               # Optimize for loss only
    dataloader_num_workers=0,                # Avoid multiprocessing issues in Colab
)

# Create trainer with proper setup
print("🚀 Creating trainer with definitive configuration...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,  # Critical: Use the proper data collator
    tokenizer=tokenizer,          # Pass tokenizer to trainer
)

print("✅ Trainer setup completed successfully!")
print("🎯 This configuration should resolve all tensor conversion errors!")

# Start training
print("🚀 Starting training with definitive fix...")
trainer.train()

print("✅ Training completed successfully!")
print("🎯 Model should now have comprehensive Singapore financial knowledge!")
