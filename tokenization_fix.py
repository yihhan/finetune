# 📚 DEFINITIVE TOKENIZATION FIX - Resolves "too many dimensions 'str'" error
print("📚 Preparing data with DEFINITIVE tokenization fix...")

# DEFINITIVE tokenization function that prevents tensor conversion errors
def tokenize_function(examples):
    """
    Fixed tokenization function that prevents 'too many dimensions str' error
    Key fixes:
    1. No return_tensors parameter (let DataCollator handle tensor conversion)
    2. padding=False (let DataCollator handle padding)
    3. Proper handling of text input
    """
    # Tokenize without converting to tensors
    tokenized = tokenizer(
        examples['text'], 
        truncation=True, 
        padding=False,      # Critical: Let DataCollator handle padding
        max_length=128,
        # Do NOT use return_tensors="pt" here
    )
    
    # Ensure we return the right format
    return tokenized

# Create dataset from training data
print(f"📊 Creating dataset from {len(training_data)} Q&A pairs...")
dataset = Dataset.from_dict({'text': training_data})

# Apply tokenization
print("🔧 Applying fixed tokenization...")
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True,
    remove_columns=dataset.column_names  # Remove original text column
)

print(f"✅ Tokenized {len(tokenized_dataset)} examples successfully!")
print("🎯 DataCollatorForLanguageModeling will handle padding and tensor conversion during training")

# Verify the tokenization worked
print(f"📋 Sample tokenized data keys: {list(tokenized_dataset[0].keys())}")
print(f"📏 Sample input_ids length: {len(tokenized_dataset[0]['input_ids'])}")
