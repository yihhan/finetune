"""
Fixed Training Script for Financial Regulation LLM

This script uses better parameters and model selection to avoid overfitting
and actually improve performance over the base model.
"""

import json
import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FixedModelArguments:
    """Fixed model arguments with better defaults"""
    model_name_or_path: str = field(
        default="microsoft/DialoGPT-small",  # Smaller model, less overfitting
        metadata={"help": "Smaller base model to reduce overfitting"}
    )
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")

@dataclass
class FixedDataArguments:
    """Fixed data arguments"""
    dataset_path: str = field(
        default="processed_data/enhanced_training_data.json",
        metadata={"help": "Path to training dataset"}
    )
    max_seq_length: int = field(
        default=256,  # Shorter sequences, faster training
        metadata={"help": "Shorter max length for better training"}
    )

@dataclass
class FixedLoRAArguments:
    """Conservative LoRA configuration to prevent overfitting"""
    use_lora: bool = field(default=True)
    lora_r: int = field(
        default=8,  # Much smaller rank
        metadata={"help": "Smaller LoRA rank to prevent overfitting"}
    )
    lora_alpha: int = field(
        default=16,  # Smaller alpha
        metadata={"help": "Smaller alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.2,  # Higher dropout
        metadata={"help": "Higher dropout to prevent overfitting"}
    )
    target_modules: str = field(
        default="c_attn",  # Only attention, not all modules
        metadata={"help": "Target only attention modules"}
    )

class FixedFinancialRegulationTrainer:
    """Fixed trainer with conservative parameters"""
    
    def __init__(self, model_args: FixedModelArguments, data_args: FixedDataArguments, 
                 lora_args: FixedLoRAArguments, training_args: TrainingArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.lora_args = lora_args
        self.training_args = training_args
        
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with conservative settings"""
        logger.info(f"Loading smaller model: {self.model_args.model_name_or_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            use_fast=self.model_args.use_fast_tokenizer,
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with conservative settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        # Configure LoRA with conservative parameters
        if self.lora_args.use_lora:
            self.setup_lora()
    
    def setup_lora(self):
        """Configure LoRA with conservative parameters"""
        logger.info("Setting up conservative LoRA configuration")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_args.lora_r,  # Small rank
            lora_alpha=self.lora_args.lora_alpha,  # Small alpha
            lora_dropout=self.lora_args.lora_dropout,  # Higher dropout
            target_modules=[self.lora_args.target_modules],  # Only attention
            bias="none",
            inference_mode=False,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def load_dataset(self):
        """Load dataset with better train/eval split"""
        logger.info(f"Loading dataset from: {self.data_args.dataset_path}")
        
        with open(self.data_args.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(data)
        
        # Better split: 80% train, 20% eval for better validation
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        self.train_dataset = train_test_split['train']
        self.eval_dataset = train_test_split['test']
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Eval samples: {len(self.eval_dataset)}")
        
        # Tokenize datasets
        self.train_dataset = self.train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names,
        )
        
        self.eval_dataset = self.eval_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.eval_dataset.column_names,
        )
    
    def tokenize_function(self, examples):
        """Improved tokenization with simpler format"""
        texts = []
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            input_text = examples['input'][i]
            output_text = examples['output'][i]
            
            # Simpler format that works better with DialoGPT
            prompt = f"Question: {input_text}\nAnswer: {output_text}{self.tokenizer.eos_token}"
            texts.append(prompt)
        
        # Tokenize with shorter max length
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.data_args.max_seq_length,
            return_tensors=None,
        )
        
        # Set labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def train(self):
        """Train with conservative parameters"""
        logger.info("Starting conservative training...")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save model
        output_dir = Path(self.training_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to: {output_dir}")
        
        # Save LoRA adapters
        if self.lora_args.use_lora:
            lora_output_dir = output_dir / "lora_adapters"
            self.model.save_pretrained(lora_output_dir)
            logger.info(f"LoRA adapters saved to: {lora_output_dir}")
        
        return trainer

def create_fixed_training_arguments(
    output_dir: str = "fixed_finetuned_financial_model",
    num_train_epochs: int = 2,  # Fewer epochs
    per_device_train_batch_size: int = 4,
    learning_rate: float = 5e-6,  # Much lower learning rate
    warmup_steps: int = 20,  # Fewer warmup steps
    logging_steps: int = 5,
    eval_steps: int = 20,
    save_steps: int = 50,
) -> TrainingArguments:
    """Create conservative training arguments"""
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=2,  # Smaller accumulation
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=False,  # Disable for stability
        save_safetensors=True,
    )

def main():
    """Main training function with fixed parameters"""
    
    # Use conservative arguments
    model_args = FixedModelArguments()
    data_args = FixedDataArguments()
    lora_args = FixedLoRAArguments()
    
    # Create conservative training arguments
    training_args = create_fixed_training_arguments()
    
    print("🔧 FIXED TRAINING CONFIGURATION:")
    print(f"  Model: {model_args.model_name_or_path} (smaller, less overfitting)")
    print(f"  LoRA rank: {lora_args.lora_r} (conservative)")
    print(f"  Learning rate: {training_args.learning_rate} (much lower)")
    print(f"  Epochs: {training_args.num_train_epochs} (fewer)")
    print(f"  Max length: {data_args.max_seq_length} (shorter)")
    print(f"  Target modules: {lora_args.target_modules} (attention only)")
    
    # Initialize trainer
    trainer = FixedFinancialRegulationTrainer(
        model_args=model_args,
        data_args=data_args,
        lora_args=lora_args,
        training_args=training_args
    )
    
    # Setup and train
    trainer.setup_model_and_tokenizer()
    trainer.load_dataset()
    model_trainer = trainer.train()
    
    print("\n🎉 Fixed training completed!")
    print(f"Model saved to: {training_args.output_dir}")
    print("This should perform BETTER than the base model!")

if __name__ == "__main__":
    main()
