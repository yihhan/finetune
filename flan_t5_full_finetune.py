"""
Full Fine-tuning Flan-T5-BASE (No LoRA)

Since LoRA keeps producing identical responses, let's try full fine-tuning
on a smaller model or with very targeted training.
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
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FullFinetuneModelArguments:
    """Full fine-tuning model arguments"""
    model_name_or_path: str = field(
        default="google/flan-t5-small",  # Use SMALL for full fine-tuning
        metadata={"help": "Smaller model for full fine-tuning"}
    )
    use_fast_tokenizer: bool = field(default=True)

@dataclass
class FullFinetuneDataArguments:
    """Data arguments for full fine-tuning"""
    dataset_path: str = field(
        default="processed_data/enhanced_training_data.json",
        metadata={"help": "Path to training dataset"}
    )
    max_input_length: int = field(
        default=256,
        metadata={"help": "Max input sequence length"}
    )
    max_target_length: int = field(
        default=128,
        metadata={"help": "Max target sequence length"}
    )

class FullFinetuneFinancialTrainer:
    """Full fine-tuning trainer (no LoRA)"""
    
    def __init__(self, model_args: FullFinetuneModelArguments, data_args: FullFinetuneDataArguments, 
                 training_args: TrainingArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer for FULL fine-tuning"""
        logger.info(f"Loading model for FULL fine-tuning: {self.model_args.model_name_or_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            use_fast=self.model_args.use_fast_tokenizer,
        )
        
        # Load model for FULL fine-tuning (all parameters trainable)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_args.model_name_or_path,
            torch_dtype=torch.float32,  # Use float32 for stability
        )
        
        # Print trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        print(f"📊 Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    def load_dataset(self):
        """Load and prepare dataset"""
        logger.info(f"Loading dataset from: {self.data_args.dataset_path}")
        
        with open(self.data_args.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Use smaller subset for full fine-tuning
        data = data[:20]  # Only use first 20 samples to prevent overfitting
        print(f"📊 Using {len(data)} samples for full fine-tuning")
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(data)
        
        # Train/eval split
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
        """Tokenize with very specific Singapore prompts"""
        inputs = []
        targets = []
        
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            input_text = examples['input'][i]
            output_text = examples['output'][i]
            
            # Very specific Singapore financial regulation prompt
            input_prompt = f"Singapore MAS regulation: {input_text}"
            inputs.append(input_prompt)
            
            # Prefix output with Singapore context
            singapore_output = f"According to MAS Singapore: {output_text}"
            targets.append(singapore_output)
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.data_args.max_input_length,
            truncation=True,
            padding="max_length",
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            targets,
            max_length=self.data_args.max_target_length,
            truncation=True,
            padding="max_length",
        )
        
        # Set labels
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    def train(self):
        """Train with full fine-tuning"""
        logger.info("Starting FULL fine-tuning (all parameters)...")
        
        # Data collator for Seq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
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
        
        return trainer

def create_full_finetune_training_arguments(
    output_dir: str = "flan_t5_full_finetune_model",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,  # Very small batch for full fine-tuning
    learning_rate: float = 5e-5,  # Conservative LR for full fine-tuning
    warmup_steps: int = 50,
    logging_steps: int = 5,
    eval_steps: int = 10,
    save_steps: int = 10,
) -> TrainingArguments:
    """Create training arguments for full fine-tuning"""
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=8,  # Large accumulation for small batch
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=1,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=False,  # Use fp32 for stability
        save_safetensors=True,
    )

def main():
    """Main full fine-tuning function"""
    
    # Use full fine-tuning arguments
    model_args = FullFinetuneModelArguments()
    data_args = FullFinetuneDataArguments()
    
    # Create training arguments
    training_args = create_full_finetune_training_arguments()
    
    print("💥 FULL FINE-TUNING CONFIGURATION:")
    print(f"  Model: {model_args.model_name_or_path} (SMALLER for full training)")
    print(f"  Approach: FULL fine-tuning (NO LoRA)")
    print(f"  All parameters: TRAINABLE")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Data: Limited to 20 samples (prevent overfitting)")
    print(f"  Output: {training_args.output_dir}")
    print("  🎯 This MUST produce different responses!")
    
    # Initialize trainer
    trainer = FullFinetuneFinancialTrainer(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    
    # Setup and train
    trainer.setup_model_and_tokenizer()
    trainer.load_dataset()
    model_trainer = trainer.train()
    
    print("\n💥 FULL fine-tuning completed!")
    print(f"Model saved to: {training_args.output_dir}")
    print("If this doesn't work, the problem is with the training data!")

if __name__ == "__main__":
    main()
