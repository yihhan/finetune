"""
Flan-T5 Training with Large Dataset (500+ samples)

This script trains Flan-T5 with the large generated dataset for proper SFT learning.
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
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LargeDatasetModelArguments:
    """Model arguments for large dataset training"""
    model_name_or_path: str = field(
        default="google/flan-t5-base",  # Use base for large dataset
        metadata={"help": "Flan-T5-base for large dataset training"}
    )
    use_fast_tokenizer: bool = field(default=True)

@dataclass
class LargeDatasetDataArguments:
    """Data arguments for large dataset"""
    dataset_path: str = field(
        default="processed_data/large_training_data.json",
        metadata={"help": "Path to large training dataset"}
    )
    max_input_length: int = field(
        default=256,
        metadata={"help": "Max input sequence length"}
    )
    max_target_length: int = field(
        default=200,  # Longer for detailed answers
        metadata={"help": "Max target sequence length"}
    )

@dataclass
class LargeDatasetLoRAArguments:
    """LoRA configuration for large dataset (can be more conservative)"""
    use_lora: bool = field(default=True)
    lora_r: int = field(
        default=16,  # Moderate rank for large dataset
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,  # Moderate alpha
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    target_modules: list = field(
        default_factory=lambda: ["q", "v", "k", "o"],
        metadata={"help": "Target modules for LoRA"}
    )

class LargeDatasetFinancialTrainer:
    """Trainer for large dataset SFT"""
    
    def __init__(self, model_args: LargeDatasetModelArguments, data_args: LargeDatasetDataArguments, 
                 lora_args: LargeDatasetLoRAArguments, training_args: TrainingArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.lora_args = lora_args
        self.training_args = training_args
        
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading model for large dataset training: {self.model_args.model_name_or_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            use_fast=self.model_args.use_fast_tokenizer,
        )
        
        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_args.model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        # Configure LoRA
        if self.lora_args.use_lora:
            self.setup_lora()
    
    def setup_lora(self):
        """Configure LoRA for large dataset"""
        logger.info("Setting up LoRA for large dataset training")
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.lora_args.lora_r,
            lora_alpha=self.lora_args.lora_alpha,
            lora_dropout=self.lora_args.lora_dropout,
            target_modules=self.lora_args.target_modules,
            bias="none",
            inference_mode=False,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Verify training mode
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Trainable parameters: {trainable:,}")
    
    def load_dataset(self):
        """Load large dataset"""
        logger.info(f"Loading large dataset from: {self.data_args.dataset_path}")
        
        if not os.path.exists(self.data_args.dataset_path):
            print(f"❌ Large dataset not found: {self.data_args.dataset_path}")
            print("🚀 Run: python generate_training_data.py first!")
            raise FileNotFoundError(f"Dataset not found: {self.data_args.dataset_path}")
        
        with open(self.data_args.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 Loaded {len(data)} training samples")
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(data)
        
        # Train/eval split (90/10 for large dataset)
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
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
        """Tokenize for large dataset training"""
        inputs = []
        targets = []
        
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            input_text = examples['input'][i]
            output_text = examples['output'][i]
            
            # Clean, professional prompt format
            input_prompt = f"Answer this Singapore financial regulation question: {input_text}"
            inputs.append(input_prompt)
            targets.append(output_text)
        
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
        """Train with large dataset"""
        logger.info("Starting large dataset training...")
        
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
        
        # Save LoRA adapters
        if self.lora_args.use_lora:
            lora_output_dir = output_dir / "lora_adapters"
            self.model.save_pretrained(lora_output_dir)
            logger.info(f"LoRA adapters saved to: {lora_output_dir}")
        
        return trainer

def create_large_dataset_training_arguments(
    output_dir: str = "flan_t5_large_dataset_model",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    learning_rate: float = 5e-5,
    warmup_steps: int = 200,  # More warmup for large dataset
    logging_steps: int = 50,
    eval_steps: int = 100,
    save_steps: int = 100,
) -> TrainingArguments:
    """Create training arguments for large dataset"""
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),
        save_safetensors=True,
    )

def main():
    """Main training function for large dataset"""
    
    # Check if large dataset exists
    dataset_path = "processed_data/large_training_data.json"
    if not os.path.exists(dataset_path):
        print("❌ Large dataset not found!")
        print("🚀 Run this first: python generate_training_data.py")
        return
    
    # Load dataset info
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("📊 LARGE DATASET SFT TRAINING")
    print("="*50)
    print(f"Dataset: {len(data)} training samples")
    print("Model: Flan-T5-base with LoRA")
    print("Expected: Much better results with large dataset!")
    
    # Use large dataset arguments
    model_args = LargeDatasetModelArguments()
    data_args = LargeDatasetDataArguments()
    lora_args = LargeDatasetLoRAArguments()
    
    # Create training arguments
    training_args = create_large_dataset_training_arguments()
    
    print(f"\n🎯 TRAINING CONFIGURATION:")
    print(f"  Model: {model_args.model_name_or_path}")
    print(f"  Dataset: {len(data)} samples")
    print(f"  LoRA rank: {lora_args.lora_r}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Output: {training_args.output_dir}")
    
    # Initialize trainer
    trainer = LargeDatasetFinancialTrainer(
        model_args=model_args,
        data_args=data_args,
        lora_args=lora_args,
        training_args=training_args
    )
    
    # Setup and train
    trainer.setup_model_and_tokenizer()
    trainer.load_dataset()
    model_trainer = trainer.train()
    
    print(f"\n🎉 Large dataset training completed!")
    print(f"Model saved to: {training_args.output_dir}")
    print("With 500+ samples, this should show significant improvement!")

if __name__ == "__main__":
    main()
