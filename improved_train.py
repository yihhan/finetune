"""
Improved Training Script for Financial Regulation LLM

This script uses better training parameters and a larger dataset for improved results.
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
    HfArgumentParser,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset, load_dataset
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    model_name_or_path: str = field(
        default="microsoft/DialoGPT-medium",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )

@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    dataset_path: str = field(
        default="processed_data/enhanced_training_data.json",
        metadata={"help": "Path to the training dataset"}
    )
    max_seq_length: int = field(
        default=1024,  # Increased for better context
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )

@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration."""
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for efficient fine-tuning"}
    )
    lora_r: int = field(
        default=32,  # Increased rank for better learning
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=64,  # Increased alpha
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj",  # More modules for DialoGPT
        metadata={"help": "Target modules for LoRA (comma-separated)"}
    )

class ImprovedFinancialRegulationTrainer:
    """Improved trainer class with better configuration"""
    
    def __init__(self, model_args: ModelArguments, data_args: DataTrainingArguments, 
                 lora_args: LoRAArguments, training_args: TrainingArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.lora_args = lora_args
        self.training_args = training_args
        
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with better configuration"""
        logger.info(f"Loading model: {self.model_args.model_name_or_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
        )
        
        # Add special tokens for better instruction following
        special_tokens = {
            "pad_token": "<pad>",
            "eos_token": "<eos>",
            "bos_token": "<bos>",
            "unk_token": "<unk>"
        }
        
        for token, value in special_tokens.items():
            if getattr(self.tokenizer, token) is None:
                setattr(self.tokenizer, token, value)
        
        # Resize token embeddings if needed
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            revision=self.model_args.model_revision,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Resize model embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Configure LoRA if enabled
        if self.lora_args.use_lora:
            self.setup_lora()
    
    def setup_lora(self):
        """Configure LoRA for efficient fine-tuning with better parameters"""
        logger.info("Setting up LoRA configuration")
        
        target_modules = self.lora_args.target_modules.split(",")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_args.lora_r,
            lora_alpha=self.lora_args.lora_alpha,
            lora_dropout=self.lora_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            inference_mode=False,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def load_dataset(self):
        """Load and preprocess the training dataset"""
        logger.info(f"Loading dataset from: {self.data_args.dataset_path}")
        
        # Load JSON dataset
        with open(self.data_args.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to HuggingFace dataset format
        dataset = Dataset.from_list(data)
        
        # Split dataset (90% train, 10% eval) - more training data
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
        self.train_dataset = train_test_split['train']
        self.eval_dataset = train_test_split['test']
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Eval samples: {len(self.eval_dataset)}")
        
        # Tokenize datasets
        self.train_dataset = self.train_dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=self.train_dataset.column_names,
        )
        
        self.eval_dataset = self.eval_dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=self.eval_dataset.column_names,
        )
    
    def tokenize_function(self, examples):
        """Improved tokenization with better prompt formatting"""
        # Create better formatted prompts
        texts = []
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            input_text = examples['input'][i]
            output_text = examples['output'][i]
            
            # Create a more structured prompt
            prompt = f"<bos><instruction>{instruction}</instruction><question>{input_text}</question><answer>{output_text}</answer><eos>"
            texts.append(prompt)
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.data_args.max_seq_length,
            return_tensors=None,
        )
        
        # Set labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def train(self):
        """Run the training process with better monitoring"""
        logger.info("Starting improved training...")
        
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
        
        # Save LoRA adapters separately if using LoRA
        if self.lora_args.use_lora:
            lora_output_dir = output_dir / "lora_adapters"
            self.model.save_pretrained(lora_output_dir)
            logger.info(f"LoRA adapters saved to: {lora_output_dir}")
        
        return trainer

def create_improved_training_arguments(output_dir: str = "improved_finetuned_model", 
                                     num_train_epochs: int = 5,  # More epochs
                                     per_device_train_batch_size: int = 2,
                                     per_device_eval_batch_size: int = 2,
                                     learning_rate: float = 2e-5,  # Lower learning rate
                                     warmup_steps: int = 50,
                                     logging_steps: int = 5,
                                     eval_steps: int = 25,
                                     save_steps: int = 100,
                                     save_total_limit: int = 3,
                                     report_to: Optional[str] = None) -> TrainingArguments:
    """Create improved training arguments"""
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=8,  # More accumulation for stability
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        save_safetensors=True,
    )

def main():
    """Main training function with improved configuration"""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LoRAArguments))
    
    # Use improved default arguments
    model_args = ModelArguments(
        model_name_or_path="microsoft/DialoGPT-medium"
    )
    data_args = DataTrainingArguments()
    lora_args = LoRAArguments()
    
    # Create improved training arguments
    training_args = create_improved_training_arguments(
        output_dir="improved_finetuned_financial_model",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        report_to=None,
    )
    
    # Initialize improved trainer
    trainer = ImprovedFinancialRegulationTrainer(
        model_args=model_args,
        data_args=data_args,
        lora_args=lora_args,
        training_args=training_args
    )
    
    # Setup model and tokenizer
    trainer.setup_model_and_tokenizer()
    
    # Load dataset
    trainer.load_dataset()
    
    # Train model
    model_trainer = trainer.train()
    
    print("\n🎉 Improved training completed successfully!")
    print(f"Model saved to: {training_args.output_dir}")
    print("This model should provide much better responses!")

if __name__ == "__main__":
    main()
