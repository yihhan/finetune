"""
Fine-tuning Script for Singapore Financial Regulation LLM

This script implements LoRA/QLoRA fine-tuning for small language models on financial regulation Q&A data.
Supports multiple base models and efficient parameter tuning.
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
import wandb
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
        default="processed_data/training_data.json",
        metadata={"help": "Path to the training dataset"}
    )
    max_seq_length: int = field(
        default=512,
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
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    target_modules: str = field(
        default="q_proj,v_proj",
        metadata={"help": "Target modules for LoRA (comma-separated)"}
    )

class FinancialRegulationTrainer:
    """Main trainer class for financial regulation model fine-tuning"""
    
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
        """Initialize model and tokenizer"""
        logger.info(f"Loading model: {self.model_args.model_name_or_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            revision=self.model_args.model_revision,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        # Configure LoRA if enabled
        if self.lora_args.use_lora:
            self.setup_lora()
    
    def setup_lora(self):
        """Configure LoRA for efficient fine-tuning"""
        logger.info("Setting up LoRA configuration")
        
        target_modules = self.lora_args.target_modules.split(",")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_args.lora_r,
            lora_alpha=self.lora_args.lora_alpha,
            lora_dropout=self.lora_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
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
        
        # Split dataset (80% train, 20% eval)
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
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
        """Tokenize the input examples"""
        # Combine instruction, input, and output
        texts = []
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            input_text = examples['input'][i]
            output_text = examples['output'][i]
            
            # Create prompt format
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}<|endoftext|>"
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
        """Run the training process"""
        logger.info("Starting training...")
        
        # Initialize wandb if enabled
        if self.training_args.report_to and "wandb" in self.training_args.report_to:
            wandb.init(
                project="financial-regulation-llm",
                name=f"finetune-{self.model_args.model_name_or_path.split('/')[-1]}",
                config={
                    "model": self.model_args.model_name_or_path,
                    "lora_r": self.lora_args.lora_r,
                    "lora_alpha": self.lora_args.lora_alpha,
                    "learning_rate": self.training_args.learning_rate,
                    "batch_size": self.training_args.per_device_train_batch_size,
                }
            )
        
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
    
    def evaluate_model(self, trainer: Trainer):
        """Evaluate the fine-tuned model"""
        logger.info("Evaluating model...")
        
        eval_results = trainer.evaluate()
        
        # Log evaluation results
        logger.info("Evaluation Results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return eval_results

def create_training_arguments(output_dir: str = "finetuned_model", 
                            num_train_epochs: int = 3,
                            per_device_train_batch_size: int = 4,
                            per_device_eval_batch_size: int = 4,
                            learning_rate: float = 5e-5,
                            warmup_steps: int = 100,
                            logging_steps: int = 10,
                            eval_steps: int = 100,
                            save_steps: int = 500,
                            save_total_limit: int = 2,
                            report_to: Optional[str] = None) -> TrainingArguments:
    """Create training arguments with sensible defaults"""
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=4,
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
    )

def main():
    """Main training function"""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LoRAArguments))
    
    # Parse arguments or use defaults
    try:
        model_args, data_args, lora_args = parser.parse_args_into_dataclasses()
    except:
        # Use default arguments if parsing fails
        model_args = ModelArguments(
            model_name_or_path="microsoft/DialoGPT-medium"  # Using smaller model for demo
        )
        data_args = DataTrainingArguments()
        lora_args = LoRAArguments()
    
    # Create training arguments
    training_args = create_training_arguments(
        output_dir="finetuned_financial_model",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Smaller batch size for demo
        learning_rate=5e-5,
        report_to=None,  # Set to "wandb" to enable logging
    )
    
    # Initialize trainer
    trainer = FinancialRegulationTrainer(
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
    
    # Evaluate model
    eval_results = trainer.evaluate_model(model_trainer)
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {training_args.output_dir}")
    print(f"Evaluation loss: {eval_results.get('eval_loss', 'N/A')}")

if __name__ == "__main__":
    main()
