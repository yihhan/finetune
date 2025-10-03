"""
Fixed Flan-T5-BASE Training Script with Better Parameters

Using more aggressive LoRA parameters to ensure actual learning
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
class FixedFlanT5ModelArguments:
    """Fixed Flan-T5-base model arguments"""
    model_name_or_path: str = field(
        default="google/flan-t5-base",
        metadata={"help": "Flan-T5-base model"}
    )
    use_fast_tokenizer: bool = field(default=True)

@dataclass
class FixedFlanT5DataArguments:
    """Data arguments for Flan-T5-base"""
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

@dataclass
class FixedFlanT5LoRAArguments:
    """AGGRESSIVE LoRA configuration to ensure learning"""
    use_lora: bool = field(default=True)
    lora_r: int = field(
        default=32,  # Much higher rank
        metadata={"help": "Higher LoRA rank for more capacity"}
    )
    lora_alpha: int = field(
        default=64,  # Higher alpha for stronger adaptation
        metadata={"help": "Higher LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,  # Lower dropout
        metadata={"help": "Lower dropout"}
    )
    target_modules: list = field(
        default_factory=lambda: ["q", "v", "k", "o", "wi", "wo"],  # More modules
        metadata={"help": "Target more modules for better adaptation"}
    )

class FixedFlanT5FinancialTrainer:
    """Fixed Flan-T5-base trainer with aggressive parameters"""
    
    def __init__(self, model_args: FixedFlanT5ModelArguments, data_args: FixedFlanT5DataArguments, 
                 lora_args: FixedFlanT5LoRAArguments, training_args: TrainingArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.lora_args = lora_args
        self.training_args = training_args
        
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def setup_model_and_tokenizer(self):
        """Initialize Flan-T5-base model and tokenizer"""
        logger.info(f"Loading Flan-T5-base model: {self.model_args.model_name_or_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            use_fast=self.model_args.use_fast_tokenizer,
        )
        
        # Load Seq2Seq model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_args.model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        # Configure AGGRESSIVE LoRA
        if self.lora_args.use_lora:
            self.setup_lora()
    
    def setup_lora(self):
        """Configure AGGRESSIVE LoRA for actual learning"""
        logger.info("Setting up AGGRESSIVE LoRA for Flan-T5-base")
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.lora_args.lora_r,  # Higher rank
            lora_alpha=self.lora_args.lora_alpha,  # Higher alpha
            lora_dropout=self.lora_args.lora_dropout,  # Lower dropout
            target_modules=self.lora_args.target_modules,  # More modules
            bias="none",
            inference_mode=False,  # CRITICAL: Must be False for training!
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Verify training mode
        print(f"📊 Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def load_dataset(self):
        """Load and prepare dataset for Flan-T5-base"""
        logger.info(f"Loading dataset from: {self.data_args.dataset_path}")
        
        with open(self.data_args.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
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
        """Tokenize for Flan-T5-base with better prompts"""
        inputs = []
        targets = []
        
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            input_text = examples['input'][i]
            output_text = examples['output'][i]
            
            # More specific prompt for Singapore financial regulations
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
        """Train Flan-T5-base model with aggressive parameters"""
        logger.info("Starting AGGRESSIVE Flan-T5-base training...")
        
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
        
        # Save LoRA adapters with inference_mode=False
        if self.lora_args.use_lora:
            lora_output_dir = output_dir / "lora_adapters"
            self.model.save_pretrained(lora_output_dir)
            logger.info(f"LoRA adapters saved to: {lora_output_dir}")
            
            # Verify the saved config
            config_path = lora_output_dir / "adapter_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"✅ Saved LoRA config: r={config['r']}, alpha={config['lora_alpha']}")
                print(f"✅ Inference mode: {config.get('inference_mode', 'not set')}")
        
        return trainer

def create_fixed_training_arguments(
    output_dir: str = "flan_t5_base_fixed_model",
    num_train_epochs: int = 4,  # More epochs
    per_device_train_batch_size: int = 2,
    learning_rate: float = 1e-4,  # Higher learning rate
    warmup_steps: int = 100,  # More warmup
    logging_steps: int = 10,
    eval_steps: int = 25,
    save_steps: int = 25,
) -> TrainingArguments:
    """Create aggressive training arguments for actual learning"""
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=4,
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
    """Main training function with AGGRESSIVE parameters"""
    
    # Use aggressive arguments
    model_args = FixedFlanT5ModelArguments()
    data_args = FixedFlanT5DataArguments()
    lora_args = FixedFlanT5LoRAArguments()
    
    # Create aggressive training arguments
    training_args = create_fixed_training_arguments()
    
    print("🔥 AGGRESSIVE FLAN-T5-BASE TRAINING CONFIGURATION:")
    print(f"  Model: {model_args.model_name_or_path}")
    print(f"  LoRA rank: {lora_args.lora_r} (MUCH higher)")
    print(f"  LoRA alpha: {lora_args.lora_alpha} (MUCH higher)")
    print(f"  Target modules: {lora_args.target_modules} (MORE modules)")
    print(f"  Learning rate: {training_args.learning_rate} (HIGHER)")
    print(f"  Epochs: {training_args.num_train_epochs} (MORE)")
    print(f"  Output: {training_args.output_dir}")
    print("  🎯 This should actually LEARN something!")
    
    # Initialize trainer
    trainer = FixedFlanT5FinancialTrainer(
        model_args=model_args,
        data_args=data_args,
        lora_args=lora_args,
        training_args=training_args
    )
    
    # Setup and train
    trainer.setup_model_and_tokenizer()
    trainer.load_dataset()
    model_trainer = trainer.train()
    
    print("\n🎉 AGGRESSIVE Flan-T5-base training completed!")
    print(f"Model saved to: {training_args.output_dir}")
    print("This should show REAL improvement over base model!")

if __name__ == "__main__":
    main()
