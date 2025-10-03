"""
Configuration file for Financial Regulation LLM Fine-tuning

This file contains all the configuration parameters for training, evaluation, and inference.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    
    # Base model settings
    base_model_name: str = "microsoft/DialoGPT-medium"  # Use smaller model for demo
    # Alternative models for production:
    # "meta-llama/Llama-2-7b-hf"  # Requires approval from Meta
    # "microsoft/DialoGPT-large"
    # "mistralai/Mistral-7B-v0.1"
    
    # Model parameters
    max_seq_length: int = 512
    use_fast_tokenizer: bool = True
    model_revision: str = "main"
    
    # Generation parameters
    max_new_tokens: int = 300
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1

@dataclass
class LoRAConfig:
    """LoRA configuration parameters"""
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Target modules for different model architectures
    target_modules_llama: List[str] = None
    target_modules_dialo: List[str] = None
    target_modules_gpt: List[str] = None
    
    def __post_init__(self):
        if self.target_modules_llama is None:
            self.target_modules_llama = ["q_proj", "v_proj", "k_proj", "o_proj"]
        if self.target_modules_dialo is None:
            self.target_modules_dialo = ["q_proj", "v_proj"]
        if self.target_modules_gpt is None:
            self.target_modules_gpt = ["c_attn", "c_proj"]

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    
    # Dataset settings
    dataset_path: str = "processed_data/training_data.json"
    train_test_split: float = 0.2
    preprocessing_num_workers: int = 4
    
    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Optimization
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = False
    
    # Logging and saving
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 2
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Output settings
    output_dir: str = "finetuned_financial_model"
    report_to: Optional[str] = None  # Set to "wandb" to enable

@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters"""
    
    # Dataset settings
    test_dataset_path: str = "processed_data/financial_regulation_qa.json"
    num_eval_samples: int = 10
    
    # Model paths
    base_model_path: str = "microsoft/DialoGPT-medium"
    finetuned_model_path: str = "finetuned_financial_model"
    
    # Evaluation metrics
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["bleu", "rouge1", "rouge2", "rougeL"]
    
    # RAG settings
    openai_api_key: Optional[str] = None
    rag_model: str = "gpt-4"
    rag_temperature: float = 0.3
    rag_max_tokens: int = 300
    
    # Output settings
    output_dir: str = "evaluation_results"

@dataclass
class InferenceConfig:
    """Inference configuration parameters"""
    
    # Model settings
    base_model_path: str = "microsoft/DialoGPT-medium"
    finetuned_model_path: str = "finetuned_financial_model"
    
    # Generation settings
    max_length: int = 300
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Device settings
    device: Optional[str] = None  # Auto-detect if None
    
    # Sample questions
    sample_questions_file: str = "sample_questions.json"

@dataclass
class DatasetConfig:
    """Dataset configuration parameters"""
    
    # Input/Output paths
    data_dir: str = "data"
    processed_data_dir: str = "processed_data"
    
    # Dataset categories
    categories: Dict[str, str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = {
                "capital_requirements": "Capital Adequacy Requirements",
                "risk_management": "Risk Management",
                "compliance": "Compliance and Reporting",
                "ai_advisory": "AI in Financial Advisory",
                "data_protection": "Data Protection and Privacy",
                "anti_money_laundering": "Anti-Money Laundering",
                "cybersecurity": "Cybersecurity",
                "digital_banking": "Digital Banking Services"
            }

class ConfigManager:
    """Configuration manager for loading and saving configs"""
    
    def __init__(self):
        self.model_config = ModelConfig()
        self.lora_config = LoRAConfig()
        self.training_config = TrainingConfig()
        self.evaluation_config = EvaluationConfig()
        self.inference_config = InferenceConfig()
        self.dataset_config = DatasetConfig()
    
    def save_config(self, filepath: str):
        """Save all configurations to a JSON file"""
        config_dict = {
            "model": self.model_config.__dict__,
            "lora": self.lora_config.__dict__,
            "training": self.training_config.__dict__,
            "evaluation": self.evaluation_config.__dict__,
            "inference": self.inference_config.__dict__,
            "dataset": self.dataset_config.__dict__,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def load_config(self, filepath: str):
        """Load configurations from a JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Update configurations
        for key, value in config_dict.items():
            if hasattr(self, f"{key}_config"):
                config_obj = getattr(self, f"{key}_config")
                for param, param_value in value.items():
                    if hasattr(config_obj, param):
                        setattr(config_obj, param, param_value)
    
    def get_target_modules(self, model_name: str) -> List[str]:
        """Get appropriate target modules based on model architecture"""
        model_name_lower = model_name.lower()
        
        if "llama" in model_name_lower:
            return self.lora_config.target_modules_llama
        elif "dialo" in model_name_lower:
            return self.lora_config.target_modules_dialo
        elif "gpt" in model_name_lower:
            return self.lora_config.target_modules_gpt
        else:
            # Default to common modules
            return ["q_proj", "v_proj"]

# Global configuration instance
config = ConfigManager()

# Preset configurations for different scenarios
def get_production_config() -> ConfigManager:
    """Get configuration optimized for production use"""
    prod_config = ConfigManager()
    
    # Use larger, more capable models
    prod_config.model_config.base_model_name = "meta-llama/Llama-2-7b-hf"
    prod_config.model_config.max_seq_length = 1024
    prod_config.model_config.max_new_tokens = 500
    
    # More aggressive LoRA settings
    prod_config.lora_config.lora_r = 32
    prod_config.lora_config.lora_alpha = 64
    
    # Longer training
    prod_config.training_config.num_train_epochs = 5
    prod_config.training_config.per_device_train_batch_size = 2
    prod_config.training_config.learning_rate = 2e-5
    
    return prod_config

def get_demo_config() -> ConfigManager:
    """Get configuration optimized for demonstration purposes"""
    demo_config = ConfigManager()
    
    # Use smaller, faster models
    demo_config.model_config.base_model_name = "microsoft/DialoGPT-medium"
    demo_config.model_config.max_seq_length = 512
    demo_config.model_config.max_new_tokens = 200
    
    # Conservative LoRA settings
    demo_config.lora_config.lora_r = 8
    demo_config.lora_config.lora_alpha = 16
    
    # Shorter training for quick demo
    demo_config.training_config.num_train_epochs = 2
    demo_config.training_config.per_device_train_batch_size = 4
    demo_config.training_config.eval_steps = 50
    demo_config.training_config.save_steps = 100
    
    return demo_config

def get_research_config() -> ConfigManager:
    """Get configuration optimized for research and experimentation"""
    research_config = ConfigManager()
    
    # Balanced settings for experimentation
    research_config.model_config.max_seq_length = 768
    research_config.lora_config.lora_r = 16
    research_config.lora_config.lora_alpha = 32
    
    # More frequent logging and evaluation
    research_config.training_config.logging_steps = 5
    research_config.training_config.eval_steps = 50
    research_config.training_config.save_steps = 200
    research_config.training_config.report_to = "wandb"
    
    return research_config
