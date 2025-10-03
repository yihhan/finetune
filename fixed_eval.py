"""
Fixed Evaluation Script for Financial Regulation LLM

This script compares the FIXED training approach with proper evaluation.
"""

import json
import os
import time
import torch
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

# Evaluation metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np

# Model imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FixedEvaluationResult:
    """Data class for fixed evaluation results"""
    question: str
    ground_truth: str
    base_model_response: str
    fixed_response: str
    base_model_bleu: float
    fixed_bleu: float
    base_model_rouge: Dict[str, float]
    fixed_rouge: Dict[str, float]
    base_model_time: float
    fixed_time: float

class FixedModelEvaluator:
    """Fixed evaluator comparing base vs fixed fine-tuned model"""
    
    def __init__(self, 
                 base_model_path: str = "microsoft/DialoGPT-small",
                 fixed_model_path: str = "fixed_finetuned_financial_model"):
        
        self.base_model_path = base_model_path
        self.fixed_model_path = fixed_model_path
        
        # Initialize models
        self.base_model = None
        self.base_tokenizer = None
        self.fixed_model = None
        self.fixed_tokenizer = None
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        
    def load_base_model(self):
        """Load the base model for comparison"""
        logger.info(f"Loading base model: {self.base_model_path}")
        
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
    
    def load_fixed_model(self):
        """Load the fixed fine-tuned model"""
        logger.info(f"Loading fixed model: {self.fixed_model_path}")
        
        try:
            # Try to load LoRA adapters first
            lora_path = Path(self.fixed_model_path) / "lora_adapters"
            if lora_path.exists():
                logger.info("Loading LoRA adapters...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                self.fixed_model = PeftModel.from_pretrained(base_model, lora_path)
                self.fixed_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            else:
                logger.info("Loading full fine-tuned model...")
                self.fixed_tokenizer = AutoTokenizer.from_pretrained(self.fixed_model_path)
                self.fixed_model = AutoModelForCausalLM.from_pretrained(
                    self.fixed_model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
            
            if self.fixed_tokenizer.pad_token is None:
                self.fixed_tokenizer.pad_token = self.fixed_tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Error loading fixed model: {e}")
            logger.info("Using base model as fallback...")
            self.fixed_model = self.base_model
            self.fixed_tokenizer = self.base_tokenizer
    
    def create_simple_prompt(self, question: str) -> str:
        """Create simple prompt matching training format"""
        return f"Question: {question}\nAnswer:"
    
    def generate_response(self, model, tokenizer, question: str, max_new_tokens: int = 100) -> str:
        """Generate response with conservative parameters"""
        
        prompt = self.create_simple_prompt(question)
        
        # Tokenize
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=200
        )
        
        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
        
        # Decode
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        if "Answer:" in full_response:
            response = full_response.split("Answer:")[-1].strip()
        else:
            response = full_response.replace(prompt, "").strip()
        
        return response.replace(tokenizer.eos_token, "").strip()
    
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score"""
        if not candidate or not reference:
            return 0.0
        
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()
        
        if not candidate_tokens:
            return 0.0
        
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=self.smoothing)
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        if not candidate or not reference:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure,
        }
    
    def evaluate_single_question(self, question: str, ground_truth: str) -> FixedEvaluationResult:
        """Evaluate a single question"""
        
        # Base model
        start_time = time.time()
        base_response = self.generate_response(self.base_model, self.base_tokenizer, question)
        base_time = time.time() - start_time
        
        # Fixed model
        start_time = time.time()
        fixed_response = self.generate_response(self.fixed_model, self.fixed_tokenizer, question)
        fixed_time = time.time() - start_time
        
        # Calculate metrics
        base_bleu = self.calculate_bleu(ground_truth, base_response)
        fixed_bleu = self.calculate_bleu(ground_truth, fixed_response)
        
        base_rouge = self.calculate_rouge(ground_truth, base_response)
        fixed_rouge = self.calculate_rouge(ground_truth, fixed_response)
        
        return FixedEvaluationResult(
            question=question,
            ground_truth=ground_truth,
            base_model_response=base_response,
            fixed_response=fixed_response,
            base_model_bleu=base_bleu,
            fixed_bleu=fixed_bleu,
            base_model_rouge=base_rouge,
            fixed_rouge=fixed_rouge,
            base_model_time=base_time,
            fixed_time=fixed_time,
        )
    
    def load_test_dataset(self, dataset_path: str = "processed_data/holdout_test_set.json") -> List[Dict]:
        """Load holdout test dataset"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def run_evaluation(self, dataset_path: str = "processed_data/holdout_test_set.json") -> List[FixedEvaluationResult]:
        """Run complete evaluation"""
        
        # Load models
        self.load_base_model()
        self.load_fixed_model()
        
        # Load test data
        test_data = self.load_test_dataset(dataset_path)
        logger.info(f"Evaluating on {len(test_data)} test samples")
        
        results = []
        for i, item in enumerate(test_data):
            logger.info(f"Evaluating question {i+1}/{len(test_data)}")
            
            result = self.evaluate_single_question(
                question=item['question'],
                ground_truth=item['answer']
            )
            results.append(result)
        
        return results
    
    def save_results(self, results: List[FixedEvaluationResult], output_dir: str = "fixed_evaluation_results"):
        """Save evaluation results"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Detailed results
        detailed_results = []
        for result in results:
            detailed_results.append({
                "question": result.question,
                "ground_truth": result.ground_truth,
                "base_model_response": result.base_model_response,
                "fixed_response": result.fixed_response,
                "base_model_bleu": result.base_model_bleu,
                "fixed_bleu": result.fixed_bleu,
                "base_model_rouge": result.base_model_rouge,
                "fixed_rouge": result.fixed_rouge,
                "base_model_time": result.base_model_time,
                "fixed_time": result.fixed_time,
            })
        
        with open(output_path / "detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # Summary metrics
        summary = {
            "base_model": {
                "avg_bleu": np.mean([r.base_model_bleu for r in results]),
                "avg_rouge1": np.mean([r.base_model_rouge["rouge1"] for r in results]),
                "avg_rouge2": np.mean([r.base_model_rouge["rouge2"] for r in results]),
                "avg_rougeL": np.mean([r.base_model_rouge["rougeL"] for r in results]),
                "avg_time": np.mean([r.base_model_time for r in results]),
            },
            "fixed_model": {
                "avg_bleu": np.mean([r.fixed_bleu for r in results]),
                "avg_rouge1": np.mean([r.fixed_rouge["rouge1"] for r in results]),
                "avg_rouge2": np.mean([r.fixed_rouge["rouge2"] for r in results]),
                "avg_rougeL": np.mean([r.fixed_rouge["rougeL"] for r in results]),
                "avg_time": np.mean([r.fixed_time for r in results]),
            }
        }
        
        with open(output_path / "summary_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("FIXED MODEL EVALUATION RESULTS")
        print("="*80)
        
        base_bleu = summary["base_model"]["avg_bleu"]
        fixed_bleu = summary["fixed_model"]["avg_bleu"]
        base_rouge1 = summary["base_model"]["avg_rouge1"]
        fixed_rouge1 = summary["fixed_model"]["avg_rouge1"]
        
        print(f"{'Model':<20} {'BLEU':<10} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'Time (s)':<10}")
        print("-" * 80)
        print(f"{'Base Model':<20} {base_bleu:<10.4f} {summary['base_model']['avg_rouge1']:<10.4f} {summary['base_model']['avg_rouge2']:<10.4f} {summary['base_model']['avg_rougeL']:<10.4f} {summary['base_model']['avg_time']:<10.2f}")
        print(f"{'Fixed Model':<20} {fixed_bleu:<10.4f} {summary['fixed_model']['avg_rouge1']:<10.4f} {summary['fixed_model']['avg_rouge2']:<10.4f} {summary['fixed_model']['avg_rougeL']:<10.4f} {summary['fixed_model']['avg_time']:<10.2f}")
        print("="*80)
        
        # Improvement analysis
        if base_bleu > 0:
            bleu_improvement = (fixed_bleu / base_bleu) if base_bleu > 0 else 0
            print(f"\n📊 BLEU Improvement: {bleu_improvement:.2f}x")
        
        if base_rouge1 > 0:
            rouge_improvement = (fixed_rouge1 / base_rouge1) if base_rouge1 > 0 else 0
            print(f"📊 ROUGE-1 Improvement: {rouge_improvement:.2f}x")
        
        if fixed_bleu > base_bleu:
            print("✅ Fixed model performs BETTER than base model!")
        else:
            print("❌ Fixed model still needs improvement")
        
        print(f"\nResults saved to: {output_path}")
        
        return summary

def main():
    """Main evaluation function"""
    
    # Initialize evaluator
    evaluator = FixedModelEvaluator(
        base_model_path="microsoft/DialoGPT-small",
        fixed_model_path="fixed_finetuned_financial_model"
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Save and display results
    summary = evaluator.save_results(results)
    
    print(f"\n🎯 Evaluation completed! Evaluated {len(results)} samples")

if __name__ == "__main__":
    main()
