"""
Flan-T5 Evaluation Script for Financial Regulation Q&A

Proper evaluation for Seq2Seq models like Flan-T5
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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FlanT5EvaluationResult:
    """Data class for Flan-T5 evaluation results"""
    question: str
    ground_truth: str
    base_model_response: str
    finetuned_response: str
    base_model_bleu: float
    finetuned_bleu: float
    base_model_rouge: Dict[str, float]
    finetuned_rouge: Dict[str, float]
    base_model_time: float
    finetuned_time: float

class FlanT5ModelEvaluator:
    """Flan-T5 evaluator comparing base vs fine-tuned model"""
    
    def __init__(self, 
                 base_model_path: str = "google/flan-t5-small",
                 finetuned_model_path: str = "flan_t5_financial_model"):
        
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        
        # Initialize models
        self.base_model = None
        self.base_tokenizer = None
        self.finetuned_model = None
        self.finetuned_tokenizer = None
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        
    def load_base_model(self):
        """Load the base Flan-T5 model"""
        logger.info(f"Loading base Flan-T5 model: {self.base_model_path}")
        
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    
    def load_finetuned_model(self):
        """Load the fine-tuned Flan-T5 model"""
        logger.info(f"Loading fine-tuned Flan-T5 model: {self.finetuned_model_path}")
        
        try:
            # Try to load LoRA adapters first
            lora_path = Path(self.finetuned_model_path) / "lora_adapters"
            if lora_path.exists():
                logger.info("Loading LoRA adapters...")
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.base_model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                self.finetuned_model = PeftModel.from_pretrained(base_model, lora_path)
                self.finetuned_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            else:
                logger.info("Loading full fine-tuned model...")
                self.finetuned_tokenizer = AutoTokenizer.from_pretrained(self.finetuned_model_path)
                self.finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.finetuned_model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            logger.info("Using base model as fallback...")
            self.finetuned_model = self.base_model
            self.finetuned_tokenizer = self.base_tokenizer
    
    def create_input_prompt(self, question: str) -> str:
        """Create input prompt for Flan-T5"""
        return f"Answer this financial regulation question: {question}"
    
    def generate_response(self, model, tokenizer, question: str, max_new_tokens: int = 128) -> str:
        """Generate response using Flan-T5"""
        
        input_prompt = self.create_input_prompt(question)
        
        # Tokenize input
        inputs = tokenizer(
            input_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256
        )
        
        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                early_stopping=True,
                repetition_penalty=1.1,
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    
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
    
    def evaluate_single_question(self, question: str, ground_truth: str) -> FlanT5EvaluationResult:
        """Evaluate a single question"""
        
        # Base model
        start_time = time.time()
        base_response = self.generate_response(self.base_model, self.base_tokenizer, question)
        base_time = time.time() - start_time
        
        # Fine-tuned model
        start_time = time.time()
        finetuned_response = self.generate_response(self.finetuned_model, self.finetuned_tokenizer, question)
        finetuned_time = time.time() - start_time
        
        # Calculate metrics
        base_bleu = self.calculate_bleu(ground_truth, base_response)
        finetuned_bleu = self.calculate_bleu(ground_truth, finetuned_response)
        
        base_rouge = self.calculate_rouge(ground_truth, base_response)
        finetuned_rouge = self.calculate_rouge(ground_truth, finetuned_response)
        
        return FlanT5EvaluationResult(
            question=question,
            ground_truth=ground_truth,
            base_model_response=base_response,
            finetuned_response=finetuned_response,
            base_model_bleu=base_bleu,
            finetuned_bleu=finetuned_bleu,
            base_model_rouge=base_rouge,
            finetuned_rouge=finetuned_rouge,
            base_model_time=base_time,
            finetuned_time=finetuned_time,
        )
    
    def load_test_dataset(self, dataset_path: str = "processed_data/holdout_test_set.json") -> List[Dict]:
        """Load holdout test dataset"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def run_evaluation(self, dataset_path: str = "processed_data/holdout_test_set.json") -> List[FlanT5EvaluationResult]:
        """Run complete Flan-T5 evaluation"""
        
        # Load models
        self.load_base_model()
        self.load_finetuned_model()
        
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
            
            # Show progress
            print(f"\nQ: {item['question']}")
            print(f"Ground Truth: {item['answer'][:100]}...")
            print(f"Base Flan-T5: {result.base_model_response[:100]}...")
            print(f"Fine-tuned:   {result.finetuned_response[:100]}...")
            print(f"BLEU: Base={result.base_model_bleu:.4f}, Fine-tuned={result.finetuned_bleu:.4f}")
            print("-" * 80)
        
        return results
    
    def save_results(self, results: List[FlanT5EvaluationResult], output_dir: str = "flan_t5_evaluation_results"):
        """Save Flan-T5 evaluation results"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Detailed results
        detailed_results = []
        for result in results:
            detailed_results.append({
                "question": result.question,
                "ground_truth": result.ground_truth,
                "base_model_response": result.base_model_response,
                "finetuned_response": result.finetuned_response,
                "base_model_bleu": result.base_model_bleu,
                "finetuned_bleu": result.finetuned_bleu,
                "base_model_rouge": result.base_model_rouge,
                "finetuned_rouge": result.finetuned_rouge,
                "base_model_time": result.base_model_time,
                "finetuned_time": result.finetuned_time,
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
            "finetuned_model": {
                "avg_bleu": np.mean([r.finetuned_bleu for r in results]),
                "avg_rouge1": np.mean([r.finetuned_rouge["rouge1"] for r in results]),
                "avg_rouge2": np.mean([r.finetuned_rouge["rouge2"] for r in results]),
                "avg_rougeL": np.mean([r.finetuned_rouge["rougeL"] for r in results]),
                "avg_time": np.mean([r.finetuned_time for r in results]),
            }
        }
        
        with open(output_path / "summary_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("FLAN-T5 MODEL EVALUATION RESULTS")
        print("="*80)
        
        base_bleu = summary["base_model"]["avg_bleu"]
        finetuned_bleu = summary["finetuned_model"]["avg_bleu"]
        base_rouge1 = summary["base_model"]["avg_rouge1"]
        finetuned_rouge1 = summary["finetuned_model"]["avg_rouge1"]
        
        print(f"{'Model':<20} {'BLEU':<10} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'Time (s)':<10}")
        print("-" * 80)
        print(f"{'Base Flan-T5':<20} {base_bleu:<10.4f} {summary['base_model']['avg_rouge1']:<10.4f} {summary['base_model']['avg_rouge2']:<10.4f} {summary['base_model']['avg_rougeL']:<10.4f} {summary['base_model']['avg_time']:<10.2f}")
        print(f"{'Fine-tuned Flan-T5':<20} {finetuned_bleu:<10.4f} {summary['finetuned_model']['avg_rouge1']:<10.4f} {summary['finetuned_model']['avg_rouge2']:<10.4f} {summary['finetuned_model']['avg_rougeL']:<10.4f} {summary['finetuned_model']['avg_time']:<10.2f}")
        print("="*80)
        
        # Improvement analysis
        if base_bleu > 0:
            bleu_improvement = (finetuned_bleu / base_bleu) if base_bleu > 0 else 0
            print(f"\n📊 BLEU Improvement: {bleu_improvement:.2f}x")
        
        if base_rouge1 > 0:
            rouge_improvement = (finetuned_rouge1 / base_rouge1) if base_rouge1 > 0 else 0
            print(f"📊 ROUGE-1 Improvement: {rouge_improvement:.2f}x")
        
        if finetuned_bleu > base_bleu:
            print("✅ Fine-tuned Flan-T5 performs BETTER than base model!")
        else:
            print("❌ Fine-tuned model still needs improvement")
        
        print(f"\nResults saved to: {output_path}")
        
        return summary

def main():
    """Main Flan-T5 evaluation function"""
    
    # Initialize evaluator
    evaluator = FlanT5ModelEvaluator(
        base_model_path="google/flan-t5-small",
        finetuned_model_path="flan_t5_financial_model"
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Save and display results
    summary = evaluator.save_results(results)
    
    print(f"\n🎯 Flan-T5 evaluation completed! Evaluated {len(results)} samples")

if __name__ == "__main__":
    main()
