"""
Flan-T5-BASE Evaluation Script
Compares Base Flan-T5-BASE vs Fine-tuned Flan-T5-BASE
"""

import json
import os
import time
import torch
from typing import List, Dict, Any
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

class FlanT5BaseEvaluator:
    """Evaluator for Flan-T5-BASE models"""
    
    def __init__(self, 
                 base_model_path: str = "google/flan-t5-base",
                 finetuned_model_path: str = "flan_t5_base_financial_model"):
        
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
        
    def load_models(self):
        """Load both base and fine-tuned models"""
        print("🤖 Loading Flan-T5-BASE models...")
        
        # Load base model
        print("Loading base Flan-T5-BASE...")
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        # Load fine-tuned model
        print("Loading fine-tuned Flan-T5-BASE...")
        try:
            # Try to load LoRA adapters first
            lora_path = Path(self.finetuned_model_path) / "lora_adapters"
            if lora_path.exists():
                print("Loading LoRA adapters...")
                base_model_copy = AutoModelForSeq2SeqLM.from_pretrained(
                    self.base_model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                self.finetuned_model = PeftModel.from_pretrained(base_model_copy, lora_path)
                self.finetuned_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            else:
                print("Loading full fine-tuned model...")
                self.finetuned_tokenizer = AutoTokenizer.from_pretrained(self.finetuned_model_path)
                self.finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.finetuned_model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                
        except Exception as e:
            print(f"⚠️ Error loading fine-tuned model: {e}")
            print("Using base model as fallback...")
            self.finetuned_model = self.base_model
            self.finetuned_tokenizer = self.base_tokenizer
    
    def generate_response(self, model, tokenizer, question: str, max_new_tokens: int = 100) -> str:
        """Generate response using Flan-T5-BASE"""
        
        input_prompt = f"Answer this financial regulation question: {question}"
        
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
                num_beams=3,
                do_sample=False,
                early_stopping=True,
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
    
    def evaluate_questions(self, test_questions: List[Dict]) -> List[Dict]:
        """Evaluate both models on test questions"""
        
        results = []
        
        print(f"\n📊 Evaluating {len(test_questions)} questions...")
        print("="*80)
        
        for i, item in enumerate(test_questions):
            question = item['question']
            ground_truth = item['answer']
            
            print(f"\n{i+1}. Question: {question}")
            print(f"Ground Truth: {ground_truth[:100]}...")
            
            # Base model response
            start_time = time.time()
            base_response = self.generate_response(self.base_model, self.base_tokenizer, question)
            base_time = time.time() - start_time
            
            # Fine-tuned model response
            start_time = time.time()
            ft_response = self.generate_response(self.finetuned_model, self.finetuned_tokenizer, question)
            ft_time = time.time() - start_time
            
            print(f"Base Model: {base_response}")
            print(f"Fine-tuned: {ft_response}")
            
            # Calculate metrics
            base_bleu = self.calculate_bleu(ground_truth, base_response)
            ft_bleu = self.calculate_bleu(ground_truth, ft_response)
            
            base_rouge = self.calculate_rouge(ground_truth, base_response)
            ft_rouge = self.calculate_rouge(ground_truth, ft_response)
            
            print(f"BLEU: Base={base_bleu:.4f}, Fine-tuned={ft_bleu:.4f}")
            print(f"ROUGE-1: Base={base_rouge['rouge1']:.4f}, Fine-tuned={ft_rouge['rouge1']:.4f}")
            
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "base_response": base_response,
                "finetuned_response": ft_response,
                "base_bleu": base_bleu,
                "finetuned_bleu": ft_bleu,
                "base_rouge": base_rouge,
                "finetuned_rouge": ft_rouge,
                "base_time": base_time,
                "finetuned_time": ft_time,
            })
            
            print("-" * 80)
        
        return results
    
    def save_and_display_results(self, results: List[Dict]):
        """Save results and display summary"""
        
        # Calculate averages
        base_avg_bleu = np.mean([r['base_bleu'] for r in results])
        ft_avg_bleu = np.mean([r['finetuned_bleu'] for r in results])
        
        base_avg_rouge1 = np.mean([r['base_rouge']['rouge1'] for r in results])
        ft_avg_rouge1 = np.mean([r['finetuned_rouge']['rouge1'] for r in results])
        
        base_avg_time = np.mean([r['base_time'] for r in results])
        ft_avg_time = np.mean([r['finetuned_time'] for r in results])
        
        # Display results
        print("\n" + "="*80)
        print("FLAN-T5-BASE EVALUATION RESULTS")
        print("="*80)
        
        print(f"{'Model':<25} {'BLEU':<10} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'Time (s)':<10}")
        print("-" * 80)
        
        base_avg_rouge2 = np.mean([r['base_rouge']['rouge2'] for r in results])
        base_avg_rougeL = np.mean([r['base_rouge']['rougeL'] for r in results])
        ft_avg_rouge2 = np.mean([r['finetuned_rouge']['rouge2'] for r in results])
        ft_avg_rougeL = np.mean([r['finetuned_rouge']['rougeL'] for r in results])
        
        print(f"{'Base Flan-T5-BASE':<25} {base_avg_bleu:<10.4f} {base_avg_rouge1:<10.4f} {base_avg_rouge2:<10.4f} {base_avg_rougeL:<10.4f} {base_avg_time:<10.2f}")
        print(f"{'Fine-tuned Flan-T5-BASE':<25} {ft_avg_bleu:<10.4f} {ft_avg_rouge1:<10.4f} {ft_avg_rouge2:<10.4f} {ft_avg_rougeL:<10.4f} {ft_avg_time:<10.2f}")
        
        print("="*80)
        
        # Improvement analysis
        if base_avg_bleu > 0:
            bleu_improvement = (ft_avg_bleu / base_avg_bleu)
            print(f"\n📊 BLEU Improvement: {bleu_improvement:.2f}x")
        
        if base_avg_rouge1 > 0:
            rouge_improvement = (ft_avg_rouge1 / base_avg_rouge1)
            print(f"📊 ROUGE-1 Improvement: {rouge_improvement:.2f}x")
        
        if ft_avg_bleu > base_avg_bleu:
            print("✅ Fine-tuned model performs BETTER than base model!")
        else:
            print("❌ Fine-tuned model needs more work")
        
        # Save detailed results
        output_dir = Path("flan_t5_base_evaluation_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        summary = {
            "base_model": {
                "avg_bleu": base_avg_bleu,
                "avg_rouge1": base_avg_rouge1,
                "avg_rouge2": base_avg_rouge2,
                "avg_rougeL": base_avg_rougeL,
                "avg_time": base_avg_time,
            },
            "finetuned_model": {
                "avg_bleu": ft_avg_bleu,
                "avg_rouge1": ft_avg_rouge1,
                "avg_rouge2": ft_avg_rouge2,
                "avg_rougeL": ft_avg_rougeL,
                "avg_time": ft_avg_time,
            }
        }
        
        with open(output_dir / "summary_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        return summary

def main():
    """Main evaluation function"""
    
    # Load test questions
    try:
        with open("processed_data/holdout_test_set.json", 'r', encoding='utf-8') as f:
            test_questions = json.load(f)
    except FileNotFoundError:
        # Fallback test questions
        test_questions = [
            {
                "question": "What are the capital adequacy requirements for banks in Singapore?",
                "answer": "Banks in Singapore must maintain minimum capital ratios as per MAS requirements and Basel III standards."
            },
            {
                "question": "What is MAS's position on AI in financial advisory services?",
                "answer": "MAS supports the responsible use of AI in financial advisory services while ensuring adequate safeguards."
            },
            {
                "question": "What are the cybersecurity requirements for financial institutions?",
                "answer": "Financial institutions must implement robust cybersecurity frameworks including risk assessments and incident response procedures."
            }
        ]
    
    # Initialize evaluator
    evaluator = FlanT5BaseEvaluator()
    
    # Load models
    evaluator.load_models()
    
    # Run evaluation
    results = evaluator.evaluate_questions(test_questions)
    
    # Display results
    summary = evaluator.save_and_display_results(results)
    
    print(f"\n🎯 Evaluation completed! Evaluated {len(results)} samples")

if __name__ == "__main__":
    main()
