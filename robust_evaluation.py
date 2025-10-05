#!/usr/bin/env python3
"""
Robust Evaluation System for GPT-2 Singapore Financial Q&A
Comprehensive metrics including BLEU, ROUGE, semantic similarity, and domain accuracy
"""

import torch
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Core libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import Dataset

# Evaluation metrics
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    print("Installing evaluation dependencies...")
    import subprocess
    subprocess.run(["pip", "install", "rouge-score", "nltk", "sentence-transformers"], check=True)
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)

try:
    from sentence_transformers import SentenceTransformer
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
except:
    semantic_model = None
    print("Semantic similarity evaluation disabled (sentence-transformers not available)")

@dataclass
class EvaluationResult:
    """Structured evaluation results"""
    question: str
    ground_truth: str
    base_response: str
    finetuned_response: str
    bleu_score: float
    rouge_1: float
    rouge_2: float
    rouge_l: float
    semantic_similarity: float
    domain_accuracy: float
    response_time: float
    singapore_content: bool
    factual_accuracy: bool

class RobustEvaluator:
    """Comprehensive evaluation system for Singapore financial Q&A"""
    
    def __init__(self, base_model_name: str = "gpt2", finetuned_model_path: str = None):
        self.base_model_name = base_model_name
        self.finetuned_model_path = finetuned_model_path
        
        # Load models
        print("🔄 Loading models for evaluation...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Base model
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        # Fine-tuned model
        if finetuned_model_path:
            try:
                self.finetuned_model = PeftModel.from_pretrained(
                    AutoModelForCausalLM.from_pretrained(base_model_name),
                    finetuned_model_path
                )
                print(f"✅ Loaded fine-tuned model from {finetuned_model_path}")
            except:
                print("⚠️ Could not load fine-tuned model, using base model for both")
                self.finetuned_model = self.base_model
        else:
            self.finetuned_model = self.base_model
        
        # Move to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model.to(self.device)
        self.finetuned_model.to(self.device)
        
        # Initialize evaluation tools
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        
        # Singapore financial keywords for domain accuracy
        self.singapore_keywords = {
            'organizations': ['mas', 'monetary authority of singapore', 'stro', 'suspicious transaction reporting office'],
            'currency': ['sgd', 'singapore dollar', 'singapore dollars'],
            'regulations': ['pdpa', 'personal data protection act', 'psa', 'payment services act', 
                          'sfa', 'securities and futures act', 'banking act', 'insurance act'],
            'concepts': ['capital ratio', 'cet1', 'common equity tier 1', 'aml', 'anti-money laundering',
                        'cft', 'countering financing terrorism', 'trm', 'technology risk management',
                        'rbc', 'risk-based capital', 'car', 'capital adequacy ratio'],
            'locations': ['singapore', 'republic of singapore'],
            'amounts': ['sgd 1 million', 'sgd 1.5 billion', '6.5%', '10%', '120%']
        }
    
    def create_comprehensive_test_set(self) -> List[Dict[str, str]]:
        """Create comprehensive test dataset with ground truth answers"""
        
        test_data = [
            # Basic Singapore Finance
            {
                "question": "What does MAS stand for?",
                "ground_truth": "MAS stands for Monetary Authority of Singapore, which is Singapore's central bank and integrated financial regulator."
            },
            {
                "question": "What currency does Singapore use?",
                "ground_truth": "Singapore uses the Singapore Dollar (SGD) as its official currency."
            },
            {
                "question": "Who regulates banks in Singapore?",
                "ground_truth": "The Monetary Authority of Singapore (MAS) regulates banks in Singapore."
            },
            
            # Capital Adequacy
            {
                "question": "What are the minimum capital requirements for banks in Singapore?",
                "ground_truth": "Banks in Singapore must maintain a minimum Common Equity Tier 1 (CET1) capital ratio of 6.5% and a Total Capital Ratio of 10% as required by MAS."
            },
            {
                "question": "How often must banks report capital adequacy to MAS?",
                "ground_truth": "Banks must submit capital adequacy returns to MAS on a monthly basis."
            },
            
            # AML/CFT
            {
                "question": "What is STRO and what does it do?",
                "ground_truth": "STRO is the Suspicious Transaction Reporting Office, which receives and analyzes suspicious transaction reports from financial institutions in Singapore."
            },
            {
                "question": "What are the AML reporting requirements for financial institutions?",
                "ground_truth": "Financial institutions must report suspicious transactions to STRO within 15 days, regardless of the transaction amount."
            },
            
            # Payment Services
            {
                "question": "What is the minimum capital requirement for major payment institutions?",
                "ground_truth": "Major payment institutions must maintain minimum base capital of SGD 1 million under the Payment Services Act."
            },
            {
                "question": "What does PSA stand for in Singapore financial regulation?",
                "ground_truth": "PSA stands for Payment Services Act, which is Singapore's regulatory framework for payment services."
            },
            
            # Cybersecurity
            {
                "question": "How often must banks conduct penetration testing?",
                "ground_truth": "Banks must conduct penetration testing of critical systems at least annually as required by MAS Technology Risk Management Guidelines."
            },
            {
                "question": "What are the cyber incident reporting requirements?",
                "ground_truth": "Financial institutions must report significant cyber incidents to MAS within 1 hour of discovery."
            },
            
            # Data Protection
            {
                "question": "What does PDPA stand for and how does it apply to banks?",
                "ground_truth": "PDPA stands for Personal Data Protection Act. Banks must comply with PDPA requirements including obtaining consent for data collection and notifying individuals of data breaches within 72 hours."
            },
            
            # Digital Banking
            {
                "question": "What is the minimum capital requirement for digital banks?",
                "ground_truth": "Digital banks must meet minimum paid-up capital of SGD 1.5 billion to obtain a banking license from MAS."
            },
            
            # Insurance
            {
                "question": "What is the minimum Capital Adequacy Ratio for insurers?",
                "ground_truth": "Insurers in Singapore must maintain a minimum Capital Adequacy Ratio (CAR) of 120% under MAS's Risk-Based Capital framework."
            },
            
            # Securities
            {
                "question": "What does SFA stand for in Singapore?",
                "ground_truth": "SFA stands for Securities and Futures Act, which governs Singapore's capital markets and requires licensing for securities activities."
            }
        ]
        
        return test_data
    
    def generate_response(self, model, question: str, max_tokens: int = 50) -> Tuple[str, float]:
        """Generate response and measure inference time"""
        prompt = f"Q: {question} A:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        start_time = time.time()
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=1.0,
                top_p=0.9
            )
        end_time = time.time()
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the answer part
        if " A:" in response:
            response = response.split(" A:", 1)[1].strip()
        
        return response, end_time - start_time
    
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score"""
        try:
            reference_tokens = reference.lower().split()
            candidate_tokens = candidate.lower().split()
            
            if len(candidate_tokens) == 0:
                return 0.0
            
            score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=self.smoothing)
            return score
        except:
            return 0.0
    
    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_semantic_similarity(self, reference: str, candidate: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        if semantic_model is None:
            return 0.0
        
        try:
            embeddings = semantic_model.encode([reference, candidate])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except:
            return 0.0
    
    def calculate_domain_accuracy(self, response: str) -> Tuple[float, bool]:
        """Calculate Singapore financial domain accuracy"""
        response_lower = response.lower()
        
        # Check for Singapore-specific content
        singapore_content = False
        keyword_matches = 0
        total_categories = len(self.singapore_keywords)
        
        for category, keywords in self.singapore_keywords.items():
            category_match = any(keyword in response_lower for keyword in keywords)
            if category_match:
                keyword_matches += 1
                singapore_content = True
        
        domain_score = keyword_matches / total_categories
        return domain_score, singapore_content
    
    def assess_factual_accuracy(self, question: str, response: str, ground_truth: str) -> bool:
        """Assess factual accuracy based on key facts"""
        response_lower = response.lower()
        ground_truth_lower = ground_truth.lower()
        
        # Extract key facts from ground truth
        key_facts = []
        
        if "mas" in question.lower():
            key_facts = ["monetary authority", "singapore", "central bank", "regulator"]
        elif "currency" in question.lower() or "sgd" in question.lower():
            key_facts = ["singapore dollar", "sgd"]
        elif "capital" in question.lower():
            key_facts = ["6.5%", "10%", "cet1", "capital ratio"]
        elif "aml" in question.lower():
            key_facts = ["suspicious transaction", "stro", "15 days"]
        elif "payment" in question.lower():
            key_facts = ["sgd 1 million", "payment services act", "psa"]
        elif "cyber" in question.lower() or "penetration" in question.lower():
            key_facts = ["annually", "trm", "1 hour"]
        elif "pdpa" in question.lower():
            key_facts = ["personal data protection", "72 hours", "consent"]
        elif "digital bank" in question.lower():
            key_facts = ["sgd 1.5 billion", "license"]
        elif "insurer" in question.lower() or "car" in question.lower():
            key_facts = ["120%", "capital adequacy ratio"]
        elif "sfa" in question.lower():
            key_facts = ["securities and futures act", "capital markets"]
        
        if not key_facts:
            return True  # No specific facts to check
        
        # Check if at least 50% of key facts are present
        matches = sum(1 for fact in key_facts if fact in response_lower)
        return matches >= len(key_facts) * 0.5
    
    def evaluate_single_question(self, question: str, ground_truth: str) -> EvaluationResult:
        """Evaluate a single question comprehensively"""
        
        # Generate responses
        base_response, base_time = self.generate_response(self.base_model, question)
        finetuned_response, ft_time = self.generate_response(self.finetuned_model, question)
        
        # Calculate metrics
        bleu_score = self.calculate_bleu_score(ground_truth, finetuned_response)
        rouge_scores = self.calculate_rouge_scores(ground_truth, finetuned_response)
        semantic_sim = self.calculate_semantic_similarity(ground_truth, finetuned_response)
        domain_score, singapore_content = self.calculate_domain_accuracy(finetuned_response)
        factual_accuracy = self.assess_factual_accuracy(question, finetuned_response, ground_truth)
        
        return EvaluationResult(
            question=question,
            ground_truth=ground_truth,
            base_response=base_response,
            finetuned_response=finetuned_response,
            bleu_score=bleu_score,
            rouge_1=rouge_scores['rouge1'],
            rouge_2=rouge_scores['rouge2'],
            rouge_l=rouge_scores['rougeL'],
            semantic_similarity=semantic_sim,
            domain_accuracy=domain_score,
            response_time=ft_time,
            singapore_content=singapore_content,
            factual_accuracy=factual_accuracy
        )
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation on full test set"""
        
        print("🧪 COMPREHENSIVE EVALUATION - GPT-2 Singapore Financial Q&A")
        print("=" * 80)
        
        test_data = self.create_comprehensive_test_set()
        results = []
        
        print(f"📊 Evaluating {len(test_data)} questions with multiple metrics...")
        
        for i, item in enumerate(test_data, 1):
            print(f"\n{i}/{len(test_data)}: {item['question']}")
            
            result = self.evaluate_single_question(item['question'], item['ground_truth'])
            results.append(result)
            
            # Show key metrics
            print(f"   BLEU: {result.bleu_score:.3f} | ROUGE-L: {result.rouge_l:.3f} | "
                  f"Domain: {result.domain_accuracy:.3f} | Factual: {'✅' if result.factual_accuracy else '❌'}")
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(results)
        
        # Display comprehensive results
        self.display_comprehensive_results(results, aggregate_metrics)
        
        # Save detailed results
        self.save_evaluation_results(results, aggregate_metrics)
        
        return {
            'individual_results': results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate aggregate metrics across all results"""
        
        metrics = {
            'avg_bleu': np.mean([r.bleu_score for r in results]),
            'avg_rouge_1': np.mean([r.rouge_1 for r in results]),
            'avg_rouge_2': np.mean([r.rouge_2 for r in results]),
            'avg_rouge_l': np.mean([r.rouge_l for r in results]),
            'avg_semantic_similarity': np.mean([r.semantic_similarity for r in results]),
            'avg_domain_accuracy': np.mean([r.domain_accuracy for r in results]),
            'avg_response_time': np.mean([r.response_time for r in results]),
            'singapore_content_rate': np.mean([r.singapore_content for r in results]),
            'factual_accuracy_rate': np.mean([r.factual_accuracy for r in results]),
            'total_questions': len(results)
        }
        
        return metrics
    
    def display_comprehensive_results(self, results: List[EvaluationResult], metrics: Dict[str, float]):
        """Display comprehensive evaluation results"""
        
        print(f"\n" + "=" * 80)
        print("🎯 COMPREHENSIVE EVALUATION RESULTS")
        print("=" * 80)
        
        print(f"\n📊 AGGREGATE METRICS ({metrics['total_questions']} questions):")
        print(f"   BLEU Score:           {metrics['avg_bleu']:.4f}")
        print(f"   ROUGE-1:              {metrics['avg_rouge_1']:.4f}")
        print(f"   ROUGE-2:              {metrics['avg_rouge_2']:.4f}")
        print(f"   ROUGE-L:              {metrics['avg_rouge_l']:.4f}")
        print(f"   Semantic Similarity:  {metrics['avg_semantic_similarity']:.4f}")
        print(f"   Domain Accuracy:      {metrics['avg_domain_accuracy']:.4f}")
        print(f"   Factual Accuracy:     {metrics['factual_accuracy_rate']:.1%}")
        print(f"   Singapore Content:    {metrics['singapore_content_rate']:.1%}")
        print(f"   Avg Response Time:    {metrics['avg_response_time']:.3f}s")
        
        print(f"\n🏆 PERFORMANCE ASSESSMENT:")
        if metrics['factual_accuracy_rate'] >= 0.8 and metrics['avg_domain_accuracy'] >= 0.3:
            print("   🎉 EXCELLENT: Production-ready performance!")
        elif metrics['factual_accuracy_rate'] >= 0.6 and metrics['avg_domain_accuracy'] >= 0.2:
            print("   ✅ GOOD: Strong performance with room for improvement")
        elif metrics['factual_accuracy_rate'] >= 0.4:
            print("   ⚠️ MODERATE: Shows promise but needs optimization")
        else:
            print("   ❌ POOR: Requires significant improvement")
        
        print(f"\n📈 DETAILED BREAKDOWN:")
        print(f"   Questions with Singapore content: {sum(r.singapore_content for r in results)}/{len(results)}")
        print(f"   Factually accurate responses: {sum(r.factual_accuracy for r in results)}/{len(results)}")
        print(f"   High BLEU scores (>0.3): {sum(r.bleu_score > 0.3 for r in results)}/{len(results)}")
        print(f"   High domain accuracy (>0.5): {sum(r.domain_accuracy > 0.5 for r in results)}/{len(results)}")
    
    def save_evaluation_results(self, results: List[EvaluationResult], metrics: Dict[str, float]):
        """Save detailed evaluation results to files"""
        
        # Create results directory
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save aggregate metrics
        with open(results_dir / "aggregate_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save detailed results
        detailed_results = []
        for result in results:
            detailed_results.append({
                'question': result.question,
                'ground_truth': result.ground_truth,
                'base_response': result.base_response,
                'finetuned_response': result.finetuned_response,
                'metrics': {
                    'bleu_score': result.bleu_score,
                    'rouge_1': result.rouge_1,
                    'rouge_2': result.rouge_2,
                    'rouge_l': result.rouge_l,
                    'semantic_similarity': result.semantic_similarity,
                    'domain_accuracy': result.domain_accuracy,
                    'response_time': result.response_time,
                    'singapore_content': result.singapore_content,
                    'factual_accuracy': result.factual_accuracy
                }
            })
        
        with open(results_dir / "detailed_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to {results_dir}/")
        print(f"   • aggregate_metrics.json - Summary statistics")
        print(f"   • detailed_results.json - Individual question results")

def main():
    """Run comprehensive evaluation"""
    
    # Initialize evaluator
    evaluator = RobustEvaluator(
        base_model_name="gpt2",
        finetuned_model_path="gpt2_singapore_production/lora_adapters"  # Adjust path as needed
    )
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    print(f"\n✅ Comprehensive evaluation completed!")
    print(f"🎯 Check evaluation_results/ for detailed analysis")

if __name__ == "__main__":
    main()
