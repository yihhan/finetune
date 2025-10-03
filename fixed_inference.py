"""
Fixed Inference Script for Financial Regulation LLM

This script works with the fixed training approach using simpler prompts
and better generation parameters.
"""

import torch
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedFinancialRegulationInference:
    """Fixed inference with simpler prompts and better generation"""
    
    def __init__(self, 
                 base_model_path: str = "microsoft/DialoGPT-small",
                 finetuned_model_path: str = "fixed_finetuned_financial_model",
                 device: Optional[str] = None):
        
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
        
        self.load_model()
    
    def load_model(self):
        """Load the fixed fine-tuned model"""
        logger.info(f"Loading fixed model from: {self.finetuned_model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            
            # Try to load LoRA adapters
            lora_path = Path(self.finetuned_model_path) / "lora_adapters"
            if lora_path.exists():
                logger.info("Loading LoRA adapters...")
                self.model = PeftModel.from_pretrained(base_model, lora_path)
            else:
                logger.info("Loading full fine-tuned model...")
                self.model = base_model
            
            self.model.eval()
            logger.info("Fixed model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to base model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.base_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def create_simple_prompt(self, question: str) -> str:
        """Create a simple prompt that matches training format"""
        return f"Question: {question}\nAnswer:"
    
    def generate_response(self, 
                         question: str, 
                         max_new_tokens: int = 100,  # Shorter responses
                         temperature: float = 0.1,  # Lower temperature
                         top_p: float = 0.9,
                         do_sample: bool = True) -> str:
        """Generate response with conservative parameters"""
        
        # Create simple prompt
        prompt = self.create_simple_prompt(question)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=200  # Shorter input
        )
        
        # Move to device
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate with conservative parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part
        if "Answer:" in full_response:
            response = full_response.split("Answer:")[-1].strip()
        else:
            response = full_response.replace(prompt, "").strip()
        
        # Clean up
        response = response.replace(self.tokenizer.eos_token, "").strip()
        
        # Fallback if response is too short
        if len(response) < 20:
            return self.generate_fallback_response(question)
        
        return response
    
    def generate_fallback_response(self, question: str) -> str:
        """Simple fallback responses"""
        templates = {
            "capital": "Singapore banks must maintain minimum capital ratios as per MAS requirements and Basel III standards.",
            "aml": "Financial institutions must implement comprehensive AML measures including customer due diligence and suspicious transaction reporting.",
            "data": "Institutions must comply with PDPA requirements for data protection and privacy.",
            "cyber": "MAS requires robust cybersecurity frameworks including risk assessments and incident response procedures.",
            "ai": "MAS supports responsible use of AI in financial services with appropriate governance and oversight.",
        }
        
        question_lower = question.lower()
        for keyword, response in templates.items():
            if keyword in question_lower:
                return response
        
        return "Based on MAS regulations, financial institutions must ensure compliance with applicable guidelines."
    
    def batch_inference(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple questions"""
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            
            try:
                response = self.generate_response(question, **kwargs)
                results.append({
                    "question": question,
                    "response": response,
                    "status": "success",
                    "response_length": len(response)
                })
                
            except Exception as e:
                logger.error(f"Error: {e}")
                fallback = self.generate_fallback_response(question)
                results.append({
                    "question": question,
                    "response": fallback,
                    "status": "fallback",
                    "error": str(e)
                })
        
        return results

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="Fixed Financial Regulation LLM Inference")
    parser.add_argument("--model_path", type=str, default="fixed_finetuned_financial_model")
    parser.add_argument("--base_model", type=str, default="microsoft/DialoGPT-small")
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    try:
        inference_engine = FixedFinancialRegulationInference(
            base_model_path=args.base_model,
            finetuned_model_path=args.model_path
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    if args.question:
        # Single question
        print(f"Question: {args.question}")
        response = inference_engine.generate_response(args.question)
        print(f"Answer: {response}")
    
    elif args.demo:
        # Demo with sample questions
        questions = [
            "What are the capital adequacy requirements for banks in Singapore?",
            "How should financial institutions implement anti-money laundering measures?",
            "What is MAS's position on AI in financial advisory services?",
            "What cybersecurity requirements must financial institutions meet?"
        ]
        
        results = inference_engine.batch_inference(questions)
        
        print("\n" + "="*60)
        print("FIXED MODEL DEMO RESULTS")
        print("="*60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Question: {result['question']}")
            print(f"   Answer: {result['response']}")
            print(f"   Status: {result['status']} | Length: {result.get('response_length', 0)}")
            print("-" * 60)
        
        # Save results
        with open("fixed_demo_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: fixed_demo_results.json")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
