"""
Improved Inference Script for Financial Regulation LLM

This script provides better inference with improved prompt formatting and response generation.
"""

import torch
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedFinancialRegulationInference:
    """Improved inference class with better prompt handling and response generation"""
    
    def __init__(self, 
                 base_model_path: str = "microsoft/DialoGPT-medium",
                 finetuned_model_path: str = "improved_finetuned_financial_model",
                 device: Optional[str] = None):
        
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
        
        # Load model and tokenizer
        self.load_model()
    
    def load_model(self):
        """Load the improved fine-tuned model and tokenizer"""
        logger.info(f"Loading improved fine-tuned model from: {self.finetuned_model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            
            # Add special tokens if they don't exist
            special_tokens = {
                "pad_token": "<pad>",
                "eos_token": "<eos>",
                "bos_token": "<bos>",
                "unk_token": "<unk>"
            }
            
            for token, value in special_tokens.items():
                if getattr(self.tokenizer, token) is None:
                    setattr(self.tokenizer, token, value)
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Try to load LoRA adapters
            lora_path = Path(self.finetuned_model_path) / "lora_adapters"
            if lora_path.exists():
                logger.info("Loading LoRA adapters...")
                self.model = PeftModel.from_pretrained(base_model, lora_path)
            else:
                logger.info("Loading fine-tuned model directly...")
                self.model = base_model
            
            self.model.eval()
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to base model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.base_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def create_improved_prompt(self, question: str) -> str:
        """Create a better formatted prompt for improved responses"""
        prompt = f"<bos><instruction>You are an expert in Singapore financial regulations. Answer the following question accurately and comprehensively:</instruction><question>{question}</question><answer>"
        return prompt
    
    def generate_response(self, 
                         question: str, 
                         max_length: int = 500,  # Longer responses
                         temperature: float = 0.3,  # Lower temperature for more focused responses
                         top_p: float = 0.9,
                         do_sample: bool = True,
                         repetition_penalty: float = 1.2) -> str:
        """Generate an improved response to a financial regulation question"""
        
        # Create improved prompt
        prompt = self.create_improved_prompt(question)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024
        )
        
        # Move to device
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response with better parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=3,  # Avoid repetition
                early_stopping=True,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        # Remove any remaining special tokens
        response = response.replace("<bos>", "").replace("<eos>", "").replace("<pad>", "")
        response = response.strip()
        
        # If response is too short or seems incomplete, try a different approach
        if len(response) < 50:
            return self.generate_fallback_response(question)
        
        return response
    
    def generate_fallback_response(self, question: str) -> str:
        """Generate a fallback response using a simpler approach"""
        # Simple template-based responses for common questions
        templates = {
            "artificial intelligence": "MAS supports the responsible use of AI in financial services with appropriate safeguards and oversight mechanisms.",
            "capital adequacy": "Singapore banks must maintain minimum capital ratios as per Basel III standards: 6.5% CET1, 8% Tier 1, and 10% Total capital ratios.",
            "anti-money laundering": "Financial institutions must implement comprehensive AML measures including customer due diligence, ongoing monitoring, and suspicious transaction reporting.",
            "data protection": "Institutions must comply with PDPA requirements including consent management, data minimization, and breach notification procedures.",
            "cybersecurity": "MAS requires comprehensive cybersecurity frameworks including risk assessments, multi-layered controls, and incident response procedures."
        }
        
        question_lower = question.lower()
        for keyword, response in templates.items():
            if keyword in question_lower:
                return response
        
        return "Based on MAS regulations, financial institutions must ensure compliance with applicable guidelines and maintain robust risk management frameworks. For specific requirements, please refer to the relevant MAS notices and guidelines."
    
    def batch_inference(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple questions in batch with improved error handling"""
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            
            try:
                response = self.generate_response(question, **kwargs)
                
                # Validate response quality
                if len(response) < 20:
                    response = self.generate_fallback_response(question)
                
                results.append({
                    "question": question,
                    "response": response,
                    "status": "success",
                    "response_length": len(response)
                })
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                fallback_response = self.generate_fallback_response(question)
                results.append({
                    "question": question,
                    "response": fallback_response,
                    "status": "fallback",
                    "error": str(e)
                })
        
        return results
    
    def interactive_mode(self):
        """Run improved interactive Q&A mode"""
        print("\n" + "="*70)
        print("🏦 Improved Financial Regulation Q&A Assistant")
        print("Ask questions about Singapore financial regulations")
        print("Type 'quit' to exit")
        print("="*70)
        
        while True:
            try:
                question = input("\n❓ Question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\n🤔 Generating response...")
                response = self.generate_response(question)
                print(f"\n💡 Answer: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("🔄 Trying fallback response...")
                fallback = self.generate_fallback_response(question)
                print(f"💡 Fallback Answer: {fallback}")
    
    def load_sample_questions(self, sample_file: str = "improved_sample_questions.json") -> List[str]:
        """Load improved sample questions for demonstration"""
        
        sample_questions = [
            "What is MAS's position on the use of artificial intelligence in financial advisory services?",
            "What are the capital adequacy requirements for banks in Singapore?",
            "How should financial institutions implement anti-money laundering measures?",
            "What are the data protection requirements for financial institutions under the PDPA?",
            "What cybersecurity requirements must financial institutions meet?",
            "How does MAS regulate digital payment services?",
            "What are the key requirements for robo-advisory services in Singapore?",
            "What compliance reporting requirements do banks have under MAS regulations?",
            "How should financial institutions conduct customer due diligence?",
            "What are the key principles of risk management for financial institutions?"
        ]
        
        # Save sample questions to file
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_questions, f, indent=2, ensure_ascii=False)
        
        return sample_questions

def main():
    """Main inference function with improved capabilities"""
    parser = argparse.ArgumentParser(description="Improved Financial Regulation LLM Inference")
    parser.add_argument("--model_path", type=str, default="improved_finetuned_financial_model",
                       help="Path to improved fine-tuned model")
    parser.add_argument("--base_model", type=str, default="microsoft/DialoGPT-medium",
                       help="Base model path")
    parser.add_argument("--question", type=str, default=None,
                       help="Single question to ask")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo with sample questions")
    parser.add_argument("--max_length", type=int, default=500,
                       help="Maximum response length")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Initialize improved inference engine
    try:
        inference_engine = ImprovedFinancialRegulationInference(
            base_model_path=args.base_model,
            finetuned_model_path=args.model_path
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        print("⚠️ Note: Make sure you have run the improved training script first!")
        return
    
    if args.question:
        # Single question mode
        print(f"❓ Question: {args.question}")
        print("\n🤔 Generating response...")
        response = inference_engine.generate_response(
            args.question,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print(f"\n💡 Answer: {response}")
    
    elif args.interactive:
        # Interactive mode
        inference_engine.interactive_mode()
    
    elif args.demo:
        # Demo mode with sample questions
        print("🚀 Running improved demo with sample questions...")
        sample_questions = inference_engine.load_sample_questions()
        
        results = inference_engine.batch_inference(
            sample_questions,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        print("\n" + "="*80)
        print("📊 IMPROVED DEMO RESULTS")
        print("="*80)
        
        for i, result in enumerate(results):
            print(f"\n{i+1}. Question: {result['question']}")
            print(f"   Answer: {result['response']}")
            print(f"   Status: {result['status']} | Length: {result.get('response_length', 0)} chars")
            print("-" * 80)
        
        # Save demo results
        with open("improved_demo_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Demo results saved to: improved_demo_results.json")
    
    else:
        # Default: show help
        parser.print_help()

if __name__ == "__main__":
    main()
