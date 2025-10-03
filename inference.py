"""
Inference Script for Fine-tuned Financial Regulation LLM

This script provides an easy interface to load and query the fine-tuned model
for Singapore financial regulation questions.
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

class FinancialRegulationInference:
    """Inference class for the fine-tuned financial regulation model"""
    
    def __init__(self, 
                 base_model_path: str = "microsoft/DialoGPT-medium",
                 finetuned_model_path: str = "finetuned_financial_model",
                 device: Optional[str] = None):
        
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
        
        # Load model and tokenizer
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading fine-tuned model from: {self.finetuned_model_path}")
        
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
        
        # Load LoRA adapters if they exist
        lora_path = Path(self.finetuned_model_path) / "lora_adapters"
        if lora_path.exists():
            logger.info("Loading LoRA adapters...")
            self.model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            logger.info("No LoRA adapters found, using base model...")
            self.model = base_model
        
        self.model.eval()
        logger.info("Model loaded successfully!")
    
    def create_prompt(self, question: str) -> str:
        """Create a properly formatted prompt for the model"""
        prompt = f"### Instruction:\nAnswer the following question about Singapore financial regulations:\n\n### Input:\n{question}\n\n### Response:\n"
        return prompt
    
    def generate_response(self, 
                         question: str, 
                         max_length: int = 300,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         do_sample: bool = True) -> str:
        """Generate a response to a financial regulation question"""
        
        # Create prompt
        prompt = self.create_prompt(question)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        # Move to device
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
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
                repetition_penalty=1.1,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    
    def batch_inference(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple questions in batch"""
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            
            try:
                response = self.generate_response(question, **kwargs)
                results.append({
                    "question": question,
                    "response": response,
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                results.append({
                    "question": question,
                    "response": f"Error: {str(e)}",
                    "status": "error"
                })
        
        return results
    
    def interactive_mode(self):
        """Run interactive Q&A mode"""
        print("\n" + "="*60)
        print("Financial Regulation Q&A Assistant")
        print("Ask questions about Singapore financial regulations")
        print("Type 'quit' to exit")
        print("="*60)
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\nGenerating response...")
                response = self.generate_response(question)
                print(f"\nAnswer: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def load_sample_questions(self, sample_file: str = "sample_questions.json") -> List[str]:
        """Load sample questions for demonstration"""
        
        sample_questions = [
            "What is MAS's position on the use of artificial intelligence in financial advisory services?",
            "What are the capital adequacy requirements for banks in Singapore?",
            "How should financial institutions implement anti-money laundering measures?",
            "What are the data protection requirements for financial institutions under the PDPA?",
            "What cybersecurity requirements must financial institutions meet?",
            "How does MAS regulate digital payment services?",
            "What are the key requirements for robo-advisory services in Singapore?",
            "What compliance reporting requirements do banks have under MAS regulations?"
        ]
        
        # Save sample questions to file
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_questions, f, indent=2, ensure_ascii=False)
        
        return sample_questions

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="Financial Regulation LLM Inference")
    parser.add_argument("--model_path", type=str, default="finetuned_financial_model",
                       help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, default="microsoft/DialoGPT-medium",
                       help="Base model path")
    parser.add_argument("--question", type=str, default=None,
                       help="Single question to ask")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo with sample questions")
    parser.add_argument("--max_length", type=int, default=300,
                       help="Maximum response length")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    try:
        inference_engine = FinancialRegulationInference(
            base_model_path=args.base_model,
            finetuned_model_path=args.model_path
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        print("Note: Make sure you have run the training script first!")
        return
    
    if args.question:
        # Single question mode
        print(f"Question: {args.question}")
        print("\nGenerating response...")
        response = inference_engine.generate_response(
            args.question,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print(f"\nAnswer: {response}")
    
    elif args.interactive:
        # Interactive mode
        inference_engine.interactive_mode()
    
    elif args.demo:
        # Demo mode with sample questions
        print("Running demo with sample questions...")
        sample_questions = inference_engine.load_sample_questions()
        
        results = inference_engine.batch_inference(
            sample_questions,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        print("\n" + "="*80)
        print("DEMO RESULTS")
        print("="*80)
        
        for i, result in enumerate(results):
            print(f"\n{i+1}. Question: {result['question']}")
            print(f"   Answer: {result['response']}")
            print("-" * 80)
        
        # Save demo results
        with open("demo_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDemo results saved to: demo_results.json")
    
    else:
        # Default: show help
        parser.print_help()

if __name__ == "__main__":
    main()
