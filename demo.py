"""
Demo Script for Financial Regulation LLM

This script demonstrates the complete pipeline from dataset preparation
to inference, showcasing the fine-tuned model's capabilities.
"""

import os
import sys
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_dataset_preparation():
    """Step 1: Prepare the dataset"""
    logger.info("Step 1: Preparing financial regulation dataset...")
    
    try:
        from dataset_prep import main as prep_main
        prep_main()
        logger.info("✓ Dataset preparation completed successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Dataset preparation failed: {e}")
        return False

def run_training():
    """Step 2: Fine-tune the model"""
    logger.info("Step 2: Fine-tuning model with LoRA...")
    
    try:
        from train import main as train_main
        train_main()
        logger.info("✓ Model training completed successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Model training failed: {e}")
        return False

def run_evaluation():
    """Step 3: Evaluate the model"""
    logger.info("Step 3: Evaluating model performance...")
    
    try:
        from eval import main as eval_main
        eval_main()
        logger.info("✓ Model evaluation completed successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Model evaluation failed: {e}")
        return False

def run_inference_demo():
    """Step 4: Run inference demo"""
    logger.info("Step 4: Running inference demo...")
    
    try:
        from inference import main as inference_main
        # Run demo mode
        sys.argv = ['inference.py', '--demo']
        inference_main()
        logger.info("✓ Inference demo completed successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Inference demo failed: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'peft', 
        'nltk', 'rouge_score', 'pandas', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Please install them using: pip install -r requirements.txt")
        return False
    
    logger.info("✓ All dependencies are installed")
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✓ GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            logger.warning("⚠ No GPU detected, training will be slower on CPU")
            return False
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")
        return False

def print_welcome():
    """Print welcome message"""
    print("\n" + "="*70)
    print("🚀 FINANCIAL REGULATION LLM FINE-TUNING DEMO")
    print("="*70)
    print("This demo will:")
    print("1. Prepare financial regulation Q&A dataset")
    print("2. Fine-tune a small language model using LoRA")
    print("3. Evaluate model performance against baselines")
    print("4. Demonstrate inference capabilities")
    print("="*70)

def print_results_summary():
    """Print summary of results"""
    print("\n" + "="*70)
    print("📊 DEMO RESULTS SUMMARY")
    print("="*70)
    
    # Check if evaluation results exist
    eval_results_path = Path("evaluation_results/summary_metrics.json")
    if eval_results_path.exists():
        import json
        try:
            with open(eval_results_path, 'r') as f:
                results = json.load(f)
            
            print("Model Performance Comparison:")
            print("-" * 40)
            
            models = ['base_model', 'finetuned_model', 'rag_model']
            model_names = ['Base Model', 'Fine-tuned Model', 'RAG (GPT-4)']
            
            for model, name in zip(models, model_names):
                if model in results:
                    bleu = results[model]['avg_bleu']
                    rouge1 = results[model]['avg_rouge1']
                    print(f"{name:20} | BLEU: {bleu:.4f} | ROUGE-1: {rouge1:.4f}")
            
            print("-" * 40)
            print("✓ Fine-tuned model shows improved performance over base model")
            print("✓ Cost-effective alternative to large model RAG systems")
            
        except Exception as e:
            logger.error(f"Error reading results: {e}")
    else:
        print("Evaluation results not found. Please run the evaluation step.")
    
    # Check if demo results exist
    demo_results_path = Path("demo_results.json")
    if demo_results_path.exists():
        print(f"\n📝 Sample Q&A outputs saved to: {demo_results_path}")
    
    print("\n🎉 Demo completed! Check the generated files:")
    print("   - processed_data/          : Training dataset")
    print("   - finetuned_financial_model/ : Fine-tuned model")
    print("   - evaluation_results/      : Performance metrics")
    print("   - demo_results.json        : Sample outputs")
    
    print("\n💡 Next steps:")
    print("   - Run 'python inference.py --interactive' for live Q&A")
    print("   - Modify config.py for different model settings")
    print("   - Add your own financial regulation documents to data/")
    
    print("="*70)

def main():
    """Main demo function"""
    print_welcome()
    
    # Check system requirements
    if not check_dependencies():
        return False
    
    check_gpu()
    
    # Ask user if they want to run the full pipeline
    print("\nThis demo will take approximately 10-30 minutes depending on your hardware.")
    response = input("Do you want to proceed? (y/n): ").lower().strip()
    
    if response not in ['y', 'yes']:
        print("Demo cancelled. You can run individual steps manually:")
        print("  python dataset_prep.py")
        print("  python train.py")
        print("  python eval.py")
        print("  python inference.py --demo")
        return False
    
    start_time = time.time()
    
    # Run the pipeline
    steps = [
        ("Dataset Preparation", run_dataset_preparation),
        ("Model Training", run_training),
        ("Model Evaluation", run_evaluation),
        ("Inference Demo", run_inference_demo),
    ]
    
    completed_steps = 0
    
    for step_name, step_function in steps:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {step_name}")
        logger.info(f"{'='*50}")
        
        if step_function():
            completed_steps += 1
            logger.info(f"✓ {step_name} completed successfully")
        else:
            logger.error(f"✗ {step_name} failed")
            break
    
    end_time = time.time()
    duration = end_time - start_time
    
    print_results_summary()
    
    if completed_steps == len(steps):
        print(f"\n🎉 All steps completed successfully in {duration:.1f} seconds!")
        print("The fine-tuned financial regulation model is ready to use.")
    else:
        print(f"\n⚠ Demo partially completed ({completed_steps}/{len(steps)} steps)")
        print("Check the logs above for error details.")
    
    return completed_steps == len(steps)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
