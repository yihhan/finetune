# 🎯 GROUND TRUTH COMPARISON - Full Response Analysis
# Compare base vs fine-tuned vs expected ground truth answers

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time

print("🎯 GROUND TRUTH COMPARISON - FULL RESPONSE ANALYSIS")
print("=" * 100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Optional: semantic similarity model (graceful fallback if unavailable)
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SEM_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
except Exception:
    SEM_MODEL = None

# Load models
print("\n🔄 Loading models...")
try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    base_model = base_model.to(device)
    print(f"✅ Loaded BASE GPT-2 model")
    
    # Load another copy for fine-tuned version
    base_for_peft = AutoModelForCausalLM.from_pretrained("gpt2")
    base_for_peft = base_for_peft.to(device)
    
    # Load fine-tuned model - auto-detect checkpoint
    model_path = "gpt2_comprehensive_singapore_model/checkpoint-1080"
    print(f"🔍 Trying to load from: {model_path}")
    
    try:
        finetuned_model = PeftModel.from_pretrained(base_for_peft, model_path)
        print(f"✅ Loaded FINE-TUNED model from: {model_path}")
    except:
        # Auto-detect latest checkpoint
        import os
        if os.path.exists("gpt2_comprehensive_singapore_model"):
            checkpoints = [d for d in os.listdir("gpt2_comprehensive_singapore_model") if d.startswith("checkpoint-")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
                model_path = f"gpt2_comprehensive_singapore_model/{latest_checkpoint}"
                print(f"🔄 Auto-detected latest checkpoint: {model_path}")
                finetuned_model = PeftModel.from_pretrained(base_for_peft, model_path)
                print(f"✅ Loaded FINE-TUNED model from: {model_path}")
            else:
                raise Exception("No checkpoints found")
        else:
            raise Exception("Model directory not found")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# Test questions with detailed ground truth answers
test_cases = [
    {
        "question": "What does MAS stand for?",
        "ground_truth": "MAS stands for Monetary Authority of Singapore. It is Singapore's central bank and integrated financial regulator, responsible for conducting monetary policy, supervising financial institutions, developing financial markets, and maintaining financial stability in Singapore."
    },
    {
        "question": "What currency does Singapore use?",
        "ground_truth": "Singapore uses the Singapore Dollar (SGD) as its official currency. The Singapore Dollar is managed by the Monetary Authority of Singapore and is one of the most stable currencies in Asia."
    },
    {
        "question": "Who regulates banks in Singapore?",
        "ground_truth": "The Monetary Authority of Singapore (MAS) regulates banks in Singapore. MAS supervises all banking institutions, sets prudential requirements, conducts regular inspections, and ensures compliance with banking regulations to maintain financial stability."
    },
    {
        "question": "What are the capital adequacy requirements for banks in Singapore?",
        "ground_truth": "Singapore banks are required to maintain a minimum Common Equity Tier 1 (CET1) capital ratio of 6.5%, a Tier 1 capital ratio of 8%, and a Total Capital Ratio of 10%. These requirements are set by MAS and align with Basel III international standards."
    },
    {
        "question": "What is MAS Notice 626?",
        "ground_truth": "MAS Notice 626 relates to Prevention of Money Laundering and Countering the Financing of Terrorism (AML/CFT) requirements for banks. It sets out the regulatory requirements for customer due diligence, record keeping, suspicious transaction reporting, and compliance programs."
    },
    {
        "question": "What is STRO?",
        "ground_truth": "STRO stands for Suspicious Transaction Reporting Office. It is a unit within MAS that receives, analyzes, and disseminates suspicious transaction reports from financial institutions. STRO serves as Singapore's financial intelligence unit for combating money laundering and terrorism financing."
    },
    {
        "question": "What are the AML reporting requirements for financial institutions?",
        "ground_truth": "Financial institutions in Singapore must report suspicious transactions to STRO within 15 days of detection, regardless of the transaction amount. They must also file Cash Transaction Reports (CTRs) for cash transactions exceeding SGD 20,000 and maintain comprehensive records for at least 5 years."
    },
    {
        "question": "What is the Payment Services Act?",
        "ground_truth": "The Payment Services Act (PSA) is Singapore's comprehensive regulatory framework for payment services, enacted in 2019. It requires payment service providers to obtain licenses from MAS, sets operational and prudential requirements, and aims to promote innovation while ensuring consumer protection and financial stability."
    }
]

def generate_full_response(model, question, max_new_tokens=200):
    """Generate full response from model"""
    prompt = f"Q: {question} A:"
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=5,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            end_time = time.time()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response, end_time - start_time
    
    except Exception as e:
        return f"Error: {str(e)}", 0.0

def _tokenize(text: str):
    import re
    tokens = re.findall(r"\w+", (text or "").lower())
    # Remove common stopwords to avoid inflating scores with filler words
    stop = {
        'the','a','an','and','or','to','of','in','on','for','by','with','is','are','as','that','this','it','be','at','from','which','these','those','was','were','will','shall','should','can','may','must'
    }
    return [t for t in tokens if t not in stop]

def _lcs_length(a_tokens, b_tokens):
    # Token-level LCS for ROUGE-L recall approximation
    n, m = len(a_tokens), len(b_tokens)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            temp = dp[j]
            if a_tokens[i - 1] == b_tokens[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[m]

def _dedupe_sentences(text: str) -> str:
    sentences = [s.strip() for s in (text or "").split('. ') if s.strip()]
    seen = set()
    ordered = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return '. '.join(ordered)

def calculate_similarity_score(ground_truth, response):
    """ROUGE-L recall approximation via token-level LCS (0..1)."""
    if not ground_truth or not response:
        return 0.0
    # Light cleaning to avoid repetition inflating/deflating scores
    gt_clean = _dedupe_sentences(ground_truth)
    resp_clean = _dedupe_sentences(response)
    gt_tokens = _tokenize(gt_clean)
    resp_tokens = _tokenize(resp_clean)
    if len(gt_tokens) == 0:
        return 0.0
    lcs = _lcs_length(gt_tokens, resp_tokens)
    return lcs / len(gt_tokens)

DOMAIN_TERMS = [
    'monetary authority of singapore','monetary authority','mas','singapore','sgd','dollar','bank','banking',
    'aml','cft','stro','suspicious transaction reporting office','payment services act','psa','notice 626',
    'notice 637','capital','adequacy','regulation','regulatory','prudential'
]

def key_term_coverage(ground_truth: str, response: str) -> float:
    """Fraction of domain terms present in response, weighted by presence in ground truth when possible."""
    if not response:
        return 0.0
    resp_lower = (response or '').lower()
    gt_lower = (ground_truth or '').lower()
    terms = DOMAIN_TERMS
    if not terms:
        return 0.0
    # Weight terms that appear in ground truth higher
    total = 0.0
    score = 0.0
    for term in terms:
        w = 2.0 if term in gt_lower else 1.0
        total += w
        if term in resp_lower:
            score += w
    return score / total if total > 0 else 0.0

def semantic_similarity(ground_truth: str, response: str):
    """Cosine similarity via Sentence Transformers if available (0..1), else None."""
    if SEM_MODEL is None or not ground_truth or not response:
        return None
    try:
        emb = SEM_MODEL.encode([ground_truth, response], convert_to_tensor=True, normalize_embeddings=True)
        sim = st_util.cos_sim(emb[0], emb[1]).item()
        # Ensure within [0,1]
        return max(0.0, min(1.0, (sim + 1) / 2)) if sim < 0 else max(0.0, min(1.0, sim))
    except Exception:
        return None

def evaluate_response_quality(ground_truth, response):
    """Evaluate response quality with multiple metrics"""
    if not response or response.startswith("Error:"):
        return {
            "similarity": 0.0,
            "length_ratio": 0.0,
            "key_terms": 0,
            "quality": "error",
            "key_term_coverage": 0.0,
            "semantic": None,
            "composite": 0.0,
        }
    
    # Similarity score
    similarity = calculate_similarity_score(ground_truth, response)
    
    # Length comparison
    length_ratio = len(response) / len(ground_truth) if len(ground_truth) > 0 else 0
    
    # Key financial terms present
    financial_terms = ['mas', 'singapore', 'monetary authority', 'bank', 'financial', 'regulation', 'capital', 'requirement']
    key_terms_found = sum(1 for term in financial_terms if term in response.lower())
    coverage = key_term_coverage(ground_truth, response)
    sem = semantic_similarity(ground_truth, response)
    
    # Composite score (weights): ROUGE-L 0.5, semantic 0.4 (if available), coverage 0.1
    if sem is None:
        composite = 0.9 * similarity + 0.1 * coverage
    else:
        composite = 0.5 * similarity + 0.4 * sem + 0.1 * coverage
    composite = max(0.0, min(1.0, composite))
    
    # Overall quality assessment
    if composite > 0.5 and key_terms_found >= 3:
        quality = "excellent"
    elif composite > 0.35 and key_terms_found >= 2:
        quality = "good"
    elif composite > 0.2 or key_terms_found >= 1:
        quality = "moderate"
    else:
        quality = "poor"
    
    return {
        "similarity": similarity,
        "length_ratio": length_ratio,
        "key_terms": key_terms_found,
        "quality": quality,
        "key_term_coverage": coverage,
        "semantic": sem,
        "composite": composite,
    }

# Run comprehensive comparison
print(f"\n🚀 RUNNING COMPREHENSIVE GROUND TRUTH COMPARISON...")
print("=" * 100)

total_base_similarity = 0
total_ft_similarity = 0
base_better_count = 0
ft_better_count = 0
tie_count = 0

for i, test_case in enumerate(test_cases, 1):
    question = test_case["question"]
    ground_truth = test_case["ground_truth"]
    
    print(f"\n" + "=" * 100)
    print(f"📝 QUESTION {i}: {question}")
    print("=" * 100)
    
    # Generate responses
    base_response, base_time = generate_full_response(base_model, question)
    ft_response, ft_time = generate_full_response(finetuned_model, question)
    
    # Evaluate responses
    base_eval = evaluate_response_quality(ground_truth, base_response)
    ft_eval = evaluate_response_quality(ground_truth, ft_response)
    
    # Update totals
    total_base_similarity += base_eval["similarity"]
    total_ft_similarity += ft_eval["similarity"]
    
    # Determine winner using composite score primarily
    if ft_eval.get("composite", 0.0) > base_eval.get("composite", 0.0) + 0.03:  # 3% threshold
        ft_better_count += 1
        winner = "🟢 FINE-TUNED BETTER"
    elif base_eval.get("composite", 0.0) > ft_eval.get("composite", 0.0) + 0.03:
        base_better_count += 1
        winner = "🔵 BASE BETTER"
    else:
        tie_count += 1
        winner = "🤝 TIE"
    
    # Print detailed comparison
    print(f"\n🎯 GROUND TRUTH:")
    print(f"   {ground_truth}")
    
    print(f"\n🔵 BASE MODEL RESPONSE:")
    print(f"   {base_response}")
    print(f"   📊 ROUGE-L: {base_eval['similarity']:.3f} | Semantic: {base_eval['semantic'] if base_eval['semantic'] is not None else 'n/a'} | Coverage: {base_eval['key_term_coverage']:.2f} | Composite: {base_eval['composite']:.3f} | Quality: {base_eval['quality']} | Time: {base_time:.2f}s")
    
    print(f"\n🟢 FINE-TUNED MODEL RESPONSE:")
    print(f"   {ft_response}")
    print(f"   📊 ROUGE-L: {ft_eval['similarity']:.3f} | Semantic: {ft_eval['semantic'] if ft_eval['semantic'] is not None else 'n/a'} | Coverage: {ft_eval['key_term_coverage']:.2f} | Composite: {ft_eval['composite']:.3f} | Quality: {ft_eval['quality']} | Time: {ft_time:.2f}s")
    
    print(f"\n🏆 WINNER: {winner}")
    print(f"   Similarity Difference: {abs(ft_eval['similarity'] - base_eval['similarity']):.3f}")

# Calculate final statistics
avg_base_similarity = total_base_similarity / len(test_cases)
avg_ft_similarity = total_ft_similarity / len(test_cases)
improvement_ratio = (avg_ft_similarity / avg_base_similarity) if avg_base_similarity > 0 else 0

# Print comprehensive final results
print(f"\n" + "=" * 100)
print("🎯 COMPREHENSIVE GROUND TRUTH COMPARISON RESULTS")
print("=" * 100)

print(f"📊 AVERAGE ROUGE-L (recall) VS GROUND TRUTH:")
print(f"   🔵 Base Model: {avg_base_similarity:.3f} ({avg_base_similarity*100:.1f}%)")
print(f"   🟢 Fine-tuned: {avg_ft_similarity:.3f} ({avg_ft_similarity*100:.1f}%)")
print(f"   📈 Improvement: {improvement_ratio:.2f}x ({(improvement_ratio-1)*100:+.1f}%)")

print(f"\n🏆 HEAD-TO-HEAD RESULTS:")
print(f"   🟢 Fine-tuned Better: {ft_better_count}/{len(test_cases)} ({ft_better_count/len(test_cases)*100:.1f}%)")
print(f"   🔵 Base Better: {base_better_count}/{len(test_cases)} ({base_better_count/len(test_cases)*100:.1f}%)")
print(f"   🤝 Ties: {tie_count}/{len(test_cases)} ({tie_count/len(test_cases)*100:.1f}%)")

print(f"\n🎯 OVERALL ASSESSMENT:")
if avg_ft_similarity > avg_base_similarity * 1.5:
    assessment = "🎉 EXCELLENT: Fine-tuning significantly improved accuracy!"
    recommendation = "✅ Your fine-tuning was highly successful! Deploy with confidence."
elif avg_ft_similarity > avg_base_similarity * 1.2:
    assessment = "✅ GOOD: Fine-tuning improved accuracy meaningfully"
    recommendation = "✅ Fine-tuning is working well. Consider more training for even better results."
elif avg_ft_similarity > avg_base_similarity * 1.05:
    assessment = "⚠️ MODERATE: Fine-tuning showed slight improvement"
    recommendation = "💡 Some improvement detected. Consider more training epochs or better data."
elif avg_ft_similarity > avg_base_similarity * 0.95:
    assessment = "😐 MINIMAL: Fine-tuning showed no significant change"
    recommendation = "💡 Need to troubleshoot: learning rate, training data quality, or model architecture."
else:
    assessment = "❌ WORSE: Fine-tuning actually reduced accuracy"
    recommendation = "🚨 Fine-tuning failed. Check training data, parameters, or try different approach."

print(f"   {assessment}")
print(f"   {recommendation}")

print(f"\n💡 DETAILED INSIGHTS:")
if ft_better_count > base_better_count:
    print(f"   ✅ Fine-tuned model wins {ft_better_count}/{len(test_cases)} comparisons")
    print(f"   ✅ Shows consistent improvement in Singapore financial knowledge")
elif base_better_count > ft_better_count:
    print(f"   ❌ Base model wins {base_better_count}/{len(test_cases)} comparisons")
    print(f"   ❌ Fine-tuning may have caused knowledge degradation")
else:
    print(f"   ⚠️ Mixed results - fine-tuning inconsistent")

if avg_ft_similarity < 0.2:
    print(f"   🚨 Both models show poor similarity to ground truth (<20%)")
    print(f"   💡 Consider: Better training data, longer training, or different model architecture")
elif avg_ft_similarity < 0.4:
    print(f"   ⚠️ Moderate similarity to ground truth (20-40%)")
    print(f"   💡 Room for improvement with more training or data")
else:
    print(f"   ✅ Good similarity to ground truth (>40%)")

print(f"\n" + "=" * 100)
print("✅ COMPREHENSIVE GROUND TRUTH COMPARISON COMPLETED!")
print("=" * 100)
