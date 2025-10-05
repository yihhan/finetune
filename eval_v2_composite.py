import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List

try:
	from sentence_transformers import SentenceTransformer, util as st_util
	SEM_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
except Exception:
	SEM_MODEL = None

DOMAIN_TERMS = [
	'monetary authority of singapore','monetary authority','mas','singapore','sgd','dollar','bank','banking',
	'aml','cft','stro','suspicious transaction reporting office','payment services act','psa','notice 626',
	'notice 637','capital','adequacy','regulation','regulatory','prudential'
]


def tokenize(text: str):
	import re
	tokens = re.findall(r"\w+", (text or '').lower())
	stop = {'the','a','an','and','or','to','of','in','on','for','by','with','is','are','as','that','this','it','be','at','from','which','these','those','was','were','will','shall','should','can','may','must'}
	return [t for t in tokens if t not in stop]


def lcs_len(a: List[str], b: List[str]) -> int:
	n, m = len(a), len(b)
	if n == 0 or m == 0:
		return 0
	dp = [0]*(m+1)
	for i in range(1, n+1):
		prev = 0
		for j in range(1, m+1):
			tmp = dp[j]
			if a[i-1] == b[j-1]:
				dp[j] = prev + 1
			else:
				dp[j] = max(dp[j], dp[j-1])
			prev = tmp
	return dp[m]


def rouge_l_recall(gt: str, resp: str) -> float:
	gt_t, r_t = tokenize(gt), tokenize(resp)
	return (lcs_len(gt_t, r_t)/len(gt_t)) if len(gt_t) else 0.0


def key_term_coverage(gt: str, resp: str) -> float:
	resp_l, gt_l = (resp or '').lower(), (gt or '').lower()
	total = 0.0; score = 0.0
	for term in DOMAIN_TERMS:
		w = 2.0 if term in gt_l else 1.0
		total += w
		if term in resp_l:
			score += w
	return score/total if total else 0.0


def semantic(gt: str, resp: str):
	if SEM_MODEL is None or not gt or not resp:
		return None
	try:
		emb = SEM_MODEL.encode([gt, resp], convert_to_tensor=True, normalize_embeddings=True)
		sim = st_util.cos_sim(emb[0], emb[1]).item()
		return max(0.0, min(1.0, sim))
	except Exception:
		return None


def load_models(base_id: str, lora_dir: str):
	tok = AutoTokenizer.from_pretrained(base_id)
	if tok.pad_token is None:
		tok.pad_token = tok.eos_token
	base = AutoModelForCausalLM.from_pretrained(base_id)
	model = PeftModel.from_pretrained(base, lora_dir)
	return tok, model


def generate(tok, model, prompt: str, max_new=160):
	inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=512)
	start = time.time()
	with torch.no_grad():
		out = model.generate(
			**inputs, num_beams=5, do_sample=False, max_new_tokens=max_new,
			no_repeat_ngram_size=3, repetition_penalty=1.2,
			pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
		)
	resp = tok.decode(out[0], skip_special_tokens=True)
	elapsed = time.time() - start
	return resp, elapsed


def main():
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument('--data', required=True, help='JSON with {question, ground_truth} list')
	p.add_argument('--base', default='gpt2')
	p.add_argument('--lora', required=True)
	args = p.parse_args()

	tok, mdl = load_models(args.base, args.lora)
	with open(args.data, 'r', encoding='utf-8') as f:
		items = json.load(f)

	agg = {'rouge': [], 'semantic': [], 'coverage': [], 'composite': [], 'time': []}
	for i, ex in enumerate(items, 1):
		q, gt = ex['question'], ex['ground_truth']
		prompt = f"Answer concisely and factually in 1–3 sentences.\nQ: {q}\nA:"
		resp, dt = generate(tok, mdl, prompt)
		resp_only = resp.replace(prompt, '').strip()
		R = rouge_l_recall(gt, resp_only)
		K = key_term_coverage(gt, resp_only)
		S = semantic(gt, resp_only)
		comp = (0.5*R + 0.4*(S if S is not None else 0.0) + 0.1*K)
		agg['rouge'].append(R); agg['semantic'].append(S if S is not None else 0.0)
		agg['coverage'].append(K); agg['composite'].append(comp); agg['time'].append(dt)
		print(f"\n{i}. Q: {q}\nGT: {gt}\nANS: {resp_only}\nScores -> ROUGE-L: {R:.3f}, Sem: {S if S is not None else 'n/a'}, Cov: {K:.2f}, Composite: {comp:.3f}, Time: {dt:.2f}s")

	import statistics as st
	print("\n=== AGGREGATE ===")
	print(f"ROUGE-L: {st.mean(agg['rouge']):.3f}")
	print(f"Semantic: {st.mean(agg['semantic']):.3f}")
	print(f"Coverage: {st.mean(agg['coverage']):.3f}")
	print(f"Composite: {st.mean(agg['composite']):.3f}")
	print(f"Avg Time: {st.mean(agg['time']):.2f}s")

if __name__ == '__main__':
	main()
