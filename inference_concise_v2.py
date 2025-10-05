import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Concise, deterministic inference using LoRA adapters

def load_finetuned(base_id: str, lora_dir: str):
	tok = AutoTokenizer.from_pretrained(base_id)
	if tok.pad_token is None:
		tok.pad_token = tok.eos_token
	base = AutoModelForCausalLM.from_pretrained(base_id)
	model = PeftModel.from_pretrained(base, lora_dir)
	return tok, model


def answer(tok, model, question: str, fewshot: str = None, max_new_tokens: int = 120):
	instr = "Answer concisely and factually in 1–3 sentences.\n"
	few = (fewshot + "\n\n") if fewshot else ""
	prompt = instr + few + f"Q: {question}\nA:"
	inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=512)
	with torch.no_grad():
		out = model.generate(
			**inputs,
			num_beams=5,
			do_sample=False,
			max_new_tokens=max_new_tokens,
			no_repeat_ngram_size=3,
			repetition_penalty=1.2,
			pad_token_id=tok.eos_token_id,
			eos_token_id=tok.eos_token_id,
		)
	text = tok.decode(out[0], skip_special_tokens=True)
	return text.replace(prompt, '').strip()


if __name__ == '__main__':
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument('--base', default='gpt2')
	p.add_argument('--lora', required=True)
	p.add_argument('--q', required=True)
	args = p.parse_args()
	tok, mdl = load_finetuned(args.base, args.lora)
	print(answer(tok, mdl, args.q))
