import json
from typing import List, Dict
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model

# New script: trains with answer-only loss (question labels masked to -100)


def build_features(q: str, a: str, tokenizer, max_len: int = 256) -> Dict[str, List[int]]:
	prompt = f"Q: {q}\nA:"
	prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
	answer_ids = tokenizer(" " + a + tokenizer.eos_token, add_special_tokens=False).input_ids
	input_ids = (prompt_ids + answer_ids)[:max_len]
	labels = ([-100] * len(prompt_ids)) + answer_ids
	labels = labels[:max_len]
	if len(labels) < len(input_ids):
		labels += [-100] * (len(input_ids) - len(labels))
	attn = [1] * len(input_ids)
	return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def load_qa_list(json_path: str) -> List[Dict[str, str]]:
	with open(json_path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	items = []
	for s in data:
		if isinstance(s, str) and s.startswith('Q: ') and ' A: ' in s:
			q, a = s.split(' A: ', 1)
			q = q.replace('Q: ', '').strip()
			a = a.strip()
			items.append({'q': q, 'a': a})
	return items


def main():
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument('--data', required=True, help='Path to cleaned Q/A list JSON')
	p.add_argument('--out', required=True, help='Output dir for LoRA adapters')
	p.add_argument('--base', default='gpt2', help='Base model id')
	args = p.parse_args()

	tokenizer = AutoTokenizer.from_pretrained(args.base)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	pairs = load_qa_list(args.data)
	features = [build_features(it['q'], it['a'], tokenizer) for it in pairs]
	ds = Dataset.from_list(features)

	base_model = AutoModelForCausalLM.from_pretrained(args.base)
	lora_cfg = LoraConfig(
		task_type=TaskType.CAUSAL_LM,
		r=16,
		lora_alpha=32,
		lora_dropout=0.1,
		target_modules=["c_attn","c_fc","c_proj"],
	)
	model = get_peft_model(base_model, lora_cfg)

	args_tr = TrainingArguments(
		output_dir=args.out,
		learning_rate=3e-5,
		num_train_epochs=4,
		per_device_train_batch_size=2,
		gradient_accumulation_steps=4,
		warmup_ratio=0.1,
		weight_decay=0.1,
		max_grad_norm=1.0,
		evaluation_strategy="no",
		report_to="none",
		dataloader_num_workers=0,
		logging_steps=50,
		save_strategy="epoch",
		load_best_model_at_end=False,
	)

	trainer = Trainer(
		model=model,
		args=args_tr,
		train_dataset=ds,
		tokenizer=tokenizer,
	)

	trainer.train()
	model.save_pretrained(args.out)
	tokenizer.save_pretrained(args.out)
	print('✅ Saved LoRA adapters to', args.out)


if __name__ == '__main__':
	main()
