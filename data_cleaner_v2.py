import json
import re
from typing import List, Dict

# Simple, conservative cleaner; does not overwrite existing datasets

def clean_answer(answer: str, max_chars: int = 800) -> str:
	# remove URLs
	answer = re.sub(r'https?://\S+|www\.\S+', '', answer)
	# sentence dedupe
	sentences = [s.strip() for s in answer.split('. ') if s.strip()]
	seen = set()
	unique = []
	for s in sentences:
		low = s.lower()
		if low.startswith('implementation should be proportionate'):
			continue
		if s not in seen:
			seen.add(s)
			unique.append(s)
	answer = '. '.join(unique)
	# collapse whitespace and cap length
	answer = ' '.join(answer.split())
	return answer[:max_chars]


def clean_dataset(input_json_path: str, output_json_path: str) -> Dict[str, int]:
	with open(input_json_path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	cleaned: List[str] = []

	stats = {
		'count': 0,
		'urls_removed': 0,
		'repetitive_removed': 0,
		'capped': 0,
	}

	for item in data:
		if not isinstance(item, str) or not item.startswith('Q: ') or ' A: ' not in item:
			continue
		q, a = item.split(' A: ', 1)
		before = a
		a_clean = clean_answer(a)
		if 'http' in before or 'www.' in before:
			stats['urls_removed'] += 1
		if a_clean.lower() != a.lower():
			stats['repetitive_removed'] += 1
		if len(a_clean) < len(a):
			stats['capped'] += int(len(a_clean) < len(a))
		cleaned.append(f"{q} A: {a_clean}")
		stats['count'] += 1

	with open(output_json_path, 'w', encoding='utf-8') as f:
		json.dump(cleaned, f, ensure_ascii=False, indent=2)

	return stats


if __name__ == '__main__':
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument('--in', dest='input_path', required=True, help='Path to original JSON (Q/A list)')
	p.add_argument('--out', dest='output_path', required=True, help='Path to write cleaned JSON')
	args = p.parse_args()
	res = clean_dataset(args.input_path, args.output_path)
	print('✅ Cleaned dataset written:', args.output_path)
	print('📊 Stats:', res)
