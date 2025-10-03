# Data Directory

This directory contains your financial regulation documents that will be processed into Q&A pairs for fine-tuning.

## How to Use

1. **Place your documents here**: Add your financial regulation documents (PDF, TXT, DOCX, etc.) to this folder
2. **Run dataset preparation**: The `dataset_prep.py` script will automatically process all files in this directory
3. **Supported formats**: Currently supports text files, but can be extended for other formats

## Example Files

- `sample_mas_guidelines.txt` - Sample MAS guidelines document
- Add your own documents here...

## File Structure

```
data/
├── README.md
├── sample_mas_guidelines.txt
├── your_regulation_doc1.pdf
├── your_regulation_doc2.txt
└── ...
```

## Processing

The dataset preparation script will:
1. Read all files in this directory
2. Extract Q&A pairs using pattern matching
3. Generate additional questions from content
4. Save processed data to `processed_data/` directory

## Tips

- Use clear Q&A format in your documents for best results
- Include question headers like "Q:" and "A:" or "Question:" and "Answer:"
- The more structured your documents are, the better the Q&A extraction will be
