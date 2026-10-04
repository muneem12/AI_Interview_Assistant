# AI Interview Preparation Assistant

An AI-powered interview preparation assistant built with LangChain, Hugging Face, FAISS, and Qwen2.5-3B-Instruct.

## Features

- Resume-based skill analysis
- Interview guide RAG
- Skill-gap analysis
- Interview question generation
- Dynamic mock interviews
- Answer evaluation
- Interview score calculation
- Final performance report

## Technologies

- Python
- PyTorch
- LangChain
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- Qwen2.5-3B-Instruct

## Project Structure

AI_Interview_Assistant/
├── data/
│   ├── interview_guide.txt
│   └── resume.pdf
├── interview_assistant.ipynb
├── interview_report.txt
├── project_summary.txt
├── requirements.txt
└── README.md

## How It Works

1. Load the interview guide and resume.
2. Split documents into chunks.
3. Create embeddings.
4. Store embeddings in FAISS.
5. Retrieve relevant information.
6. Compare resume skills with interview topics.
7. Generate interview questions.
8. Evaluate candidate answers.
9. Calculate the interview score.
10. Generate the final report.
