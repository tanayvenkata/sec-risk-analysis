#!/usr/bin/env python3
"""Quick eval run with summary output."""

import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models import DeepEvalBaseLLM

# Config
OPENROUTER_API_KEY = 'REDACTED_API_KEY'
INDEX_DIR = 'vector_store'
TOP_K = 5
THRESHOLD = 0.7

class OpenRouterLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.client = OpenAI(api_key=OPENROUTER_API_KEY, base_url='https://openrouter.ai/api/v1')
    def load_model(self): return self.client
    def generate(self, prompt):
        return self.client.chat.completions.create(
            model='openai/gpt-4o-mini',
            messages=[{'role':'user','content':prompt}],
            max_tokens=1000,
            temperature=0
        ).choices[0].message.content
    async def a_generate(self, prompt): return self.generate(prompt)
    def get_model_name(self): return 'openai/gpt-4o-mini'

# Load resources
print("Loading resources...")
with open(f'{INDEX_DIR}/config.json') as f:
    config = json.load(f)
model = SentenceTransformer(config['model_name'])
index = faiss.read_index(f'{INDEX_DIR}/faiss_index.bin')
with open(f'{INDEX_DIR}/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
with open('eval_test_cases.json') as f:
    test_cases = json.load(f)

def retrieve(query):
    emb = model.encode([query], normalize_embeddings=True)
    _, indices = index.search(emb.astype(np.float32), k=TOP_K)
    return [metadata[i]['content'] for i in indices[0] if i >= 0]

# Build test cases
print(f"Building {len(test_cases)} test cases...")
llm = OpenRouterLLM()
deepeval_cases = []
for tc in test_cases:
    retrieved = retrieve(tc['query'])
    deepeval_cases.append(LLMTestCase(
        input=tc['query'],
        actual_output='[Retrieval only]',
        retrieval_context=retrieved,
        expected_output=tc['ground_truth_context']
    ))

# Evaluate
print("Running evaluation...")
recall = ContextualRecallMetric(threshold=THRESHOLD, model=llm, include_reason=False)
precision = ContextualPrecisionMetric(threshold=THRESHOLD, model=llm, include_reason=False)

results = evaluate(test_cases=deepeval_cases, metrics=[recall, precision])

# Summary
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
