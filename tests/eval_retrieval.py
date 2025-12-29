#!/usr/bin/env python3
"""
Evaluate retrieval quality using DeepEval.
Generates synthetic test questions from chunks and evaluates retrieval.
"""

import json
import os
import pickle
import random
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models import DeepEvalBaseLLM

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is required")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-4o-mini"

INDEX_DIR = "vector_store"
CHUNKS_FILE = "sec_corpus/META/chunked/all_chunks.json"

# Eval settings
NUM_QUESTIONS = 25
EVOLUTION_MIX = {"simple": 0.7, "reasoning": 0.3}
TOP_K = 5
THRESHOLD = 0.7

# Filter settings - set to None to run all, or "simple"/"reasoning" to filter
QUESTION_TYPE_FILTER = "simple"  # Only run simple queries for retrieval testing


# ─────────────────────────────────────────────────────────────────────────────
# OPENROUTER LLM WRAPPER FOR DEEPEVAL
# ─────────────────────────────────────────────────────────────────────────────
class OpenRouterLLM(DeepEvalBaseLLM):
    """Custom LLM wrapper for OpenRouter API."""

    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
def load_resources():
    """Load model, index, and metadata."""
    with open(f"{INDEX_DIR}/config.json") as f:
        config = json.load(f)

    model = SentenceTransformer(config["model_name"])
    index = faiss.read_index(f"{INDEX_DIR}/faiss_index.bin")

    with open(f"{INDEX_DIR}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)

    return model, index, metadata, chunks


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def retrieve(query: str, model, index, metadata, top_k: int = TOP_K) -> list[str]:
    """Retrieve top-k chunk contents for a query."""
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding.astype(np.float32), k=top_k)

    results = []
    for idx in indices[0]:
        if idx >= 0:
            results.append(metadata[idx]["content"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC TEST GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_test_cases(chunks: list[dict], llm: OpenRouterLLM) -> list[dict]:
    """Generate synthetic test questions from chunks."""

    # Filter to substantive chunks (longer, not just bullets)
    substantive = [
        c for c in chunks
        if c["token_count"] > 50 and c["chunk_type"] in ["paragraph", "lead_sentence"]
    ]

    # Sample diverse chunks across years
    random.seed(42)
    sampled = random.sample(substantive, min(NUM_QUESTIONS, len(substantive)))

    # Determine question types based on evolution mix
    num_simple = int(NUM_QUESTIONS * EVOLUTION_MIX["simple"])
    num_reasoning = NUM_QUESTIONS - num_simple

    test_cases = []

    print(f"\nGenerating {NUM_QUESTIONS} test questions...")
    print(f"  - Simple: {num_simple}")
    print(f"  - Reasoning: {num_reasoning}")
    print("=" * 60)

    for i, chunk in enumerate(sampled):
        q_type = "simple" if i < num_simple else "reasoning"

        if q_type == "simple":
            prompt = f"""Based on this text from Meta's 10-K Risk Factors (FY{chunk['fiscal_year']}):

"{chunk['content'][:1200]}"

Generate a simple search query (5-10 words) that someone would type to find this information.
Focus on key entities, topics, or facts mentioned.

Respond with ONLY the search query, nothing else."""

        else:  # reasoning
            prompt = f"""Based on this text from Meta's 10-K Risk Factors (FY{chunk['fiscal_year']}):

"{chunk['content'][:1200]}"

Generate a reasoning-based question that requires understanding this text to answer.
The question should be natural, like what an analyst would ask.
Keep it under 15 words.

Respond with ONLY the question, nothing else."""

        print(f"[{i+1}/{NUM_QUESTIONS}] Generating {q_type} question for {chunk['chunk_id']}...")

        try:
            query = llm.generate(prompt).strip().strip('"')

            test_cases.append({
                "query": query,
                "question_type": q_type,
                "ground_truth_context": chunk["content"],
                "chunk_id": chunk["chunk_id"],
                "fiscal_year": chunk["fiscal_year"],
                "section": chunk["section"],
            })
            print(f"    → \"{query[:60]}...\"")

        except Exception as e:
            print(f"    Error: {e}")
            continue

    return test_cases


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(test_cases: list[dict], model, index, metadata, llm: OpenRouterLLM):
    """Run DeepEval evaluation on test cases."""

    print(f"\n" + "=" * 60)
    print("RUNNING DEEPEVAL EVALUATION")
    print("=" * 60)

    # Create DeepEval test cases
    deepeval_test_cases = []

    for tc in test_cases:
        # Retrieve context for this query
        retrieved_context = retrieve(tc["query"], model, index, metadata, top_k=TOP_K)

        # Create LLMTestCase
        # - input: the query
        # - actual_output: placeholder (we're only testing retrieval)
        # - retrieval_context: what our retriever returned
        # - expected_output: the ground truth chunk content
        deepeval_test_cases.append(
            LLMTestCase(
                input=tc["query"],
                actual_output="[Retrieval only - no generation]",
                retrieval_context=retrieved_context,
                expected_output=tc["ground_truth_context"],
            )
        )

    # Define metrics
    recall_metric = ContextualRecallMetric(
        threshold=THRESHOLD,
        model=llm,
        include_reason=True,
    )

    precision_metric = ContextualPrecisionMetric(
        threshold=THRESHOLD,
        model=llm,
        include_reason=True,
    )

    # Run evaluation
    results = evaluate(
        test_cases=deepeval_test_cases,
        metrics=[recall_metric, precision_metric],
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("10-K RISK ANALYSIS - RETRIEVAL EVALUATION")
    print("=" * 60)
    print(f"Config:")
    print(f"  - Questions: {NUM_QUESTIONS}")
    print(f"  - Evolution: {EVOLUTION_MIX}")
    print(f"  - Filter: {QUESTION_TYPE_FILTER or 'None (all)'}")
    print(f"  - top_k: {TOP_K}")
    print(f"  - Threshold: {THRESHOLD}")
    print(f"  - LLM: {LLM_MODEL}")

    # Load resources
    print("\nLoading resources...")
    model, index, metadata, chunks = load_resources()
    print(f"  - Chunks: {len(chunks)}")
    print(f"  - Indexed: {index.ntotal}")

    # Initialize LLM
    llm = OpenRouterLLM()

    # Load existing test cases if available, otherwise generate
    test_cases_file = Path("eval_test_cases.json")
    if test_cases_file.exists():
        print(f"\nLoading existing test cases from {test_cases_file}...")
        with open(test_cases_file) as f:
            test_cases = json.load(f)
        print(f"  - Loaded {len(test_cases)} test cases")
    else:
        # Generate test cases
        test_cases = generate_test_cases(chunks, llm)
        # Save test cases
        with open("eval_test_cases.json", "w") as f:
            json.dump(test_cases, f, indent=2)
        print(f"\nSaved {len(test_cases)} test cases to eval_test_cases.json")

    # Filter by question type if specified
    if QUESTION_TYPE_FILTER:
        test_cases = [tc for tc in test_cases if tc["question_type"] == QUESTION_TYPE_FILTER]
        print(f"  - Filtered to {len(test_cases)} {QUESTION_TYPE_FILTER} test cases")

    # Run evaluation
    results = run_evaluation(test_cases, model, index, metadata, llm)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
