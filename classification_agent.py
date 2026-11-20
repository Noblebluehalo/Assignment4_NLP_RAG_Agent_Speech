import os
import re
import json
import time
import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

ALLOWED_CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]
CATEGORY_NAMES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

DEFAULT_MODEL = os.environ.get("NAUTILUS_MODEL", "gpt-oss")
NAUTILUS_BASE_URL = os.environ.get("NAUTILUS_BASE_URL", "https://ellm.nrp-nautilus.io/v1")

SYSTEM_PROMPT = """You are a news topic classification agent.
Classify the provided AG News article into EXACTLY ONE of the following four categories:
- World
- Sports
- Business
- Sci/Tech

Rules:
1. Return ONLY a valid JSON object with no markdown formatting, no extra text, and no chain-of-thought.
2. The JSON object must contain ONLY these two keys:
   - "category": one of "World", "Sports", "Business", or "Sci/Tech"
   - "justification": a single concise sentence explaining why the article belongs to that category."""

USER_PROMPT_TEMPLATE = """Classify the following news article:

Article:
{article_text}

Respond with valid JSON containing only "category" and "justification"."""


def load_ag_news_test():
    try:
        from torchtext.datasets import AG_NEWS
        test_iter = AG_NEWS(split="test")
        test_data = list(test_iter)
        test_labels = [int(label) for label, _ in test_data]
        test_texts = [text for _, text in test_data]
        return test_texts, test_labels
    except Exception:
        from datasets import load_dataset
        ds = load_dataset("ag_news", split="test")
        cols = ds.column_names
        if "title" in cols and "description" in cols:
            texts = [f"{t} {d}".strip() for t, d in zip(ds["title"], ds["description"])]
        else:
            texts = list(ds["text"])
        labels = [int(lbl) + 1 for lbl in ds["label"]]
        return texts, labels


def build_classification_agent():
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=0,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=NAUTILUS_BASE_URL,
    )
    # Create LangChain agent with zero-shot system prompt (no external tools needed)
    agent = create_agent(llm, tools=[], system_prompt=SYSTEM_PROMPT)
    return agent


def parse_and_validate_json(raw_text: str) -> dict:
    """
    Validates that output is valid JSON with exact keys {'category', 'justification'}
    and that 'category' is one of the 4 permitted categories.
    """
    cleaned = raw_text.strip()
    # Strip any hidden reasoning tags if a reasoning model emits them
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # Extract JSON object substring if surrounded by stray text
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Output is not a JSON object.")

    expected_keys = {"category", "justification"}
    if set(data.keys()) != expected_keys:
        raise ValueError(f"Invalid JSON keys {set(data.keys())}; expected {expected_keys}.")

    category = data["category"]
    justification = data["justification"]

    if category not in ALLOWED_CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Must be one of {ALLOWED_CATEGORIES}.")
    if not isinstance(justification, str) or not justification.strip():
        raise ValueError("Justification must be a non-empty string.")

    return {
        "category": category,
        "justification": justification.strip(),
    }


def classify_article(agent, article_text: str, max_retries: int = 1) -> dict:
    """
    Classifies a single AG News article and validates JSON output.
    Retries once with explicit error feedback if output validation fails.
    """
    messages = [("user", USER_PROMPT_TEMPLATE.format(article_text=article_text))]

    for attempt in range(max_retries + 1):
        response = agent.invoke({"messages": messages})
        raw_output = response["messages"][-1].content
        try:
            return parse_and_validate_json(raw_output)
        except Exception as err:
            if attempt < max_retries:
                messages.append(("assistant", raw_output))
                messages.append((
                    "user",
                    f"Validation error: {err}. Return ONLY valid JSON with exact keys "
                    f'"category" (one of {ALLOWED_CATEGORIES}) and "justification".'
                ))
            else:
                return {
                    "category": "INVALID",
                    "justification": f"Validation failed after retry: {err}",
                }


def evaluate_comparison(sample_size: int = 200, random_seed: int = 42):
    print("=" * 70)
    print("PART 2: ZERO-SHOT LANGCHAIN AGENT VS. PART 1 BEST MLP")
    print("=" * 70)

    print("Zero-Shot System Prompt:\n" + "-" * 40)
    print(SYSTEM_PROMPT)
    print("-" * 40 + "\n")

    test_texts, test_labels = load_ag_news_test()
    y_test = np.array(test_labels)

    # Stratified sample of 200 examples with seed 42
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=random_seed)
    _, sample_idx = next(splitter.split(test_texts, y_test))

    sample_texts = [test_texts[i] for i in sample_idx]
    sample_true_names = [CATEGORY_NAMES[int(y_test[i])] for i in sample_idx]
    sample_true_ids = y_test[sample_idx]

    print(f"Stratified sample size: {len(sample_texts)} (seed={random_seed})")

    # 1. Evaluate Best Part 1 MLP on the exact same 200 examples
    if not os.path.exists("part1_best_model.joblib"):
        raise FileNotFoundError(
            "part1_best_model.joblib not found. Please run topic_classifier.py first."
        )
    bundle = joblib.load("part1_best_model.joblib")
    vectorizer = bundle["vectorizer"]
    mlp = bundle["model"]

    t0_mlp = time.perf_counter()
    X_sample = vectorizer.transform(sample_texts)
    mlp_preds_ids = mlp.predict(X_sample)
    mlp_total_time = time.perf_counter() - t0_mlp
    mlp_acc = accuracy_score(sample_true_ids, mlp_preds_ids)
    mlp_latency = mlp_total_time / len(sample_texts)

    # 2. Evaluate LangChain LLM Agent on the exact same 200 examples
    agent = build_classification_agent()
    agent_preds = []
    validation_failures = 0

    print("\nRunning LangChain Agent evaluation...")
    t0_agent = time.perf_counter()
    for idx, (text, true_cat) in enumerate(zip(sample_texts, sample_true_names), start=1):
        result = classify_article(agent, text, max_retries=1)
        pred_cat = result["category"]
        agent_preds.append(pred_cat)
        if pred_cat == "INVALID":
            validation_failures += 1

        # Print first 5 examples for inspection and report evidence
        if idx <= 5:
            print(f"\n[Sample {idx}] True: {true_cat} | Predicted: {pred_cat}")
            print(f"  JSON Output: {json.dumps(result)}")
        elif idx % 25 == 0:
            print(f"  Processed {idx}/{len(sample_texts)} examples...")

    agent_total_time = time.perf_counter() - t0_agent
    agent_latency = agent_total_time / len(sample_texts)
    agent_acc = sum(1 for p, t in zip(agent_preds, sample_true_names) if p == t) / len(sample_texts)

    print("\n" + "=" * 70)
    print("PART 2.2: COMPARISON SUMMARY TABLE")
    print("=" * 70)
    print(f"Best Part 1 Config:      {bundle['config_name']}")
    print(f"Part 1 Sample Accuracy:  {mlp_acc * 100:.2f}%")
    print(f"Part 1 Total Time:       {mlp_total_time:.4f} s (Avg latency: {mlp_latency * 1000:.3f} ms/example)")
    print("-" * 70)
    print(f"Agent Model:             {DEFAULT_MODEL} @ {NAUTILUS_BASE_URL}")
    print(f"Agent Sample Accuracy:   {agent_acc * 100:.2f}% (Validation failures: {validation_failures})")
    print(f"Agent Total Time:        {agent_total_time:.2f} s (Avg latency: {agent_latency:.3f} s/example)")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_comparison(sample_size=200, random_seed=42)
