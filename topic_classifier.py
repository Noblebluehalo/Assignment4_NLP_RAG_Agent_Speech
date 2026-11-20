import os
import time
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

CATEGORY_NAMES = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}
ORDERED_LABELS = [1, 2, 3, 4]
ORDERED_NAMES = [CATEGORY_NAMES[i] for i in ORDERED_LABELS]


def load_ag_news():
    """
    Loads the official 120,000 train and 7,600 test AG News splits.
    Tries torchtext.datasets.AG_NEWS first, then falls back to Hugging Face datasets.
    """
    try:
        from torchtext.datasets import AG_NEWS
        train_iter, test_iter = AG_NEWS(split=("train", "test"))
        train_data = list(train_iter)
        test_data = list(test_iter)
        # torchtext format: (label in 1..4, text)
        train_labels = [int(label) for label, _ in train_data]
        train_texts = [text for _, text in train_data]
        test_labels = [int(label) for label, _ in test_data]
        test_texts = [text for _, text in test_data]
        print("Loaded AG News via torchtext.datasets.AG_NEWS")
        return train_texts, train_labels, test_texts, test_labels
    except Exception as e:
        print(f"torchtext unavailable or unsupported ({e}); falling back to Hugging Face datasets...")
        from datasets import load_dataset
        ds = load_dataset("ag_news")
        # HF ag_news has 'text' (title + description) and 'label' in 0..3
        # Check if separate 'title'/'description' columns exist just in case
        def format_split(split_ds):
            cols = split_ds.column_names
            if "title" in cols and "description" in cols:
                texts = [f"{t} {d}".strip() for t, d in zip(split_ds["title"], split_ds["description"])]
            else:
                texts = list(split_ds["text"])
            labels = [int(lbl) + 1 for lbl in split_ds["label"]]
            return texts, labels

        train_texts, train_labels = format_split(ds["train"])
        test_texts, test_labels = format_split(ds["test"])
        print("Loaded AG News via Hugging Face 'ag_news' dataset")
        return train_texts, train_labels, test_texts, test_labels


def print_dataset_summary(train_texts, train_labels, test_texts, test_labels):
    print("\n" + "=" * 70)
    print("PART 1.1: DATASET SUMMARY & EXAMPLE ARTICLES")
    print("=" * 70)
    print(f"Total training examples: {len(train_texts)}")
    print(f"Total test examples:     {len(test_texts)}\n")

    print(f"{'Category':<15} | {'Train Count':<12} | {'Test Count':<12}")
    print("-" * 45)
    for cat_id in ORDERED_LABELS:
        tr_cnt = sum(1 for y in train_labels if y == cat_id)
        te_cnt = sum(1 for y in test_labels if y == cat_id)
        print(f"{cat_id}: {CATEGORY_NAMES[cat_id]:<12} | {tr_cnt:<12} | {te_cnt:<12}")

    print("\n--- One Example Article per Category (Training Split) ---")
    for cat_id in ORDERED_LABELS:
        idx = train_labels.index(cat_id)
        print(f"[{cat_id}: {CATEGORY_NAMES[cat_id]}] {train_texts[idx]}\n")


def inspect_vectorizers(train_texts, test_texts):
    print("=" * 70)
    print("PART 1.1: VECTORIZATION & TRAINING DTM SPARSITY")
    print("=" * 70)
    count_vec = CountVectorizer()
    X_train_dtm = count_vec.fit_transform(train_texts)
    X_test_dtm = count_vec.transform(test_texts)

    tfidf_vec = TfidfVectorizer()
    X_train_tfidf = tfidf_vec.fit_transform(train_texts)
    X_test_tfidf = tfidf_vec.transform(test_texts)

    num_docs, num_features = X_train_dtm.shape
    nonzero_entries = X_train_dtm.nnz
    total_entries = num_docs * num_features
    sparsity = (1.0 - (nonzero_entries / total_entries)) * 100.0

    print(f"Training DTM Shape:   {num_docs} documents x {num_features} features")
    print(f"Test DTM Shape:       {X_test_dtm.shape[0]} documents x {X_test_dtm.shape[1]} features")
    print(f"Training TF-IDF Shape:{X_train_tfidf.shape[0]} documents x {X_train_tfidf.shape[1]} features")
    print(f"Nonzero Entries:      {nonzero_entries:,}")
    print(f"Total Entries:        {total_entries:,}")
    print(f"Training DTM Sparsity:{sparsity:.6f}%\n")


def build_mlp():
    """
    Consistent MLP architecture and training hyperparameters across all 4 runs.
    """
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=15,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=3,
        random_state=42,
        verbose=True,
    )


def format_confusion_matrix(cm):
    header = f"{'True \\ Pred':<12} | " + " | ".join(f"{name:<10}" for name in ORDERED_NAMES)
    lines = [header, "-" * len(header)]
    for i, row_name in enumerate(ORDERED_NAMES):
        row_vals = " | ".join(f"{cm[i, j]:<10}" for j in range(len(ORDERED_NAMES)))
        lines.append(f"{row_name:<12} | {row_vals}")
    return "\n".join(lines)


def run_experiments():
    train_texts, train_labels, test_texts, test_labels = load_ag_news()
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    print_dataset_summary(train_texts, train_labels, test_texts, test_labels)
    inspect_vectorizers(train_texts, test_texts)

    configs = [
        ("CountVectorizer", "No", CountVectorizer(stop_words=None)),
        ("TfidfVectorizer", "No", TfidfVectorizer(stop_words=None)),
        ("CountVectorizer", "Yes", CountVectorizer(stop_words="english")),
        ("TfidfVectorizer", "Yes", TfidfVectorizer(stop_words="english")),
    ]

    results = []
    best_acc = -1.0
    best_bundle = None

    for vec_name, stop_flag, vectorizer in configs:
        print("=" * 70)
        print(f"TRAINING MLP: Vectorizer={vec_name} | Stop-word removal={stop_flag}")
        print("=" * 70)

        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)

        mlp = build_mlp()
        t0 = time.perf_counter()
        mlp.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        y_pred = mlp.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_prec = precision_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred, labels=ORDERED_LABELS)

        print(f"\n[{vec_name} | Stop-words={stop_flag}]")
        print(f"Training Time:         {train_time:.2f}s (epochs run: {mlp.n_iter_})")
        print(f"Test Accuracy:         {acc * 100:.2f}%")
        print(f"Macro-Avg Precision:   {macro_prec * 100:.2f}%")
        print("Confusion Matrix:")
        print(format_confusion_matrix(cm))
        print()

        # Part 1.2 requires 3 misclassified examples specifically for CountVectorizer (No stop-words)
        if vec_name == "CountVectorizer" and stop_flag == "No":
            print("--- Three Misclassified Test Examples (CountVectorizer, No Stop-words) ---")
            mis_indices = np.where(y_pred != y_test)[0]
            for count, idx in enumerate(mis_indices[:3], start=1):
                true_name = CATEGORY_NAMES[int(y_test[idx])]
                pred_name = CATEGORY_NAMES[int(y_pred[idx])]
                print(f"Example {count} (Test Index {idx}):")
                print(f"  True Category:      {true_name}")
                print(f"  Predicted Category: {pred_name}")
                print(f"  Text:               {test_texts[idx]}\n")

        results.append({
            "vectorizer": vec_name,
            "stop_words": stop_flag,
            "accuracy": acc,
            "macro_precision": macro_prec,
            "confusion_matrix": cm,
        })

        if acc > best_acc:
            best_acc = acc
            best_bundle = {
                "config_name": f"{vec_name} (stop_words={stop_flag})",
                "vectorizer": vectorizer,
                "model": mlp,
                "test_accuracy": acc,
                "macro_precision": macro_prec,
            }

    print("=" * 70)
    print("PART 1.3: SUMMARY COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Vectorizer':<18} | {'Stop-Word Removal':<18} | {'Test Accuracy':<14} | {'Macro Precision':<15}")
    print("-" * 74)
    for r in results:
        print(
            f"{r['vectorizer']:<18} | {r['stop_words']:<18} | "
            f"{r['accuracy'] * 100:>12.2f}% | {r['macro_precision'] * 100:>13.2f}%"
        )

    best_res = max(results, key=lambda x: x["accuracy"])
    print(f"\nBest Configuration: {best_res['vectorizer']} (Stop-word removal: {best_res['stop_words']})")
    print("Best Configuration Confusion Matrix:")
    print(format_confusion_matrix(best_res["confusion_matrix"]))

    # Save best model + vectorizer so Part 2 can evaluate it on the exact same 200-sample subset
    joblib.dump(best_bundle, "part1_best_model.joblib")
    print("\nSaved best Part 1 pipeline to 'part1_best_model.joblib' for Part 2 evaluation.")


if __name__ == "__main__":
    run_experiments()
