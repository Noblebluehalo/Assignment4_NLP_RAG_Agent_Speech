import torch
import torch.nn as nn
from torchtext.datasets import AG_NEWS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

def load_and_preprocess():
    # 1(a): Load the 120,000 training and 7,600 test examples
    train_iter, test_iter = AG_NEWS(split=('train', 'test'))
    
    train_texts, train_labels = [], []
    for label, text in train_iter:
        train_labels.append(label)
        train_texts.append(text) 
        
    test_texts, test_labels = [], []
    for label, text in test_iter:
        test_labels.append(label)
        test_texts.append(text)
        
    return train_texts, train_labels, test_texts, test_labels

def vectorize_text(train_texts, test_texts):
    # 1(d): Create DTM and TF-IDF matrices
    count_vec = CountVectorizer()
    tfidf_vec = TfidfVectorizer()
    
    # Fit strictly on training data
    X_train_counts = count_vec.fit_transform(train_texts)
    X_test_counts = count_vec.transform(test_texts)
    
    # Calculate sparsity for 1(e)
    total_entries = X_train_counts.shape[0] * X_train_counts.shape[1]
    nonzero_entries = X_train_counts.nnz
    sparsity = (1 - (nonzero_entries / total_entries)) * 100
    
    print(f"Training DTM Shape: {X_train_counts.shape}")
    print(f"Sparsity: {sparsity:.4f}%")
    
    return X_train_counts, X_test_counts
