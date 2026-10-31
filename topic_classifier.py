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
