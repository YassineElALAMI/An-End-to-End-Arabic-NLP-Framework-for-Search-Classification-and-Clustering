# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import sys
import io

# Force UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load serialized objects
with open('resources/models/tfidf/vocabulary.ser', 'rb') as f:
    vocabulary = pickle.load(f)

with open('resources/models/tfidf/doc_index.ser', 'rb') as f:
    doc_index = pickle.load(f)

with open('resources/models/tfidf/tfidf_matrix.ser', 'rb') as f:
    tfidf_matrix = pickle.load(f)

# Create DataFrame with UTF-8 encoding
df = pd.DataFrame(tfidf_matrix, columns=list(vocabulary.keys()), index=doc_index)

# Save to CSV with UTF-8 encoding (utf-8-sig includes BOM for Excel compatibility)
df.to_csv('resources/visualization/tfidf_matrix.csv', encoding='utf-8-sig')

print("✅ TF-IDF matrix converted to CSV (with Arabic support)")
print(f"   Shape: {df.shape[0]} documents x {df.shape[1]} terms")
print("   Saved to: resources/visualization/tfidf_matrix.csv")
print("\n📝 Sample - First 5 documents x first 5 terms:")
print(df.iloc[:5, :5])
