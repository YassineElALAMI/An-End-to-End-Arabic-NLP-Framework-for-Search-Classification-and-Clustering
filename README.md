# Arabic NLP System

A comprehensive Natural Language Processing system for Arabic text classification and document search. This project demonstrates the complete NLP pipeline: from text preprocessing and TF-IDF vectorization, through document search, to supervised text classification using Naive Bayes.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Architecture](#project-architecture)
3. [Implementation Details](#implementation-details)
4. [Building & Running](#building--running)

## Project Overview

The Arabic NLP System processes Arabic text documents from multiple news sources (Agadir24, Hespres, Medi1TV, Voice of Morocco) and organizes them into 5 categories:
- **مجتمع** (Society)
- **سياسة** (Politics)
- **اقتصاد** (Economics)
- **رياضة** (Sports)
- **فنون** (Arts)

The system implements a complete NLP pipeline with search and classification capabilities.

## Project Architecture

### Directory Structure

```
ArabicNLPSystem/
├── lib/
│   └── arabic-stemmer.jar              # Arabic stemming library (external)
│
├── resources/
│   ├── corpus/                         # Original text corpora by source & category
│   │   ├── agadir24/
│   │   ├── Hespres/
│   │   ├── Medi1TV/
│   │   └── Voice of morocco/
│   │
│   ├── data/
│   │   ├── flattened_docs/             # Preprocessed documents (all in one directory)
│   │   └── classified_clusters/        # Documents organized by predicted category
│   │
│   ├── models/
│   │   └── tfidf/                      # TF-IDF matrix and vocabulary (serialized)
│   │
│   └── stopwords/
│       └── arabic_stopwords.txt        # List of Arabic stopwords
│
├── src/main/java/ma/yassine/arabicnlp/
│   │
│   ├── nlp/                            # Text preprocessing pipeline
│   │   ├── Tokenizer.java              # Splits text into tokens
│   │   ├── StopWordsFilter.java        # Removes common Arabic words
│   │   ├── StemmerWrapper.java         # Wraps the external Arabic stemmer
│   │   └── TextPreprocessor.java       # Orchestrates the 3-step pipeline
│   │
│   ├── vectorization/                  # TF-IDF vectorization
│   │   ├── TFIDFBuilder.java           # Builds and saves TF-IDF matrix
│   │   ├── TFIDFMatrix.java            # TF-IDF matrix structure
│   │   └── TFIDFLoader.java            # Loads saved TF-IDF data
│   │
│   ├── search/                         # Document search system
│   │   ├── SearchEngine.java           # Core search logic
│   │   ├── QueryVectorizer.java        # Converts queries to TF-IDF vectors
│   │   ├── CosineSimilarity.java       # Similarity computation
│   │   ├── SearchResult.java           # Result wrapper
│   │   └── SearchAPI.java              # REST API server (Spark)
│   │
│   ├── classification/
│   │   └── supervised/                 # Naive Bayes classifier
│   │       ├── NaiveBayesModel.java    # Model parameters storage
│   │       ├── NaiveBayesClassifier.java # Prediction logic
│   │       ├── NaiveBayesTrainer.java  # Training logic
│   │       ├── NBMain.java             # K-means clustering & document organization
│   │       └── LabelExtractor.java     # Extracts class from filename
│   │
│   ├── indexing/
│   │   └── CorpusFlattener.java        # Flattens corpus into single directory
│   │
│   ├── evaluation/                     # Model evaluation utilities
│   │   ├── ConfusionMatrix.java
│   │   ├── Metrics.java
│   │   └── CrossValidation.java
│   │
│   ├── utils/
│   │   ├── FileUtils.java
│   │   ├── SerializationUtils.java
│   │   └── MathUtils.java
│   │
│   └── Main.java                       # Application entry point
│
└── pom.xml                             # Maven configuration
```

## Implementation Details

### Phase 1: Text Preprocessing

**What it does:** Converts raw Arabic text into clean, normalized tokens.

**Implementation:** `TextPreprocessor.java` orchestrates a 3-step pipeline:

1. **Tokenization** (`Tokenizer.java`)
   - Splits text into individual tokens/words
   - Handles Arabic text properly

2. **Stop Word Removal** (`StopWordsFilter.java`)
   - Loads Arabic stopwords from `resources/stopwords/arabic_stopwords.txt`
   - Removes common words that don't carry semantic meaning
   - Example stopwords: في (in), من (from), هذا (this)

3. **Stemming** (`StemmerWrapper.java`)
   - Uses external Arabic stemmer library (`lib/arabic-stemmer.jar`)
   - Reduces words to their root form
   - Example: يكتب، كتب، كاتب → كتب (root form)

**Input:** Raw Arabic text from news sources
**Output:** List of normalized tokens

### Phase 2: TF-IDF Vectorization

**What it does:** Converts preprocessed documents into numerical vectors.

**Implementation:** `TFIDFBuilder.java`

The TF-IDF (Term Frequency-Inverse Document Frequency) algorithm creates a matrix where:
- **Rows** = Documents (from `resources/data/flattened_docs/`)
- **Columns** = Unique terms in vocabulary
- **Values** = TF-IDF scores

**Steps:**

1. **Load and Preprocess Documents**
   - Read all `.txt` files from `resources/data/flattened_docs/`
   - Apply the NLP preprocessing pipeline to each document
   - Build vocabulary (unique terms across all documents)

2. **Calculate Term Frequency (TF)**
   ```
   TF(term, doc) = count of term in doc / total words in doc
   ```

3. **Calculate Inverse Document Frequency (IDF)**
   ```
   IDF(term) = log(total documents / documents containing term)
   ```

4. **Compute TF-IDF**
   ```
   TF-IDF(term, doc) = TF(term, doc) × IDF(term)
   ```

5. **Serialize and Save**
   - Save TF-IDF matrix to `resources/models/tfidf/tfidf_matrix.bin`
   - Save vocabulary to `resources/models/tfidf/vocabulary.bin`
   - Save document list to `resources/models/tfidf/documents.bin`

**Output:** 
- `tfidf_matrix.bin` - 2D array of TF-IDF scores [num_docs × vocab_size]
- `vocabulary.bin` - Map from term to vocabulary index
- `documents.bin` - List of document filenames

### Phase 3: Document Search Engine

**What it does:** Enables full-text search over documents using TF-IDF similarity.

**Implementation:** 
- `SearchEngine.java` - Core search logic
- `QueryVectorizer.java` - Converts query to TF-IDF vector
- `CosineSimilarity.java` - Computes document relevance
- `SearchAPI.java` - REST API endpoint

**How it works:**

1. **Load TF-IDF Model** (`TFIDFLoader.java`)
   - Loads the saved TF-IDF matrix, vocabulary, and document list into memory

2. **Vectorize Query** (`QueryVectorizer.java`)
   - Apply same preprocessing pipeline to user query
   - Create TF-IDF vector from query terms using the saved vocabulary

3. **Compute Similarity** (`CosineSimilarity.java`)
   - Calculate cosine similarity between query vector and each document vector
   - Formula: `similarity = (A · B) / (||A|| × ||B||)`
   - Returns values between 0 and 1 (1 = perfect match)

4. **Rank Results**
   - Sort documents by similarity score (highest first)
   - Return top K results

**REST API Endpoints:**

```
GET /search?q=<query>
  Returns JSON array of top 10 results
  
GET /file?name=<filename>
  Returns full text content of a file
```

**Run Search API:**
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.search.SearchAPI"
```
Server listens on `http://localhost:4567`

### Phase 4: Naive Bayes Text Classifier

**What it does:** Classifies documents into one of 5 predefined categories.

**Implementation:**
- `NaiveBayesTrainer.java` - Trains the classifier
- `NaiveBayesClassifier.java` - Makes predictions
- `NaiveBayesModel.java` - Stores model parameters
- `NBMain.java` - Main entry point

**Training Process** (`NaiveBayesTrainer.java`):

1. **Extract Labels**
   - Documents are labeled based on filename patterns extracted by `LabelExtractor.java`
   - Filters for documents in allowed classes: مجتمع, سياسة, اقتصاد, رياضة, فنون

2. **Calculate Class Priors**
   ```
   P(Class) = log(count of docs in class / total docs)
   ```

3. **Calculate Conditional Probabilities**
   - For each term and class, calculate: P(term | class)
   - Uses Laplace smoothing to avoid zero probabilities:
   ```
   P(term | class) = log((term count in class + 1) / (total terms in class + vocab size))
   ```

4. **Store Model Parameters**
   - Class priors: `Map<String, Double> classPriors`
   - Conditional probabilities: `Map<String, Map<Integer, Double>> condProb`
   - Vocabulary size

**Prediction Process** (`NaiveBayesClassifier.java`):

For a document vector, compute score for each class:
```
Score(class) = log(P(class)) + Σ log(P(term | class)) for all terms in document
```
Return the class with the highest score.

**Run Classifier:**
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.classification.supervised.NBMain"
```

**What NBMain does:**
1. Loads the TF-IDF matrix
2. Trains Naive Bayes model on documents
3. Uses K-means clustering (k=5) to organize documents
4. Creates output directory: `resources/data/classified_clusters/`
5. Organizes documents into subdirectories by predicted category

**Output:**
```
classified_clusters/
├── مجتمع/           # Society documents
├── سياسة/           # Politics documents
├── اقتصاد/           # Economics documents
├── رياضة/           # Sports documents
└── فنون/            # Arts documents
```

## Building & Running

### Prerequisites
- Java 11+
- Maven 3.6+
- Windows/Linux/Mac

### Build Project

```bash
mvn clean install
```

### Run Components

**1. Build TF-IDF Model** (if not already done)
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.vectorization.TFIDFBuilder"
```
Creates:
- `resources/models/tfidf/tfidf_matrix.bin`
- `resources/models/tfidf/vocabulary.bin`
- `resources/models/tfidf/documents.bin`

**2. Start Search API Server**
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.search.SearchAPI"
```
Server runs on `http://localhost:4567`

**3. Run Naive Bayes Classifier**
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.classification.supervised.NBMain"
```
Creates classified clusters in `resources/data/classified_clusters/`

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Java 11 |
| Build Tool | Maven 3.6+ |
| Stemming | Custom Arabic Stemmer Library |
| Web Server | Spark Framework 2.9.4 |
| JSON | Google GSON 2.10.1 |
| Testing | JUnit 4.13.2 |

## Key Design Patterns

1. **Serialization Pattern**: TF-IDF model is built once and serialized to disk, then loaded by search and classification modules
2. **Pipeline Pattern**: NLP preprocessing follows a clear pipeline (tokenize → filter → stem)
3. **Vector Space Model**: Documents represented as vectors in high-dimensional space (vocab size)
4. **REST API Pattern**: Search functionality exposed via HTTP endpoints

## Performance Characteristics

- **Preprocessing**: O(n) where n = total number of tokens
- **TF-IDF Building**: O(n × log d) where d = number of documents
- **Search**: O(d × v) where v = vocabulary size (matrix-vector multiplication)
- **Classification**: O(d × v) for training, O(v) per document for prediction

