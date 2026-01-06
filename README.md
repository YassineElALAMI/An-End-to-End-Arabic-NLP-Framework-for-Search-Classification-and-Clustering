# Arabic NLP System

A comprehensive **Natural Language Processing (NLP)** system for Arabic text classification and semantic search. This project demonstrates the complete NLP pipeline from data preprocessing, vectorization, search capabilities, to both supervised and unsupervised classification with comprehensive evaluation metrics.

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Implementation Guide](#implementation-guide)
   - [1. Data Flattening](#1-data-flattening)
   - [2. Text Preprocessing Pipeline](#2-text-preprocessing-pipeline)
   - [3. TF-IDF Vectorization](#3-tf-idf-vectorization)
   - [4. Search Engine](#4-search-engine)
   - [5. Supervised Classification - Naive Bayes](#5-supervised-classification---naive-bayes)
   - [6. Unsupervised Clustering - K-Means](#6-unsupervised-clustering---k-means)
   - [7. Unsupervised Clustering - Hierarchical](#7-unsupervised-clustering---hierarchical)
   - [8. Evaluation & Cross-Validation](#8-evaluation--cross-validation)
4. [Building & Running](#building--running)
5. [Technology Stack](#technology-stack)

---

## Project Overview

The system processes Arabic text documents from multiple news sources and organizes them into **5 categories**:
- **مجتمع** (Society)
- **سياسة** (Politics)  
- **اقتصاد** (Economics)
- **رياضة** (Sports)
- **فنون** (Arts)

**Key Features:**
- ✅ Full-text document search with semantic relevance ranking
- ✅ Supervised text classification (Naive Bayes)
- ✅ Unsupervised document clustering (K-Means & Hierarchical)
- ✅ Cross-validation for model evaluation
- ✅ Comprehensive classification metrics (Precision, Recall, F1, Confusion Matrix)

---

## System Architecture

### Directory Structure

```
ArabicNLPSystem/
├── lib/
│   └── arabic-stemmer.jar              # Arabic stemming library
│
├── resources/
│   ├── corpus/                         # Original news documents organized by source & category
│   │   ├── agadir24/
│   │   ├── Hespres/
│   │   ├── Medi1TV/
│   │   └── Voice of morocco/
│   │
│   ├── data/
│   │   ├── flattened_docs/             # All preprocessed documents in single directory
│   │   ├── classified_clusters/        # Documents organized by predicted category
│   │   └── HierarchicalClusters/       # Hierarchical clustering output
│   │
│   ├── models/
│   │   ├── tfidf/                      # TF-IDF matrix & vocabulary (serialized)
│   │   ├── kmeans/                     # K-Means model output
│   │   ├── hierarchical/               # Hierarchical clustering output
│   │   └── naive_bayes/                # Naive Bayes model (if saved)
│   │
│   └── stopwords/
│       └── arabic_stopwords.txt        # Arabic stopwords list
│
├── src/main/java/ma/yassine/arabicnlp/
│   ├── nlp/                            # Text preprocessing
│   │   ├── Tokenizer.java
│   │   ├── TextPreprocessor.java
│   │   ├── StopWordsFilter.java
│   │   └── StemmerWrapper.java
│   │
│   ├── indexing/
│   │   └── CorpusFlattener.java        # Flattens corpus structure
│   │
│   ├── vectorization/
│   │   └── TFIDFBuilder.java           # Builds TF-IDF matrix
│   │
│   ├── search/
│   │   ├── SearchEngine.java
│   │   ├── SearchAPI.java              # REST API server
│   │   ├── QueryVectorizer.java
│   │   ├── CosineSimilarity.java
│   │   ├── TFIDFLoader.java
│   │   └── SearchResult.java
│   │
│   ├── classification/
│   │   ├── supervised/
│   │   │   ├── NaiveBayesTrainer.java
│   │   │   ├── NaiveBayesClassifier.java
│   │   │   ├── NBMain.java
│   │   │   ├── LabelExtractor.java
│   │   │   └── NaiveBayesModel.java
│   │   │
│   │   └── unsupervised/
│   │       ├── KMeans.java
│   │       ├── KMeansMain.java
│   │       ├── HierarchicalClustering.java
│   │       └── HierarchicalMain.java
│   │
│   ├── evaluation/
│   │   ├── CrossValidation.java        # K-fold cross-validation
│   │   ├── ConfusionMatrix.java
│   │   ├── Metrics.java                # Precision, Recall, F1-Score
│   │   ├── ClusteringEvaluator.java
│   │   └── MetricsMain.java
│   │
│   └── utils/
│       ├── FileUtils.java
│       ├── SerializationUtils.java
│       └── MathUtils.java
│
└── pom.xml
```

---

## Implementation Guide

### 1. Data Flattening

**Purpose:** Organize hierarchical corpus structure into a single flat directory for processing.

**Input Structure:**
```
corpus/agadir24/الاقتصاد/f1.txt
corpus/agadir24/سياسية/f2.txt
corpus/Medi1TV/اخبار/f1.txt
```

**Implementation:** [CorpusFlattener.java](src/main/java/ma/yassine/arabicnlp/indexing/CorpusFlattener.java)

**Process:**
1. Recursively traverse all directories in `resources/corpus/`
2. For each text file, extract path components:
   - Source (e.g., "agadir24")
   - Category (e.g., "الاقتصاد")
   - Original filename (e.g., "f1.txt")
3. Create flattened filename: `f1_agadir24_الاقتصاد.txt`
4. Copy file to `resources/data/flattened_docs/`

**Output:**
```
flattened_docs/
├── f1_agadir24_الاقتصاد والمال.txt
├── f1_agadir24_سياسية.txt
├── f1_Medi1TV_اخبار_المغرب.txt
└── ... (all documents in one directory)
```

**Run:**
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.indexing.CorpusFlattener"
```

---

### 2. Text Preprocessing Pipeline

**Purpose:** Convert raw Arabic text into clean, normalized tokens.

**Implementation:** [TextPreprocessor.java](src/main/java/ma/yassine/arabicnlp/nlp/TextPreprocessor.java)

The pipeline consists of **3 sequential steps**:

#### Step 1: Tokenization
**File:** [Tokenizer.java](src/main/java/ma/yassine/arabicnlp/nlp/Tokenizer.java)

- Splits text into individual words/tokens
- Handles Arabic diacritics and special characters
- Preserves word boundaries

**Example:**
```
Input:  "هذا نص عربي مثير للاهتمام"
Output: ["هذا", "نص", "عربي", "مثير", "للاهتمام"]
```

#### Step 2: Stop Words Removal
**File:** [StopWordsFilter.java](src/main/java/ma/yassine/arabicnlp/nlp/StopWordsFilter.java)

- Loads Arabic stopwords from `resources/stopwords/arabic_stopwords.txt`
- Removes common words that don't carry semantic meaning
- Example stopwords: في (in), من (from), هذا (this), ال (the)

**Example:**
```
Input:  ["هذا", "نص", "عربي", "مثير", "للاهتمام"]
Output: ["نص", "عربي", "مثير", "اهتمام"]  # "هذا" and "ال" removed
```

#### Step 3: Stemming
**File:** [StemmerWrapper.java](src/main/java/ma/yassine/arabicnlp/nlp/StemmerWrapper.java)

- Uses external Arabic stemmer library (`lib/arabic-stemmer.jar`)
- Reduces words to their morphological root
- Crucial for Arabic NLP (many forms of same word)

**Example:**
```
Input:  ["نص", "عربي", "مثير", "اهتمام"]
Output: ["نص", "عرب", "ثير", "هتم"]  # Root forms
```

**Full Pipeline Example:**
```
Text: "هذا نص عربي مثير جدا للاهتمام"
  ↓ Tokenization
["هذا", "نص", "عربي", "مثير", "جدا", "للاهتمام"]
  ↓ Stop words removal  
["نص", "عربي", "مثير", "جدا", "اهتمام"]
  ↓ Stemming
["نص", "عرب", "ثير", "جد", "هتم"]
```

---

### 3. TF-IDF Vectorization

**Purpose:** Convert preprocessed documents into numerical vectors for machine learning.

**Implementation:** [TFIDFBuilder.java](src/main/java/ma/yassine/arabicnlp/vectorization/TFIDFBuilder.java)

#### TF-IDF Concept

TF-IDF is a numerical statistic that reflects how important a term is to a document in a collection.

**Formula:**
```
TF-IDF(term, doc) = TF(term, doc) × IDF(term)

where:
  TF(term, doc)  = (count of term in doc) / (total words in doc)
  IDF(term)      = log((D + 1) / (df(term) + 1)) + 1
  
  D              = total number of documents
  df(term)       = number of documents containing term
```

The smoothing prevents division by zero and handles unseen terms.

#### Building Process

**Step 1: Load and Preprocess**
```java
// Read all documents from flattened_docs/
List<Path> files = Files.list(Paths.get("resources/data/flattened_docs"))
    .filter(p -> p.toString().endsWith(".txt"))
    .toList();

// Preprocess each document
Map<String, List<String>> documents = new LinkedHashMap<>();
for (Path file : files) {
    String content = Files.readString(file, StandardCharsets.UTF_8);
    List<String> tokens = TextPreprocessor.preprocess(content);  // Applies full pipeline
    documents.put(file.getFileName().toString(), tokens);
}
```

**Step 2: Build Vocabulary**
```java
// Collect all unique terms
Map<String, Integer> vocabulary = new LinkedHashMap<>();
for (List<String> tokens : documents.values()) {
    for (String token : tokens) {
        vocabulary.putIfAbsent(token, vocabulary.size());
    }
}
```

**Step 3: Calculate Term Frequency (TF)**
```java
// For each document
Map<String, Long> tf = tokens.stream()
    .collect(Collectors.groupingBy(t -> t, Collectors.counting()));

for (Map.Entry<String, Long> entry : tf.entrySet()) {
    String term = entry.getKey();
    double tfValue = entry.getValue() / (double) tokens.size();
}
```

**Step 4: Calculate Document Frequency (DF) and IDF**
```java
// Count how many documents contain each term
Map<String, Integer> df = new HashMap<>();
for (List<String> tokens : documents.values()) {
    Set<String> uniqueTerms = new HashSet<>(tokens);
    for (String term : uniqueTerms) {
        df.put(term, df.getOrDefault(term, 0) + 1);
    }
}

// Calculate IDF
double idfValue = Math.log((D + 1.0) / (df.get(term) + 1.0)) + 1.0;
```

**Step 5: Compute TF-IDF Matrix**
```java
// Create D × V matrix (D=documents, V=vocabulary size)
double[][] tfidf = new double[D][V];

for (int d = 0; d < D; d++) {
    for (int t = 0; t < V; t++) {
        double tfValue = ...;
        double idfValue = ...;
        tfidf[d][t] = tfValue * idfValue;
    }
}
```

#### Output Files

The builder saves:
- **tfidf_matrix.ser** - 2D array [numDocs × vocabSize]
- **vocabulary.ser** - Map<String, Integer> for term→index lookup
- **doc_index.ser** - List<String> document filenames

**Run:**
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.vectorization.TFIDFBuilder"
```

---

### 4. Search Engine

**Purpose:** Enable full-text semantic search over document corpus.

**Implementation:** 
- [SearchEngine.java](src/main/java/ma/yassine/arabicnlp/search/SearchEngine.java) - Search logic
- [SearchAPI.java](src/main/java/ma/yassine/arabicnlp/search/SearchAPI.java) - REST API server
- [QueryVectorizer.java](src/main/java/ma/yassine/arabicnlp/search/QueryVectorizer.java) - Query encoding
- [CosineSimilarity.java](src/main/java/ma/yassine/arabicnlp/search/CosineSimilarity.java) - Similarity computation

#### How It Works

**Step 1: Load TF-IDF Model** ([TFIDFLoader.java](src/main/java/ma/yassine/arabicnlp/search/TFIDFLoader.java))
```java
TFIDFLoader.load("resources/models/tfidf");
// Loads into memory:
// - TFIDFLoader.tfidfMatrix: double[][]
// - TFIDFLoader.vocabulary: Map<String, Integer>
// - TFIDFLoader.documents: List<String>
```

**Step 2: Vectorize Query** ([QueryVectorizer.java](src/main/java/ma/yassine/arabicnlp/search/QueryVectorizer.java))

Convert user query to TF-IDF vector using same preprocessing:
```java
String query = "اقتصاد المغرب";
  ↓ TextPreprocessor.preprocess()
["قتصاد", "مغرب"]  // After stemming
  ↓ Create TF-IDF vector
double[] queryVector = new double[vocabSize];
queryVector[indexOf("قتصاد")] = tfidf_value;
queryVector[indexOf("مغرب")] = tfidf_value;
```

**Step 3: Compute Similarity** ([CosineSimilarity.java](src/main/java/ma/yassine/arabicnlp/search/CosineSimilarity.java))

Calculate cosine similarity between query vector and each document:
```
Cosine Similarity = (A · B) / (||A|| × ||B||)

where:
  A · B = Σ(a_i × b_i)           [dot product]
  ||A|| = √(Σ(a_i²))             [magnitude of A]
  ||B|| = √(Σ(b_i²))             [magnitude of B]

Result: value between 0 and 1
  - 0 = completely dissimilar
  - 1 = perfect match
```

**Step 4: Rank Results**
```java
// Sort documents by similarity score (highest first)
results.sort((a, b) -> Double.compare(b.score, a.score));

// Return top K results
return results.subList(0, Math.min(topK, results.size()));
```

#### REST API Endpoints

**Search Documents:**
```bash
GET /search?q=<query>&k=10

Example:
  GET /search?q=سياسة+المغرب&k=10
  
Response:
  [
    {
      "filename": "f1_Medi1TV_سياسة_المغرب.txt",
      "score": 0.85
    },
    {
      "filename": "f2_agadir24_سياسية.txt",
      "score": 0.72
    }
  ]
```

**Get Document Content:**
```bash
GET /file?name=<filename>

Example:
  GET /file?name=f1_Medi1TV_سياسة_المغرب.txt
  
Response:
  [Raw text content of the file]
```

**Run Search API Server:**
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.search.SearchAPI"
```
Server listens on `http://localhost:4567`

---

### 5. Supervised Classification - Naive Bayes

**Purpose:** Classify documents into predefined categories using probabilistic approach.

**Implementation:**
- [NaiveBayesTrainer.java](src/main/java/ma/yassine/arabicnlp/classification/supervised/NaiveBayesTrainer.java) - Model training
- [NaiveBayesClassifier.java](src/main/java/ma/yassine/arabicnlp/classification/supervised/NaiveBayesClassifier.java) - Prediction
- [NaiveBayesModel.java](src/main/java/ma/yassine/arabicnlp/classification/supervised/NaiveBayesModel.java) - Model storage
- [LabelExtractor.java](src/main/java/ma/yassine/arabicnlp/classification/supervised/LabelExtractor.java) - Label extraction from filename

#### Naive Bayes Classifier

**Concept:** Uses Bayes' theorem to compute probability of each class given document features.

**Formula:**
```
P(Class | Document) = P(Document | Class) × P(Class) / P(Document)

For simplicity, we ignore P(Document) and compute:
  Class = argmax_c [ P(Class=c) × Π P(term_i | Class=c) ]

Using log to prevent underflow:
  Class = argmax_c [ log(P(c)) + Σ log(P(term_i | c)) ]
```

#### Training Process

**Step 1: Extract Labels** ([LabelExtractor.java](src/main/java/ma/yassine/arabicnlp/classification/supervised/LabelExtractor.java))

Parse document filename to extract true category:
```
Filename: f1_Medi1TV_سياسة_المغرب.txt
              └─ Label: سياسة
```

**Step 2: Count Class Occurrences**
```java
// Count how many documents are in each class
Map<String, Double> classCounts = new HashMap<>();
for (int d = 0; d < D; d++) {
    String label = LabelExtractor.extractLabel(documents.get(d));
    classCounts.put(label, classCounts.getOrDefault(label, 0.0) + 1);
}
```

**Step 3: Calculate Class Priors**
```java
// P(Class) = log(count of docs in class / total docs)
Map<String, Double> priors = new HashMap<>();
for (String c : classCounts.keySet()) {
    priors.put(c, Math.log(classCounts.get(c) / D));
}
```

**Step 4: Count Term Occurrences per Class**
```java
// For each term in each class, sum TF-IDF values
Map<String, Map<Integer, Double>> termCounts = new HashMap<>();
for (int d = 0; d < D; d++) {
    String label = LabelExtractor.extractLabel(documents.get(d));
    termCounts.putIfAbsent(label, new HashMap<>());
    
    for (int t = 0; t < V; t++) {
        double val = tfidfMatrix[d][t];
        if (val > 0) {
            termCounts.get(label).put(t, 
                termCounts.get(label).getOrDefault(t, 0.0) + val);
        }
    }
}
```

**Step 5: Calculate Conditional Probabilities** (with Laplace smoothing)
```java
// P(term | class) = log((term_count + 1) / (total_terms_in_class + vocab_size))
Map<String, Map<Integer, Double>> condProb = new HashMap<>();
for (String c : termCounts.keySet()) {
    Map<Integer, Double> probs = new HashMap<>();
    double sum = termCounts.get(c).values().stream().mapToDouble(x -> x).sum();
    
    for (int t = 0; t < V; t++) {
        double count = termCounts.get(c).getOrDefault(t, 0.0);
        double prob = Math.log((count + 1) / (sum + V));  // Laplace smoothing
        probs.put(t, prob);
    }
    condProb.put(c, probs);
}
```

#### Prediction Process

**Step 1: Encode Document as TF-IDF Vector**
```
Document → TextPreprocessor → TF-IDF Vector
```

**Step 2: Score Each Class**
```java
// For each class, compute score
Map<String, Double> classScores = new HashMap<>();
for (String c : model.classPriors.keySet()) {
    double score = model.classPriors.get(c);  // log P(class)
    
    for (int t = 0; t < tfidfVector.length; t++) {
        if (tfidfVector[t] > 0) {
            double termProb = model.condProb.get(c).get(t);
            score += tfidfVector[t] * termProb;  // Add log P(term | class)
        }
    }
    classScores.put(c, score);
}
```

**Step 3: Predict Class**
```java
// Return class with highest score
String prediction = classScores.entrySet().stream()
    .max(Map.Entry.comparingByValue())
    .get()
    .getKey();
```

#### Example Usage

**Run Training & Classification:**
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.classification.supervised.NBMain"
```

**What NBMain Does:**
1. Loads TF-IDF matrix from `resources/models/tfidf/`
2. Trains Naive Bayes model on all documents
3. Classifies each document
4. Organizes documents into category directories:
```
resources/data/classified_clusters/
├── مجتمع/           # Society documents
├── سياسة/           # Politics documents
├── اقتصاد/           # Economics documents
├── رياضة/           # Sports documents
└── فنون/            # Arts documents
```

---

### 6. Unsupervised Clustering - K-Means

**Purpose:** Automatically group similar documents without predefined labels.

**Implementation:**
- [KMeans.java](src/main/java/ma/yassine/arabicnlp/classification/unsupervised/KMeans.java) - Core algorithm
- [KMeansMain.java](src/main/java/ma/yassine/arabicnlp/classification/unsupervised/KMeansMain.java) - Entry point

#### K-Means Algorithm

**Concept:** Partition documents into K clusters by minimizing within-cluster variance.

**Formula:**
```
Objective: minimize Σ_c Σ_x∈C_c ||x - centroid_c||²

where:
  C_c = cluster c
  centroid_c = mean of all points in cluster c
  ||·|| = Euclidean distance
```

#### Algorithm Steps

**Step 1: Initialize Centroids**
```java
// Randomly select K documents as initial centroids
Random rand = new Random(42);  // Fixed seed for reproducibility
Set<Integer> chosen = new HashSet<>();
for (int i = 0; i < K; i++) {
    int idx = rand.nextInt(data.length);
    centroids[i] = Arrays.copyOf(data[idx], data[idx].length);
    chosen.add(idx);
}
```

**Step 2: Assign Documents to Nearest Centroid**
```java
// For each document, find nearest centroid (minimum Euclidean distance)
for (int i = 0; i < data.length; i++) {
    double bestDist = Double.MAX_VALUE;
    int bestCluster = -1;
    
    for (int c = 0; c < K; c++) {
        double dist = euclidean(data[i], centroids[c]);
        if (dist < bestDist) {
            bestDist = dist;
            bestCluster = c;
        }
    }
    labels[i] = bestCluster;
}
```

**Step 3: Recompute Centroids**
```java
// Update each centroid to mean of assigned documents
double[][] newCentroids = new double[K][data[0].length];
int[] counts = new int[K];

for (int i = 0; i < data.length; i++) {
    int c = labels[i];
    counts[c]++;
    for (int j = 0; j < data[i].length; j++) {
        newCentroids[c][j] += data[i][j];
    }
}

for (int c = 0; c < K; c++) {
    if (counts[c] > 0) {
        for (int j = 0; j < newCentroids[c].length; j++) {
            newCentroids[c][j] /= counts[c];  // Average
        }
    }
}
centroids = newCentroids;
```

**Step 4: Iterate Until Convergence**
```java
// Repeat steps 2-3 until centroids don't change (or max iterations)
for (int iter = 0; iter < maxIterations; iter++) {
    boolean changed = assignClusters();
    recomputeCentroids();
    if (!changed) break;  // Converged
}
```

**Euclidean Distance:**
```java
double euclidean(double[] x, double[] y) {
    double sum = 0;
    for (int i = 0; i < x.length; i++) {
        double d = x[i] - y[i];
        sum += d * d;
    }
    return Math.sqrt(sum);
}
```

#### Run K-Means Clustering

```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.classification.unsupervised.KMeansMain"
```

**Output:** Documents organized by K-Means clusters in `resources/data/KMclusters/`

---

### 7. Unsupervised Clustering - Hierarchical

**Purpose:** Build a hierarchical tree of document clusters for exploratory analysis.

**Implementation:**
- [HierarchicalClustering.java](src/main/java/ma/yassine/arabicnlp/classification/unsupervised/HierarchicalClustering.java) - Core algorithm
- [HierarchicalMain.java](src/main/java/ma/yassine/arabicnlp/classification/unsupervised/HierarchicalMain.java) - Entry point

#### Agglomerative Hierarchical Clustering

**Concept:** Start with each document as its own cluster, then repeatedly merge closest clusters until target number of clusters is reached.

**Algorithm:**

**Step 1: Initialize Clusters**
```java
// Each document is its own cluster initially
List<Set<Integer>> clusters = new ArrayList<>();
for (int i = 0; i < data.length; i++) {
    Set<Integer> c = new HashSet<>();
    c.add(i);
    clusters.add(c);
}
```

**Step 2: Find Closest Pair of Clusters**
```java
// Find pair of clusters with minimum distance (single-link)
double minDist = Double.MAX_VALUE;
int c1 = -1, c2 = -1;

for (int i = 0; i < clusters.size(); i++) {
    for (int j = i + 1; j < clusters.size(); j++) {
        double dist = clusterDistance(clusters.get(i), clusters.get(j));
        if (dist < minDist) {
            minDist = dist;
            c1 = i;
            c2 = j;
        }
    }
}
```

**Step 3: Merge Closest Clusters**
```java
// Merge cluster j into cluster i
clusters.get(c1).addAll(clusters.get(c2));
clusters.remove(c2);  // Remove the merged cluster
```

**Step 4: Repeat Until Target Clusters**
```java
while (clusters.size() > targetClusters) {
    mergeClosestClusters();
}
```

**Single-Link Distance (Minimum Distance):**
```java
// Distance between clusters = minimum distance between any two points
double clusterDistance(Set<Integer> a, Set<Integer> b) {
    double min = Double.MAX_VALUE;
    for (int i : a) {
        for (int j : b) {
            double d = euclidean(data[i], data[j]);
            if (d < min) min = d;
        }
    }
    return min;
}
```

#### Run Hierarchical Clustering

```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.classification.unsupervised.HierarchicalMain"
```

**Output:** Documents organized by hierarchical clusters in `resources/data/HierarchicalClusters/`

---

### 8. Evaluation & Cross-Validation

**Purpose:** Rigorously evaluate supervised classifier performance using systematic methodology.

**Implementation:**
- [CrossValidation.java](src/main/java/ma/yassine/arabicnlp/evaluation/CrossValidation.java) - K-fold cross-validation
- [ConfusionMatrix.java](src/main/java/ma/yassine/arabicnlp/evaluation/ConfusionMatrix.java) - Confusion matrix
- [Metrics.java](src/main/java/ma/yassine/arabicnlp/evaluation/Metrics.java) - Classification metrics
- [MetricsMain.java](src/main/java/ma/yassine/arabicnlp/evaluation/MetricsMain.java) - Evaluation entry point

#### K-Fold Cross-Validation

**Concept:** Divide data into K folds; train K models (each leaving out one fold) and average results for unbiased evaluation.

**Why Cross-Validation?**
- Prevents overfitting to a single test set
- Uses all data for both training and testing
- Provides more reliable performance estimate
- Standard practice in machine learning

#### Algorithm

**Step 1: Shuffle Data**
```java
int N = TFIDFLoader.tfidfMatrix.length;
List<Integer> indices = new ArrayList<>();
for (int i = 0; i < N; i++) indices.add(i);
Collections.shuffle(indices, new Random(42));  // Fixed seed for reproducibility
```

**Step 2: Split into K Folds**
```java
int foldSize = N / k;
for (int fold = 0; fold < k; fold++) {
    int start = fold * foldSize;
    int end = (fold == k - 1) ? N : start + foldSize;
    
    Set<Integer> testIdx = new HashSet<>(indices.subList(start, end));
    Set<Integer> trainIdx = new HashSet<>();
    for (int i = 0; i < N; i++) {
        if (!testIdx.contains(i)) trainIdx.add(i);
    }
}
```

**Step 3: Train & Test Each Fold**
```java
// Train on all documents EXCEPT test fold
NaiveBayesModel model = NaiveBayesTrainer.trainWithSubset(testIdx, false);

// Evaluate on test fold
ConfusionMatrix cm = new ConfusionMatrix();
for (int i : testIdx) {
    String trueLabel = LabelExtractor.extractLabel(documents.get(i));
    String predLabel = NaiveBayesClassifier.predict(tfidfMatrix[i], model);
    cm.add(trueLabel, predLabel);
}
```

**Step 4: Compute Metrics**
```java
double f1 = Metrics.macroF1(cm);
totalF1 += f1;
System.out.println("Fold " + (fold + 1) + " Macro-F1 = " + f1);
```

**Step 5: Average Results**
```java
double averageF1 = totalF1 / k;
System.out.println("AVERAGE MACRO-F1 (" + k + "-fold) = " + averageF1);
```

#### Classification Metrics

**Confusion Matrix:**
A matrix showing predicted vs. actual labels for all classes.

```
                 Predicted
              مجتمع  سياسة  اقتصاد  رياضة  فنون
Actual مجتمع    45     2      1      0     2
      سياسة     3    42      2      1     2
      اقتصاد     1     2     43      1     3
      رياضة      0     1      1     44     4
      فنون       2     1      3      2    42

Where diagonal = correct predictions
```

**Per-Class Metrics:**

```
Precision(class) = TP / (TP + FP)
  - Of documents we predicted as class, how many were actually class?
  - Measures: how trustworthy are our positive predictions?

Recall(class) = TP / (TP + FN)  
  - Of all documents that should be class, how many did we find?
  - Measures: completeness of finding all positives

F1-Score(class) = 2 × (Precision × Recall) / (Precision + Recall)
  - Harmonic mean of Precision and Recall
  - Balances both metrics
  - Good for imbalanced datasets
```

**Implementation:**
```java
public static double precision(String label, ConfusionMatrix cm) {
    int tp = cm.get(label, label);           // Correctly predicted as this class
    int fp = 0;
    for (String l : cm.labels()) {
        if (!l.equals(label)) 
            fp += cm.get(l, label);          // Incorrectly predicted as this class
    }
    return tp + fp == 0 ? 0 : (double) tp / (tp + fp);
}

public static double recall(String label, ConfusionMatrix cm) {
    int tp = cm.get(label, label);
    int fn = 0;
    for (String l : cm.labels()) {
        if (!l.equals(label))
            fn += cm.get(label, l);          // This class predicted as something else
    }
    return tp + fn == 0 ? 0 : (double) tp / (tp + fn);
}

public static double f1(String label, ConfusionMatrix cm) {
    double p = precision(label, cm);
    double r = recall(label, cm);
    return p + r == 0 ? 0 : 2 * p * r / (p + r);
}
```

**Overall Metrics:**

```
Accuracy = (Total Correct) / (Total Samples)
  - Overall correctness across all classes

Macro-F1 = Average(F1-Score for each class)
  - Treats all classes equally (doesn't weight by frequency)
  - Good for balanced evaluation
```

#### Evaluation Workflow

**Run Full Evaluation:**
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.evaluation.MetricsMain"
```

**What MetricsMain Does:**
1. Loads TF-IDF matrix from `resources/models/tfidf/`
2. Trains Naive Bayes on all documents
3. Predicts label for each document
4. Builds confusion matrix
5. Computes and displays all metrics

**Sample Output:**
```
CONFUSION MATRIX:
               مجتمع  سياسة  اقتصاد  رياضة  فنون
مجتمع           45     2      1      0     2
سياسة            3    42      2      1     2
اقتصاد            1     2     43      1     3
رياضة             0     1      1     44     4
فنون              2     1      3      2    42

CLASSIFICATION METRICS:

مجتمع:  Precision = 0.900  Recall = 0.900  F1-Score = 0.900
سياسة:  Precision = 0.880  Recall = 0.840  F1-Score = 0.860
اقتصاد:  Precision = 0.860  Recall = 0.860  F1-Score = 0.860
رياضة:  Precision = 0.880  Recall = 0.880  F1-Score = 0.880
فنون:   Precision = 0.840  Recall = 0.840  F1-Score = 0.840

OVERALL:
Accuracy   = 0.88
Macro-F1   = 0.87
```

**Interpreting Results:**
- Macro-F1 = 0.87 means on average, 87% perfect balance between finding all documents of a class and ensuring predictions are correct
- Higher F1 = better classifier performance
- Compare macro-F1 across CV folds to assess stability

---

## Building & Running

### Prerequisites
- **Java 11+** (JDK)
- **Maven 3.6+**
- Windows/Linux/Mac with UTF-8 support

### Complete Workflow

**1. Build Project**
```bash
mvn clean install
```

**2. Flatten Corpus** (one-time setup)
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.indexing.CorpusFlattener"
```
Creates: `resources/data/flattened_docs/`

**3. Build TF-IDF Model** (one-time setup)
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.vectorization.TFIDFBuilder"
```
Creates: `resources/models/tfidf/`

**4. Run Search API** (persistent server)
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.search.SearchAPI"
```
Listens on: `http://localhost:4567`

**5. Classify with Naive Bayes** (one-time)
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.classification.supervised.NBMain"
```
Creates: `resources/data/classified_clusters/`

**6. Cluster with K-Means** (one-time)
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.classification.unsupervised.KMeansMain"
```
Creates: `resources/data/KMclusters/`

**7. Cluster Hierarchically** (one-time)
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.classification.unsupervised.HierarchicalMain"
```
Creates: `resources/data/HierarchicalClusters/`

**8. Evaluate Classifier** (one-time)
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.evaluation.MetricsMain"
```
Displays: Confusion matrix and metrics

**9. Cross-Validate** (one-time)
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.evaluation.CrossValidation"
# (May need to run with arguments for fold count)
```

---

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Java | 11+ |
| Build Tool | Maven | 3.6+ |
| Web Framework | Spark Framework | 2.9.4 |
| JSON Library | Google GSON | 2.10.1 |
| Arabic Stemming | Custom Library | 1.0 |
| Testing | JUnit | 4.13.2 |

---

## Key Takeaways

**This project demonstrates:**

1. **Complete NLP Pipeline** - From raw text to predictions
2. **Vector Space Model** - Document representation as high-dimensional vectors
3. **Information Retrieval** - Semantic search using cosine similarity
4. **Supervised Learning** - Probabilistic classification with Naive Bayes
5. **Unsupervised Learning** - Clustering with K-Means and Hierarchical methods
6. **Rigorous Evaluation** - Cross-validation and comprehensive metrics
7. **Arabic Language Processing** - Tokenization, stopwords, stemming for Arabic text
8. **REST APIs** - Exposing ML functionality via web services

This is a production-ready example of modern NLP engineering practices!

## Project Overview

The Arabic NLP System processes Arabic text documents from multiple news sources (Agadir24, Hespres, Medi1TV, Voice of Morocco) and organizes them into 5 categories:
- **مجتمع** (Society)
- **سياسة** (Politics)
- **اقتصاد** (Economics)
- **رياضة** (Sports)
- **فنون** (Arts)


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

## Evaluation & Metrics

**What it does:** Evaluates the performance of the Naive Bayes classifier using various metrics.

**Implementation:**
- `ConfusionMatrix.java` - Builds and manages confusion matrix
- `Metrics.java` - Computes classification metrics
- `MetricsMain.java` - Main evaluation entry point

**Evaluation Process** (`MetricsMain.java`):

1. **Load Trained Models**
   - Loads TF-IDF model from `resources/models/tfidf/`
   - Loads Naive Bayes model from `resources/models/naive_bayes/`

2. **Generate Predictions**
   - Converts all documents to TF-IDF vectors
   - Classifies each document using the trained Naive Bayes classifier
   - Extracts true labels from document filenames

3. **Build Confusion Matrix**
   - Compares predicted vs. actual labels for all documents
   - Creates a 5×5 matrix (one row/column per category)
   - Rows = true labels, Columns = predicted labels

4. **Calculate Metrics**
   ```
   Accuracy = (TP + TN) / Total Samples
   Precision(class) = TP / (TP + FP)
   Recall(class) = TP / (TP + FN)
   F1-Score(class) = 2 × (Precision × Recall) / (Precision + Recall)
   Macro-F1 = Average F1-Score across all classes
   ```

5. **Display Results**
   - Prints confusion matrix
   - Shows per-class metrics (Precision, Recall, F1-Score)
   - Displays overall accuracy and macro-F1 score

**Run Evaluation:**
```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.evaluation.MetricsMain"
```

**Sample Output:**
```
Confusion Matrix:
           مجتمع  سياسة  اقتصاد  رياضة  فنون
مجتمع      45     2      1      0     2
سياسة       3    42      2      1     2
اقتصاد       1     2     43      1     3
رياضة        0     1      1     44     4
فنون         2     1      3      2    42

Classification Metrics:
مجتمع:  Precision=0.90  Recall=0.90  F1-Score=0.90
سياسة:  Precision=0.88  Recall=0.84  F1-Score=0.86
اقتصاد:  Precision=0.86  Recall=0.86  F1-Score=0.86
رياضة:  Precision=0.88  Recall=0.88  F1-Score=0.88
فنون:   Precision=0.84  Recall=0.84  F1-Score=0.84

Overall Accuracy: 0.88
Macro-F1 Score: 0.87
```

**Key Metrics Explained:**

- **Accuracy**: Overall correctness of predictions
- **Precision**: Of documents predicted as class X, how many were actually class X
- **Recall**: Of all documents that should be class X, how many were correctly identified
- **F1-Score**: Harmonic mean of Precision and Recall (good for imbalanced datasets)
- **Macro-F1**: Unweighted average F1-Score across classes (treats all classes equally)

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

