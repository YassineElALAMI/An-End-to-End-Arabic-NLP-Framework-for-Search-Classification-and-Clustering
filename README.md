# Arabic NLP System

A comprehensive Natural Language Processing system for Arabic text analysis, classification, and clustering.

## Project Structure

```
ArabicNLPSystem/
├── lib/                          # External libraries
│   └── arabic-stemmer.jar
│
├── resources/                    # Application resources
│   ├── stopwords/               # Stop word lists
│   │   └── arabic_stopwords.txt
│   ├── corpus/                  # Text corpora
│   │   ├── newspapers/
│   │   ├── hespress/
│   │   ├── voice_of_morocco/
│   │   └── ...
│   ├── data/                    # Processed data
│   │   └── flattened_docs/
│   └── models/                  # Pre-trained models
│       ├── tfidf/              # TF-IDF models
│       ├── naive_bayes/        # Naive Bayes models
│       ├── kmeans/             # K-Means models
│       └── hierarchical/       # Hierarchical clustering models
│
├── src/main/java/ma/yassine/arabicnlp/
│   ├── nlp/                     # Core NLP processing
│   │   ├── Tokenizer.java
│   │   ├── StopWordsFilter.java
│   │   ├── StemmerWrapper.java
│   │   └── TextPreprocessor.java
│   │
│   ├── vectorization/           # Document vectorization
│   │   ├── TFIDFBuilder.java
│   │   ├── TFIDFMatrix.java
│   │   └── TFIDFLoader.java
│   │
│   ├── indexing/                # Document indexing
│   │   └── CorpusIndexer.java
│   │
│   ├── search/                  # Search functionality
│   │   ├── SearchEngine.java
│   │   ├── Similarity.java
│   │   └── SearchAPI.java
│   │
│   ├── classification/          # Classification models
│   │   ├── supervised/
│   │   │   ├── NaiveBayes.java
│   │   │   └── NBTrainer.java
│   │   └── unsupervised/
│   │       ├── KMeans.java
│   │       └── HierarchicalClustering.java
│   │
│   ├── evaluation/              # Model evaluation
│   │   ├── CrossValidation.java
│   │   ├── ConfusionMatrix.java
│   │   └── Metrics.java
│   │
│   ├── utils/                   # Utility classes
│   │   ├── FileUtils.java
│   │   ├── SerializationUtils.java
│   │   └── MathUtils.java
│   │
│   └── Main.java               # Application entry point
│
└── pom.xml                      # Maven configuration
```

## Features

- **Text Preprocessing**: Tokenization, stop word removal, stemming
- **Document Vectorization**: TF-IDF implementation
- **Search**: Document search and similarity computation
- **Classification**: 
  - Supervised: Naive Bayes
  - Unsupervised: K-Means, Hierarchical Clustering
- **Evaluation**: Cross-validation, confusion matrix, metrics

## Building

```bash
mvn clean install
```

## Running

```bash
mvn exec:java -Dexec.mainClass="ma.yassine.arabicnlp.Main"
```

## Dependencies

- Java 11+
- Maven 3.6+
- Arabic Stemmer Library

## License

MIT License
