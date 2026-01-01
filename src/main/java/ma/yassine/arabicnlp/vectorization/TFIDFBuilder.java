package ma.yassine.arabicnlp.vectorization;

import ma.yassine.arabicnlp.nlp.TextPreprocessor;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.text.DecimalFormat;
import java.util.*;
import java.util.stream.Collectors;

public class TFIDFBuilder {

    private static final String INPUT_DIR = "resources/data/flattened_docs";
    private static final String OUTPUT_DIR = "resources/models/tfidf";

    public static void main(String[] args) throws Exception {
        buildAndSave();
    }

    public static void buildAndSave() throws Exception {

        /* ===================== LOAD + NLP ===================== */
        Map<String, List<String>> documents = new LinkedHashMap<>();
        Map<String, Integer> vocabulary = new LinkedHashMap<>();

        List<Path> files = Files.list(Paths.get(INPUT_DIR))
                .filter(p -> p.toString().endsWith(".txt"))
                .toList();

        for (Path file : files) {
            String content = Files.readString(file, StandardCharsets.UTF_8);

            // NLP PIPELINE: normalize → stopwords → stemming
            List<String> tokens = TextPreprocessor.preprocess(content);

            documents.put(file.getFileName().toString(), tokens);

            for (String token : tokens) {
                vocabulary.putIfAbsent(token, vocabulary.size());
            }
        }

        int D = documents.size();
        int V = vocabulary.size();

        double[][] tfidf = new double[D][V];

        /* ===================== DOCUMENT FREQUENCY ===================== */
        Map<String, Integer> df = new HashMap<>();

        for (List<String> tokens : documents.values()) {
            Set<String> uniqueTerms = new HashSet<>(tokens);
            for (String term : uniqueTerms) {
                df.put(term, df.getOrDefault(term, 0) + 1);
            }
        }

        /* ===================== TF-IDF ===================== */
        int docIndex = 0;
        for (List<String> tokens : documents.values()) {

            Map<String, Long> tf = tokens.stream()
                    .collect(Collectors.groupingBy(t -> t, Collectors.counting()));

            for (Map.Entry<String, Long> entry : tf.entrySet()) {
                String term = entry.getKey();
                int termIndex = vocabulary.get(term);

                double tfValue = entry.getValue() / (double) tokens.size();

                // ✅ CORRECT SMOOTHED IDF
                double idfValue = Math.log((D + 1.0) / (df.get(term) + 1.0)) + 1.0;

                tfidf[docIndex][termIndex] = tfValue * idfValue;
            }
            docIndex++;
        }

        Files.createDirectories(Paths.get(OUTPUT_DIR));

        /* ===================== SERIALIZATION ===================== */
        serialize(vocabulary, OUTPUT_DIR + "/vocabulary.ser");
        serialize(new ArrayList<>(documents.keySet()), OUTPUT_DIR + "/doc_index.ser");
        serialize(tfidf, OUTPUT_DIR + "/tfidf_matrix.ser");

        /* ===================== HUMAN-READABLE FILES ===================== */
        writeVocabularyCSV(vocabulary);
        writeDocIndexTXT(documents.keySet());
        writeTFIDFCSV(tfidf, vocabulary, documents.keySet());

        System.out.println("✅ TF-IDF GENERATED CORRECTLY");
    }

    /* ===================== UTILITIES ===================== */

    private static void serialize(Object obj, String path) throws IOException {
        try (ObjectOutputStream oos =
                     new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(obj);
        }
    }

    private static void writeVocabularyCSV(Map<String, Integer> vocab) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(
                Paths.get(OUTPUT_DIR, "vocabulary.csv"),
                StandardCharsets.UTF_8)) {

            w.write("term;index\n");
            for (var e : vocab.entrySet()) {
                w.write(e.getKey() + ";" + e.getValue());
                w.newLine();
            }
        }
    }

    private static void writeDocIndexTXT(Set<String> docs) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(
                Paths.get(OUTPUT_DIR, "doc_index.txt"),
                StandardCharsets.UTF_8)) {

            for (String doc : docs) {
                w.write(doc);
                w.newLine();
            }
        }
    }

    private static void writeTFIDFCSV(double[][] matrix,
                                      Map<String, Integer> vocab,
                                      Set<String> docs) throws IOException {

        List<String> terms = new ArrayList<>(vocab.keySet());
        List<String> docList = new ArrayList<>(docs);

        DecimalFormat df = new DecimalFormat("0.000000");

        try (BufferedWriter w = Files.newBufferedWriter(
                Paths.get(OUTPUT_DIR, "tfidf_matrix.csv"),
                StandardCharsets.UTF_8)) {

            // Header
            w.write("doc");
            for (String term : terms) {
                w.write(";" + term);
            }
            w.newLine();

            for (int d = 0; d < matrix.length; d++) {
                w.write(docList.get(d));
                for (int t = 0; t < matrix[d].length; t++) {
                    w.write(";" + df.format(matrix[d][t]).replace(",", "."));
                }
                w.newLine();
            }
        }
    }
}
