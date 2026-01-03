// FILE: src/main/java/ma/yassine/arabicnlp/classification/unsupervised/KMeansMain.java
package ma.yassine.arabicnlp.classification.unsupervised;

import ma.yassine.arabicnlp.search.TFIDFLoader;
import ma.yassine.arabicnlp.classification.supervised.LabelExtractor;

import java.util.*;
import java.io.*;
import java.nio.file.*;
import java.nio.file.StandardCopyOption;

public class KMeansMain {

    // Define cluster names for the 5 clusters
    private static final String[] CLUSTER_NAMES = {
            "مجتمع",      // Society
            "سياسة",      // Politics
            "اقتصاد",      // Economics
            "رياضة",      // Sports
            "فنون"        // Arts
    };

    public static void main(String[] args) throws Exception {

        TFIDFLoader.load("resources/models/tfidf");

        System.out.println("Starting K-means clustering on TF-IDF matrix...");
        System.out.println("Documents: " + TFIDFLoader.documents.size());
        System.out.println("Vocabulary size: " + TFIDFLoader.vocabulary.size());

        // Perform K-means clustering with k=5
        int K = 5;
        KMeans km = new KMeans(TFIDFLoader.tfidfMatrix, K);
        km.fit();

        int[] clusters = km.getLabels();

        // Create output directory for clusters
        String outputDir = "resources/data/KMclusters";
        Files.createDirectories(Paths.get(outputDir));

        // Map to store cluster -> list of documents
        Map<Integer, List<String>> clusterMap = new HashMap<>();
        for (int i = 0; i < K; i++) {
            clusterMap.put(i, new ArrayList<>());
        }

        // Assign documents to clusters
        for (int i = 0; i < clusters.length; i++) {
            String doc = TFIDFLoader.documents.get(i);
            int clusterId = clusters[i];
            clusterMap.get(clusterId).add(doc);
        }

        // Create cluster folders and save files
        for (int clusterId = 0; clusterId < K; clusterId++) {
            String clusterName = CLUSTER_NAMES[clusterId];
            String clusterPath = outputDir + "/" + sanitizeFileName(clusterName);
            Files.createDirectories(Paths.get(clusterPath));

            List<String> docsInCluster = clusterMap.get(clusterId);
            System.out.println("\nCluster " + clusterId + " (" + clusterName + ") with " + docsInCluster.size() + " documents");

            for (String docFileName : docsInCluster) {
                try {
                    // Try to find the file in flattened_docs directory
                    Path sourcePath = Paths.get("resources/data/flattened_docs/" + docFileName);
                    if (Files.exists(sourcePath)) {
                        Path destPath = Paths.get(clusterPath, docFileName);
                        Files.copy(sourcePath, destPath, StandardCopyOption.REPLACE_EXISTING);
                        System.out.println("  " + docFileName);
                    } else {
                        System.err.println("Source file not found: " + sourcePath);
                    }
                } catch (Exception e) {
                    System.err.println("Error copying " + docFileName + ": " + e.getMessage());
                }
            }
        }

        System.out.println("\n✓ K-means clustering complete! Clusters saved to: " + outputDir);
    }

    /**
     * Sanitizes cluster name to be a valid folder name
     */
    private static String sanitizeFileName(String name) {
        return name.replaceAll("[<>:\"/\\|?*]", "_")
                   .replaceAll("\\s+", "_")
                   .replaceAll("_+", "_");
    }
}
