// FILE: src/main/java/ma/yassine/arabicnlp/classification/supervised/NBMain.java
package ma.yassine.arabicnlp.classification.supervised;

import ma.yassine.arabicnlp.search.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class NBMain {

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
        KMeansClusterer.KMeansResult result = KMeansClusterer.cluster(
                TFIDFLoader.tfidfMatrix, 
                5
        );

        // Create output directory for clusters
        String outputDir = "resources/data/classified_clusters";
        Files.createDirectories(Paths.get(outputDir));

        // Map to store cluster -> list of documents
        Map<Integer, List<String>> clusters = new HashMap<>();
        for (int i = 0; i < 5; i++) {
            clusters.put(i, new ArrayList<>());
        }

        // Assign documents to clusters
        for (int i = 0; i < result.assignments.length; i++) {
            String doc = TFIDFLoader.documents.get(i);
            int clusterId = result.assignments[i];
            clusters.get(clusterId).add(doc);
            System.out.println(doc + " -> Cluster " + clusterId);
        }

        // Create cluster folders and copy files
        for (int clusterId = 0; clusterId < 5; clusterId++) {
            String clusterName = CLUSTER_NAMES[clusterId];
            String clusterPath = outputDir + "/" + sanitizeFileName(clusterName);
            Files.createDirectories(Paths.get(clusterPath));

            List<String> docsInCluster = clusters.get(clusterId);
            System.out.println("\nProcessing cluster " + clusterId + " (" + clusterName + ") with " + docsInCluster.size() + " documents");

            for (String docFileName : docsInCluster) {
                try {
                    // Try to find the file in flattened_docs directory
                    Path sourcePath = Paths.get("resources/data/flattened_docs/" + docFileName);
                    if (Files.exists(sourcePath)) {
                        Path destPath = Paths.get(clusterPath, docFileName);
                        Files.copy(sourcePath, destPath, StandardCopyOption.REPLACE_EXISTING);
                        System.out.println("Copied: " + docFileName + " -> " + clusterName);
                    } else {
                        System.err.println("Source file not found: " + sourcePath);
                    }
                } catch (Exception e) {
                    System.err.println("Error copying " + docFileName + ": " + e.getMessage());
                }
            }
        }

        System.out.println("\nClustering complete! Files organized in: " + outputDir);
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
