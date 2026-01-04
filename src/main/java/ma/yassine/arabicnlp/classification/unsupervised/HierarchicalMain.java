// FILE: src/main/java/ma/yassine/arabicnlp/classification/unsupervised/HierarchicalMain.java
package ma.yassine.arabicnlp.classification.unsupervised;

import ma.yassine.arabicnlp.search.TFIDFLoader;
import ma.yassine.arabicnlp.classification.supervised.LabelExtractor;

import java.io.*;
import java.nio.file.*;

import java.util.*;

public class HierarchicalMain {

    private static final String[] CLUSTER_NAMES = {
            "مجتمع",      // Society
            "سياسة",      // Politics
            "اقتصاد",      // Economics
            "رياضة",      // Sports
            "فنون"        // Arts
    };

    public static void main(String[] args) throws Exception {

        TFIDFLoader.load("resources/models/tfidf");

        // Create 5 fixed clusters
        int K = 5;

        HierarchicalClustering hc =
                new HierarchicalClustering(TFIDFLoader.tfidfMatrix, K);
        hc.fit();

        List<Set<Integer>> clusters = hc.getClusters();

        // Create output directory structure
        String outputDir = "resources/data/HierarchicalClusters";
        Files.createDirectories(Paths.get(outputDir));

        // Save clusters to files
        saveClustersToFiles(clusters, outputDir);

        System.out.println("Clusters saved to: " + Paths.get(outputDir).toAbsolutePath());
    }

    private static void saveClustersToFiles(List<Set<Integer>> clusters, String outputDir) throws IOException {
        for (int i = 0; i < clusters.size() && i < CLUSTER_NAMES.length; i++) {
            Set<Integer> cluster = clusters.get(i);
            String clusterName = CLUSTER_NAMES[i];
            String clusterPath = outputDir + "/" + sanitizeFileName(clusterName);
            Files.createDirectories(Paths.get(clusterPath));

            System.out.println("\nCluster " + i + " (" + clusterName + ") with " + cluster.size() + " documents");

            for (int docId : cluster) {
                String docFileName = TFIDFLoader.documents.get(docId);
                try {
                    // Copy file from flattened_docs directory
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
    }

    private static String sanitizeFileName(String fileName) {
        // Remove special characters that are invalid in file paths
        return fileName.replaceAll("[\\\\/:*?\"<>|]", "_");
    }
}
