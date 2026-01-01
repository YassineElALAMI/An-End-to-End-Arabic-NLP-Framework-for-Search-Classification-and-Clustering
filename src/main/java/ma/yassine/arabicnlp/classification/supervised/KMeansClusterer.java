package ma.yassine.arabicnlp.classification.supervised;

import java.util.*;

public class KMeansClusterer {

    private static final int MAX_ITERATIONS = 100;
    private static final double EPSILON = 1e-4;

    public static class KMeansResult {
        public int[] assignments;        // which cluster each document belongs to
        public double[][] centroids;    // cluster centroids
        public int k;                   // number of clusters

        public KMeansResult(int[] assignments, double[][] centroids, int k) {
            this.assignments = assignments;
            this.centroids = centroids;
            this.k = k;
        }
    }

    public static KMeansResult cluster(double[][] data, int k) {
        int n = data.length;        // number of documents
        int d = data[0].length;     // dimensionality (vocabulary size)

        // Initialize centroids randomly from data points
        double[][] centroids = new double[k][d];
        Random rand = new Random(42);
        for (int i = 0; i < k; i++) {
            int idx = rand.nextInt(n);
            System.arraycopy(data[idx], 0, centroids[i], 0, d);
        }

        int[] assignments = new int[n];
        int[] newAssignments = new int[n];
        boolean converged = false;
        int iteration = 0;

        while (!converged && iteration < MAX_ITERATIONS) {
            // Assign documents to nearest centroid
            for (int i = 0; i < n; i++) {
                double minDist = Double.MAX_VALUE;
                int bestCluster = 0;

                for (int j = 0; j < k; j++) {
                    double dist = cosineSimilarity(data[i], centroids[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = j;
                    }
                }
                newAssignments[i] = bestCluster;
            }

            // Check convergence
            converged = Arrays.equals(assignments, newAssignments);
            assignments = newAssignments.clone();

            // Update centroids
            for (int j = 0; j < k; j++) {
                double[] newCentroid = new double[d];
                int count = 0;

                for (int i = 0; i < n; i++) {
                    if (assignments[i] == j) {
                        for (int t = 0; t < d; t++) {
                            newCentroid[t] += data[i][t];
                        }
                        count++;
                    }
                }

                if (count > 0) {
                    for (int t = 0; t < d; t++) {
                        newCentroid[t] /= count;
                    }
                    // Normalize
                    double norm = Math.sqrt(Arrays.stream(newCentroid).map(x -> x * x).sum());
                    if (norm > 0) {
                        for (int t = 0; t < d; t++) {
                            newCentroid[t] /= norm;
                        }
                    }
                    centroids[j] = newCentroid;
                }
            }

            iteration++;
            System.out.println("K-means iteration " + iteration + " completed");
        }

        System.out.println("K-means converged in " + iteration + " iterations");
        return new KMeansResult(assignments, centroids, k);
    }

    /**
     * Cosine distance (1 - cosine similarity)
     */
    private static double cosineSimilarity(double[] a, double[] b) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        double similarity = 0;
        if (normA > 0 && normB > 0) {
            similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
        }

        return 1 - similarity;  // convert to distance
    }
}
