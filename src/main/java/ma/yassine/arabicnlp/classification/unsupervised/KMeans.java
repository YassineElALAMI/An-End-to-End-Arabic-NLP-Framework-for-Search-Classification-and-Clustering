// FILE: src/main/java/ma/yassine/arabicnlp/classification/unsupervised/KMeans.java
package ma.yassine.arabicnlp.classification.unsupervised;

import java.util.*;

public class KMeans {

    private final int K;
    private final int maxIterations = 100;
    private final double[][] data;
    private double[][] centroids;
    private int[] labels;

    public KMeans(double[][] data, int K) {
        this.data = data;
        this.K = K;
        this.labels = new int[data.length];
        this.centroids = new double[K][data[0].length];
        initCentroids();
    }

    private void initCentroids() {
        Random rand = new Random(42);
        Set<Integer> chosen = new HashSet<>();
        for (int i = 0; i < K; i++) {
            int idx;
            do {
                idx = rand.nextInt(data.length);
            } while (chosen.contains(idx));
            chosen.add(idx);
            centroids[i] = Arrays.copyOf(data[idx], data[idx].length);
        }
    }

    public void fit() {
        for (int iter = 0; iter < maxIterations; iter++) {
            boolean changed = assignClusters();
            recomputeCentroids();
            if (!changed) break;
        }
    }

    private boolean assignClusters() {
        boolean changed = false;
        for (int i = 0; i < data.length; i++) {
            int bestCluster = -1;
            double bestDist = Double.MAX_VALUE;

            for (int c = 0; c < K; c++) {
                double dist = euclidean(data[i], centroids[c]);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestCluster = c;
                }
            }

            if (labels[i] != bestCluster) {
                labels[i] = bestCluster;
                changed = true;
            }
        }
        return changed;
    }

    private void recomputeCentroids() {
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
            if (counts[c] == 0) continue;
            for (int j = 0; j < newCentroids[c].length; j++) {
                newCentroids[c][j] /= counts[c];
            }
        }
        centroids = newCentroids;
    }

    private double euclidean(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    public int[] getLabels() {
        return labels;
    }
}
