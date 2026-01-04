// FILE: src/main/java/ma/yassine/arabicnlp/classification/unsupervised/HierarchicalClustering.java
package ma.yassine.arabicnlp.classification.unsupervised;

import java.util.*;

public class HierarchicalClustering {

    private final double[][] data;
    private final int targetClusters;
    private List<Set<Integer>> clusters;

    public HierarchicalClustering(double[][] data, int targetClusters) {
        this.data = data;
        this.targetClusters = targetClusters;
        initClusters();
    }

    private void initClusters() {
        clusters = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            Set<Integer> c = new HashSet<>();
            c.add(i);
            clusters.add(c);
        }
    }

    public void fit() {
        while (clusters.size() > targetClusters) {
            mergeClosestClusters();
        }
    }

    private void mergeClosestClusters() {
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

        clusters.get(c1).addAll(clusters.get(c2));
        clusters.remove(c2);
    }

    // Single-link (minimum distance)
    private double clusterDistance(Set<Integer> a, Set<Integer> b) {
        double min = Double.MAX_VALUE;
        for (int i : a) {
            for (int j : b) {
                double d = euclidean(data[i], data[j]);
                if (d < min) min = d;
            }
        }
        return min;
    }

    private double euclidean(double[] x, double[] y) {
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            double d = x[i] - y[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    public List<Set<Integer>> getClusters() {
        return clusters;
    }
}
