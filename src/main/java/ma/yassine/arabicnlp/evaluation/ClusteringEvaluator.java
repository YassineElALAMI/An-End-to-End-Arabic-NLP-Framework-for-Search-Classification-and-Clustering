package ma.yassine.arabicnlp.evaluation;

import java.util.*;

public class ClusteringEvaluator {

    public static ConfusionMatrix evaluate(
            List<String> trueLabels,
            int[] clusterAssignments
    ) {
        ConfusionMatrix cm = new ConfusionMatrix();

        // Map cluster → majority true label
        Map<Integer, Map<String, Integer>> clusterLabelCount = new HashMap<>();

        for (int i = 0; i < clusterAssignments.length; i++) {
            clusterLabelCount
                .computeIfAbsent(clusterAssignments[i], k -> new HashMap<>())
                .merge(trueLabels.get(i), 1, Integer::sum);
        }

        Map<Integer, String> clusterToLabel = new HashMap<>();
        for (var e : clusterLabelCount.entrySet()) {
            clusterToLabel.put(
                e.getKey(),
                e.getValue().entrySet()
                    .stream()
                    .max(Map.Entry.comparingByValue())
                    .get()
                    .getKey()
            );
        }

        // Build confusion matrix
        for (int i = 0; i < clusterAssignments.length; i++) {
            String actual = trueLabels.get(i);
            String predicted = clusterToLabel.get(clusterAssignments[i]);
            cm.add(actual, predicted);
        }

        return cm;
    }
}
