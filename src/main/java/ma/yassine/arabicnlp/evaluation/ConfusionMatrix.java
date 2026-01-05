package ma.yassine.arabicnlp.evaluation;

import java.util.*;

public class ConfusionMatrix {

    private final Map<String, Map<String, Integer>> matrix = new HashMap<>();

    public void add(String actual, String predicted) {
        matrix
            .computeIfAbsent(actual, k -> new HashMap<>())
            .merge(predicted, 1, Integer::sum);
    }

    public int get(String actual, String predicted) {
        return matrix.getOrDefault(actual, Map.of())
                     .getOrDefault(predicted, 0);
    }

    public Set<String> labels() {
        Set<String> labels = new HashSet<>(matrix.keySet());
        matrix.values().forEach(m -> labels.addAll(m.keySet()));
        return labels;
    }

    public Map<String, Map<String, Integer>> getMatrix() {
        return matrix;
    }
}
