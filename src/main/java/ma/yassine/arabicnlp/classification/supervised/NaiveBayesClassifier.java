// FILE: src/main/java/ma/yassine/arabicnlp/classification/supervised/NaiveBayesClassifier.java
package ma.yassine.arabicnlp.classification.supervised;

import java.util.Map;

public class NaiveBayesClassifier {

    public static String predict(double[] vector, NaiveBayesModel model) {

        String bestClass = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (String c : model.classPriors.keySet()) {
            double score = model.classPriors.get(c);

            Map<Integer, Double> probs = model.condProb.get(c);
            for (int i = 0; i < vector.length; i++) {
                if (vector[i] > 0) {
                    score += probs.get(i);
                }
            }

            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }
        return bestClass;
    }
}
