// FILE: src/main/java/ma/yassine/arabicnlp/classification/supervised/NaiveBayesTrainer.java
package ma.yassine.arabicnlp.classification.supervised;

import ma.yassine.arabicnlp.search.TFIDFLoader;

import java.io.*;
import java.util.*;

public class NaiveBayesTrainer {



    public static NaiveBayesModel train() {

        Map<String, Double> classCounts = new HashMap<>();
        Map<String, Map<Integer, Double>> termCounts = new HashMap<>();

        int D = TFIDFLoader.documents.size();
        int V = TFIDFLoader.vocabulary.size();

        for (int d = 0; d < D; d++) {
            String doc = TFIDFLoader.documents.get(d);
            if (doc == null) continue;
            String label = LabelExtractor.extractLabel(doc);

            classCounts.put(label, classCounts.getOrDefault(label, 0.0) + 1);

            termCounts.putIfAbsent(label, new HashMap<>());

            for (int t = 0; t < V; t++) {
                double val = TFIDFLoader.tfidfMatrix[d][t];
                if (val > 0) {
                    termCounts.get(label).put(
                            t,
                            termCounts.get(label).getOrDefault(t, 0.0) + val
                    );
                }
            }
        }

        Map<String, Double> priors = new HashMap<>();
        for (String c : classCounts.keySet()) {
            priors.put(c, Math.log(classCounts.get(c) / D));
        }

        Map<String, Map<Integer, Double>> condProb = new HashMap<>();

        for (String c : termCounts.keySet()) {
            Map<Integer, Double> probs = new HashMap<>();
            double sum = termCounts.get(c).values().stream().mapToDouble(x -> x).sum();

            for (int t = 0; t < V; t++) {
                double count = termCounts.get(c).getOrDefault(t, 0.0);
                probs.put(t, Math.log((count + 1) / (sum + V)));
            }
            condProb.put(c, probs);
        }

        NaiveBayesModel model = new NaiveBayesModel();
        model.classPriors = priors;
        model.condProb = condProb;
        model.vocabSize = V;

        return model;
    }
}
