// FILE: src/main/java/ma/yassine/arabicnlp/search/QueryVectorizer.java
package ma.yassine.arabicnlp.search;

import ma.yassine.arabicnlp.nlp.TextPreprocessor;

import java.util.*;

public class QueryVectorizer {

    public static double[] vectorize(String query,
                                     Map<String, Integer> vocab,
                                     int docCount) {

        List<String> tokens = TextPreprocessor.preprocess(query);
        double[] vector = new double[vocab.size()];

        Map<String, Long> tf = new HashMap<>();
        for (String t : tokens) {
            tf.put(t, tf.getOrDefault(t, 0L) + 1);
        }

        for (Map.Entry<String, Long> e : tf.entrySet()) {
            String term = e.getKey();
            if (!vocab.containsKey(term)) continue;

            int idx = vocab.get(term);
            double tfVal = e.getValue() / (double) tokens.size();
            double idfVal = 1.0; // query-side smoothing
            vector[idx] = tfVal * idfVal;
        }

        return vector;
    }
}
