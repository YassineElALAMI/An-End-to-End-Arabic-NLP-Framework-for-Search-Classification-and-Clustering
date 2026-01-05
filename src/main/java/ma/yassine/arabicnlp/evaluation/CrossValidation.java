package ma.yassine.arabicnlp.evaluation;

import ma.yassine.arabicnlp.classification.supervised.*;
import ma.yassine.arabicnlp.search.TFIDFLoader;

import java.util.*;

public class CrossValidation {

    public static void runKFold(int k) throws Exception {

        TFIDFLoader.load("resources/models/tfidf");

        int N = TFIDFLoader.tfidfMatrix.length;
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < N; i++) indices.add(i);

        Collections.shuffle(indices, new Random(42));

        int foldSize = N / k;
        double totalF1 = 0.0;

        for (int fold = 0; fold < k; fold++) {

            int start = fold * foldSize;
            int end = (fold == k - 1) ? N : start + foldSize;

            Set<Integer> testIdx = new HashSet<>(indices.subList(start, end));

            // ===== TRAIN NAIVE BAYES =====
            NaiveBayesModel model = NaiveBayesTrainer.trainWithSubset(
                    testIdx, false
            );

            ConfusionMatrix cm = new ConfusionMatrix();

            for (int i : testIdx) {
                String trueLabel =
                        LabelExtractor.extractLabel(TFIDFLoader.documents.get(i));

                String pred = NaiveBayesClassifier.predict(
                        TFIDFLoader.tfidfMatrix[i], model
                );

                cm.add(trueLabel, pred);
            }

            double f1 = Metrics.macroF1(cm);
            totalF1 += f1;

            System.out.println("Fold " + (fold + 1) +
                    " Macro-F1 = " + f1);
        }

        System.out.println("\nAVERAGE MACRO-F1 (" + k + "-fold) = "
                + (totalF1 / k));
    }
}
