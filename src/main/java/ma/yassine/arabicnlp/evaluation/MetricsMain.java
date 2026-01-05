package ma.yassine.arabicnlp.evaluation;

import ma.yassine.arabicnlp.search.TFIDFLoader;
import ma.yassine.arabicnlp.classification.supervised.*;
import ma.yassine.arabicnlp.classification.unsupervised.*;

import java.util.*;

public class MetricsMain {

    public static void main(String[] args) throws Exception {

        TFIDFLoader.load("resources/models/tfidf");

        // TRUE LABELS
        List<String> trueLabels = new ArrayList<>();
        for (String doc : TFIDFLoader.documents) {
            trueLabels.add(LabelExtractor.extractLabel(doc));
        }

        /* ===== NAIVE BAYES ===== */
        NaiveBayesModel model = NaiveBayesTrainer.train();
        ConfusionMatrix nbCM = new ConfusionMatrix();

        for (int i = 0; i < TFIDFLoader.tfidfMatrix.length; i++) {
            String pred = NaiveBayesClassifier.predict(
                    TFIDFLoader.tfidfMatrix[i], model);
            nbCM.add(trueLabels.get(i), pred);
        }

        System.out.println("NAIVE BAYES MACRO-F1 = " +
                Metrics.macroF1(nbCM));

        /* ===== K-MEANS ===== */
        Set<String> uniq = new HashSet<>(trueLabels);
        KMeans km = new KMeans(TFIDFLoader.tfidfMatrix, uniq.size());
        km.fit();

        ConfusionMatrix kmCM =
            ClusteringEvaluator.evaluate(trueLabels, km.getLabels());

        System.out.println("K-MEANS MACRO-F1 = " +
                Metrics.macroF1(kmCM));

        /* ===== HIERARCHICAL ===== */
        HierarchicalClustering hc =
            new HierarchicalClustering(TFIDFLoader.tfidfMatrix, uniq.size());
        hc.fit();

        int[] hLabels = new int[trueLabels.size()];
        int c = 0;
        for (Set<Integer> cluster : hc.getClusters()) {
            for (int id : cluster) hLabels[id] = c;
            c++;
        }

        ConfusionMatrix hcCM =
            ClusteringEvaluator.evaluate(trueLabels, hLabels);

        System.out.println("HIERARCHICAL MACRO-F1 = " +
                Metrics.macroF1(hcCM));
    }
}
