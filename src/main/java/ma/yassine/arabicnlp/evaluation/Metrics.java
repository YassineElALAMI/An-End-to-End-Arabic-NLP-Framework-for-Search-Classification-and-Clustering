package ma.yassine.arabicnlp.evaluation;

import java.util.*;

public class Metrics {

    public static double precision(String label, ConfusionMatrix cm) {
        int tp = cm.get(label, label);
        int fp = 0;
        for (String l : cm.labels()) {
            if (!l.equals(label)) fp += cm.get(l, label);
        }
        return tp + fp == 0 ? 0 : (double) tp / (tp + fp);
    }

    public static double recall(String label, ConfusionMatrix cm) {
        int tp = cm.get(label, label);
        int fn = 0;
        for (String l : cm.labels()) {
            if (!l.equals(label)) fn += cm.get(label, l);
        }
        return tp + fn == 0 ? 0 : (double) tp / (tp + fn);
    }

    public static double f1(String label, ConfusionMatrix cm) {
        double p = precision(label, cm);
        double r = recall(label, cm);
        return p + r == 0 ? 0 : 2 * p * r / (p + r);
    }

    public static double macroF1(ConfusionMatrix cm) {
        double sum = 0;
        for (String label : cm.labels()) {
            sum += f1(label, cm);
        }
        return sum / cm.labels().size();
    }
}
