package ma.yassine.arabicnlp.evaluation;

/**
 * Metrics for model evaluation
 */
public class Metrics {
    
    /**
     * Calculate precision
     * @param truePositives number of true positives
     * @param falsePositives number of false positives
     * @return precision score
     */
    public static double precision(int truePositives, int falsePositives) {
        return (double) truePositives / (truePositives + falsePositives);
    }
    
    /**
     * Calculate recall
     * @param truePositives number of true positives
     * @param falseNegatives number of false negatives
     * @return recall score
     */
    public static double recall(int truePositives, int falseNegatives) {
        return (double) truePositives / (truePositives + falseNegatives);
    }
    
    /**
     * Calculate F1 score
     * @param precision precision score
     * @param recall recall score
     * @return F1 score
     */
    public static double f1Score(double precision, double recall) {
        return 2 * (precision * recall) / (precision + recall);
    }
}
