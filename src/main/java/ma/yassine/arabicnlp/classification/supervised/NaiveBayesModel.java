// FILE: src/main/java/ma/yassine/arabicnlp/classification/supervised/NaiveBayesModel.java
package ma.yassine.arabicnlp.classification.supervised;

import java.io.Serializable;
import java.util.Map;

public class NaiveBayesModel implements Serializable {

    public Map<String, Double> classPriors;
    public Map<String, Map<Integer, Double>> condProb;
    public int vocabSize;
}
