// FILE: src/main/java/ma/yassine/arabicnlp/classification/supervised/LabelExtractor.java
package ma.yassine.arabicnlp.classification.supervised;

public class LabelExtractor {

    public static String extractLabel(String filename) {
        // f1_agadir24_الاقتصاد.txt → الاقتصاد
        String[] parts = filename.replace(".txt", "").split("_");
        return parts[parts.length - 1];
    }
}
