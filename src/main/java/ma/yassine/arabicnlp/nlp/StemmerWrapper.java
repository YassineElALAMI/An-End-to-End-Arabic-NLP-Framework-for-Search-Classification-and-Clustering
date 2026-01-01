package ma.yassine.arabicnlp.nlp;

import safar.basic.morphology.stemmer.interfaces.IStemmer;
import safar.basic.morphology.stemmer.impl.Light10Stemmer;
import safar.basic.morphology.stemmer.model.StemmerAnalysis;

import java.util.List;

public class StemmerWrapper {

    private static final IStemmer stemmer;

    static {
        try {
            stemmer = new Light10Stemmer();
            System.out.println("✅ SAFAR Light10 stemmer initialized");
        } catch (Exception e) {
            throw new RuntimeException("❌ Failed to initialize SAFAR stemmer", e);
        }
    }

    public static String stem(String word) {
        try {
            List<?> results = stemmer.stem(word);

            if (results != null && !results.isEmpty()) {
                Object r = results.get(0);

                if (r instanceof StemmerAnalysis) {
                    StemmerAnalysis analysis = (StemmerAnalysis) r;
                    String morpheme = analysis.getMorpheme();
                    if (morpheme != null && !morpheme.isEmpty()) {
                        return morpheme;
                    }
                }

                // fallback: parse string representation
                String s = r.toString();
                if (s.contains("morpheme = ")) {
                    int start = s.indexOf("morpheme = ") + 11;
                    int end = s.indexOf("}", start);
                    if (end > start) {
                        return s.substring(start, end).trim();
                    }
                }
            }
        } catch (Exception e) {
            // silent fallback
        }
        return word;
    }
}
