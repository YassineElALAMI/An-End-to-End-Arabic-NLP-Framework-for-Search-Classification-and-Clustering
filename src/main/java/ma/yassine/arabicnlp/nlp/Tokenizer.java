package ma.yassine.arabicnlp.nlp;

import java.util.ArrayList;
import java.util.List;
public class Tokenizer {

    public static List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();

        // Normalize Arabic text
        text = normalizeArabic(text);

        // Split by whitespace
        String[] words = text.split("\\s+");

        for (String word : words) {
            if (word.length() > 1) {
                tokens.add(word);
            }
        }
        return tokens;
    }

    private static String normalizeArabic(String text) {
        return text
                .replaceAll("[إأآا]", "ا")
                .replaceAll("ى", "ي")
                .replaceAll("ؤ", "و")
                .replaceAll("ئ", "ي")
                .replaceAll("ة", "ه")
                .replaceAll("[^؀-ۿ\\s]", "") // remove non-Arabic chars
                .toLowerCase();
    }
}

