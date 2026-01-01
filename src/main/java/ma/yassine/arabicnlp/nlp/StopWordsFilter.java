package ma.yassine.arabicnlp.nlp;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class StopWordsFilter {

    private static final Set<String> stopWords = new HashSet<>();

    static {
        try {
            stopWords.addAll(
                    Files.readAllLines(
                            Paths.get("resources/stopwords/arabic_stopwords.txt")
                    )
            );
        } catch (IOException e) {
            System.err.println("❌ Could not load stopwords file");
        }
    }

    public static List<String> removeStopWords(List<String> tokens) {
        tokens.removeIf(stopWords::contains);
        return tokens;
    }
}

