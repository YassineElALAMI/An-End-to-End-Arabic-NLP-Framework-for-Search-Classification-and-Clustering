package ma.yassine.arabicnlp.nlp;

import java.util.ArrayList;
import java.util.List;

public class TextPreprocessor {

    public static List<String> preprocess(String text) {

        // 1. Tokenize
        List<String> tokens = Tokenizer.tokenize(text);

        // 2. Remove stopwords
        tokens = StopWordsFilter.removeStopWords(tokens);

        // 3. Apply stemming
        List<String> stemmedTokens = new ArrayList<>();
        for (String token : tokens) {
            stemmedTokens.add(StemmerWrapper.stem(token));
        }

        return stemmedTokens;
    }
}