package ma.yassine.arabicnlp;

import ma.yassine.arabicnlp.nlp.TextPreprocessor;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {

        String text = Files.readString(
                Paths.get("resources/data/flattened_docs/f1_agadir24_الاقتصاد والمال.txt")
        );

        List<String> tokens = TextPreprocessor.preprocess(text);

        tokens.stream().limit(30).forEach(System.out::println);
    }
}
