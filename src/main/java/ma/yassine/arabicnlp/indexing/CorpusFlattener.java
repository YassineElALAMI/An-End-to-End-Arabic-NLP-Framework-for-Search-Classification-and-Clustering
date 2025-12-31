package ma.yassine.arabicnlp.indexing;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.stream.Stream;

public class CorpusFlattener {

    // CHANGE ONLY IF NEEDED
    private static final String CORPUS_ROOT = "resources/corpus";
    private static final String OUTPUT_DIR = "resources/data/flattened_docs";

    public static void main(String[] args) {
        try {
            flattenCorpus();
            System.out.println("✅ Corpus flattening finished successfully.");
        } catch (IOException e) {
            System.err.println("❌ Error while flattening corpus");
            e.printStackTrace();
        }
    }

    public static void flattenCorpus() throws IOException {

        Path corpusPath = Paths.get(CORPUS_ROOT);
        Path outputPath = Paths.get(OUTPUT_DIR);

        // Create output directory if it doesn't exist
        if (!Files.exists(outputPath)) {
            Files.createDirectories(outputPath);
        }

        // Walk through corpus recursively
        try (Stream<Path> paths = Files.walk(corpusPath)) {
            paths
                .filter(Files::isRegularFile)
                .filter(p -> p.toString().endsWith(".txt"))
                .forEach(file -> {
                    try {
                        processFile(file, corpusPath, outputPath);
                    } catch (IOException e) {
                        System.err.println("⚠️ Failed processing: " + file);
                    }
                });
        }
    }

    private static void processFile(Path file,
                                    Path corpusRoot,
                                    Path outputDir) throws IOException {

        // Get relative path from corpus root
        Path relativePath = corpusRoot.relativize(file);

        /*
         Example:
         agadir24/الاقتصاد/f1.txt
        */

        String originalFileName = file.getFileName().toString();

        // Build new filename: f1_agadir24_الاقتصاد.txt
        StringBuilder newFileName = new StringBuilder(
                originalFileName.replace(".txt", "")
        );

        for (Path part : relativePath) {
            if (!part.toString().equals(originalFileName)) {
                newFileName.append("_").append(part.toString());
            }
        }

        newFileName.append(".txt");

        // Read content
        String content = Files.readString(file, StandardCharsets.UTF_8);

        // Write flattened file
        Path outputFile = outputDir.resolve(newFileName.toString());
        Files.writeString(outputFile, content, StandardCharsets.UTF_8);

        System.out.println("✔ " + newFileName);
    }
}

