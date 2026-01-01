// FILE: src/main/java/ma/yassine/arabicnlp/search/SearchEngine.java
package ma.yassine.arabicnlp.search;

import java.util.*;

public class SearchEngine {

    public static List<SearchResult> search(String query, int topK) {

        double[] queryVector = QueryVectorizer.vectorize(
                query,
                TFIDFLoader.vocabulary,
                TFIDFLoader.documents.size()
        );

        List<SearchResult> results = new ArrayList<>();

        for (int i = 0; i < TFIDFLoader.tfidfMatrix.length; i++) {
            double score = CosineSimilarity.compute(
                    queryVector,
                    TFIDFLoader.tfidfMatrix[i]
            );
            if (score > 0) {
                results.add(new SearchResult(
                        TFIDFLoader.documents.get(i),
                        score
                ));
            }
        }

        results.sort((a, b) -> Double.compare(b.score, a.score));

        return results.subList(0, Math.min(topK, results.size()));
    }
}
