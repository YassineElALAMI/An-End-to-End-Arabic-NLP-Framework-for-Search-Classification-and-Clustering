// FILE: src/main/java/ma/yassine/arabicnlp/search/SearchResult.java
package ma.yassine.arabicnlp.search;

public class SearchResult {
    public String document;
    public double score;

    public SearchResult(String document, double score) {
        this.document = document;
        this.score = score;
    }
}
