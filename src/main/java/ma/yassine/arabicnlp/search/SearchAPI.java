package ma.yassine.arabicnlp.search;

/**
 * API for search operations
 */
public class SearchAPI {
    private SearchEngine searchEngine;
    private Similarity similarity;
    
    /**
     * Initialize search API
     */
    public SearchAPI() {
        this.searchEngine = new SearchEngine();
        this.similarity = new Similarity();
    }
}
