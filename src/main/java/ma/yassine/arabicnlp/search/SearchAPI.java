// FILE: src/main/java/ma/yassine/arabicnlp/search/SearchAPI.java
package ma.yassine.arabicnlp.search;

import static spark.Spark.*;
import com.google.gson.Gson;
import java.nio.file.Files;
import java.nio.file.Paths;

public class SearchAPI {
    public static void main(String[] args) throws Exception {

        TFIDFLoader.load("resources/models/tfidf");
        port(4567);

        // Enable CORS for all origins
        before((req, res) -> {
            res.header("Access-Control-Allow-Origin", "*");
            res.header("Access-Control-Allow-Methods", "GET, OPTIONS");
            res.header("Access-Control-Allow-Headers", "Content-Type");
        });

        options("/*", (req, res) -> {
            res.header("Access-Control-Allow-Origin", "*");
            res.header("Access-Control-Allow-Methods", "GET, OPTIONS");
            res.header("Access-Control-Allow-Headers", "Content-Type");
            return "ok";
        });

        get("/search", (req,res)->{
            res.type("application/json; charset=UTF-8");
            String q=req.queryParams("q");
            if(q==null||q.isBlank()) return "[]";
            return new Gson().toJson(SearchEngine.search(q,10));
        });

        get("/file", (req,res)->{
            res.type("text/plain; charset=UTF-8");
            String filename=req.queryParams("name");
            if(filename==null||filename.isBlank()) return "";
            
            try {
                String filePath="resources/data/flattened_docs/"+filename;
                String content=new String(Files.readAllBytes(Paths.get(filePath)), "UTF-8");
                return content;
            } catch(Exception e) {
                return "Error reading file: "+e.getMessage();
            }
        });
    }
}
