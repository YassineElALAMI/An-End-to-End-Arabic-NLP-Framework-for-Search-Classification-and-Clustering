package ma.yassine.arabicnlp.search;

import java.io.*;
import java.util.*;

public class TFIDFLoader {

    public static Map<String, Integer> vocabulary;
    public static List<String> documents;
    public static double[][] tfidfMatrix;

    public static void load(String basePath) throws Exception {
        vocabulary = (Map<String, Integer>) deserialize(basePath + "/vocabulary.ser");
        documents = (List<String>) deserialize(basePath + "/doc_index.ser");
        tfidfMatrix = (double[][]) deserialize(basePath + "/tfidf_matrix.ser");
    }

    private static Object deserialize(String path) throws Exception {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(path))) {
            return in.readObject();
        }
    }
}
