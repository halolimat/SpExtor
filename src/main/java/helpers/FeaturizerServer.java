package helpers;

import static spark.Spark.*;

public class FeaturizerServer {
    public static void main(String[] args) {

        CoreNLPFeaturizer f = new CoreNLPFeaturizer();

        post("/", "application/json", (req,res) -> {
            String sentence = req.queryParams("sentence");
            String is_raw = req.queryParams("is_raw");

            if (is_raw.toLowerCase().equals("true"))
                return  f.featurizer.extractFeatures(sentence, "raw");
            else
                return  f.featurizer.extractFeatures(sentence, "conll");
        });
    }
}