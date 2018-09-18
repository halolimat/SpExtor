package helpers;

import static spark.Spark.*;

public class FeaturizerServer {
    public static void main(String[] args) {

        CoreNLPFeaturizer f = new CoreNLPFeaturizer();

        post("/", "application/json", (req,res) -> {
            String sentence = req.queryParams("sentence");

            return  f.featurizer.extractFeatures(sentence);
        });
    }
}

