package helpers;

import static spark.Spark.post;

public class HelperServer {
    public static void main(String[] args) {
        CoreNLPFeaturizer f = new CoreNLPFeaturizer();
        PolarityAnnoation nd = new PolarityAnnoation();
        post("/", "application/json", (req,res) -> {
            String requestType=req.queryParams("type");
            String sentence = req.queryParams("sentence");
            String is_raw = req.queryParams("is_raw");
            if (requestType.equals("featurize")){
                if (is_raw.toLowerCase().equals("true"))
                    return f.featurizer.extractFeatures(sentence, "raw");
                else
                    return  f.featurizer.extractFeatures(sentence, "conll");
            }
            else if(requestType.equals("negation"))
                return nd.get_polarityAnnotation(sentence, is_raw.toLowerCase().equals("true"));
            else
                return null;
        });
    }
}