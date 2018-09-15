package helpers;

public class Test_CoreNLPFeaturizer {

    public static void main(String[] args){

        String doc =    "what O\n" +
                        "flights O\n" +
                        "leave O\n" +
                        "Atlanta B-LOC\n" +
                        "at O\n" +
                        "About O\n" +
                        "Digit O\n" +
                        "in O\n" +
                        "the O\n" +
                        "Afternoon O\n" +
                        "and O\n" +
                        "arrive O\n" +
                        "in O\n" +
                        "San B-LOC\n" +
                        "Francisco I-LOC\n" +
                        ". O\n" +
                        "\n" +
                        "what O\n" +
                        "is O\n" +
                        "the O\n" +
                        "abbreviation O\n" +
                        "for O\n" +
                        "Canadian O\n" +
                        "Airlines O\n" +
                        "International O\n" +
                        ". O";

        CoreNLPFeaturizer cf = new CoreNLPFeaturizer();

        String features_tsv = cf.featurizer.extractFeatures(doc);

        System.out.println(features_tsv);
    }
}
