package helpers;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.naturalli.NaturalLogicAnnotations;

import java.util.*;
import java.util.stream.Collectors;

public class NegationDetection {

    private Properties props;
    private StanfordCoreNLP pipeline;

    public NegationDetection(){
        // set up pipeline properties
        props = new Properties();
        // set the list of annotators to run
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,natlog");
        // build pipeline
        pipeline = new StanfordCoreNLP(props);
    }

    public String get_negations(String sentence, boolean is_raw){

        if (!is_raw){
            ArrayList<String> str = new ArrayList<>();
            for (String line : sentence.split("\n")){
                str.add(line.split(" ")[0]);
            }
            sentence=str.stream().collect(Collectors.joining(" "));
        }

        // create a document object
        CoreDocument document = new CoreDocument(sentence);
        // annnotate the document
        pipeline.annotate(document);

        List<String> result = new ArrayList<>();

        for (CoreLabel token : document.tokens()) {
            String res = String.format("%s\t%s", token.word(),
                            token.get(NaturalLogicAnnotations.PolarityAnnotation.class));
            result.add(res);
        }
        return result.stream().collect(Collectors.joining("\n"));
    }
}