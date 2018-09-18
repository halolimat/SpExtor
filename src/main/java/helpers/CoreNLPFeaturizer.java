package helpers;

import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ie.crf.CRFDatum;
import edu.stanford.nlp.ie.crf.CRFLabel;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.AmericanizeFunction;
import edu.stanford.nlp.sequences.SeqClassifierFlags;

import java.util.*;
import java.util.stream.Collectors;

class Featurizer extends CRFClassifier {

    private char delimiter = '\t';
    private static final String eol = System.lineSeparator();

    private static String ubPrefixFeatureString(String feat)
    {
        if (feat.endsWith("|C")) {
            return "U-" + feat;
        } else if (feat.endsWith("|CpC")) {
            return "B-" + feat;
        } else {
            return feat;
        }
    }

    private String getFeatureString(List<CoreLabel> document) {
        int docSize = document.size();
        if (this.flags.useReverse) {
            Collections.reverse(document);
        }

        StringBuilder sb = new StringBuilder();
        for (int j = 0; j < docSize; j++) {
            CoreLabel token = document.get(j);
            sb.append(token.get(CoreAnnotations.TextAnnotation.class));
            sb.append(delimiter);
            sb.append(token.get(CoreAnnotations.AnswerAnnotation.class));

            CRFDatum<List<String>, CRFLabel> d = this.makeDatum(document, j, this.featureFactories);

            List<List<String>> features = d.asFeatures();
            for (Collection<String> cliqueFeatures : features) {
                List<String> sortedFeatures = new ArrayList<>(cliqueFeatures);
                Collections.sort(sortedFeatures);
                for (String feat : sortedFeatures) {
                    feat = ubPrefixFeatureString(feat);
                    sb.append(delimiter);
                    sb.append(feat);
                }
            }
            sb.append(eol);
        }
        if (this.flags.useReverse) {
            Collections.reverse(document);
        }
        return sb.toString();
    }

    public Featurizer(CRFClassifier<CoreLabel> crf){
        super(crf.flags);
    }

    /**
     *  Extracts features for a given CoNLL formatted string
     *
     * @return a tsv string where each token is in its line.
     */
    public String extractFeatures (String conll_string) {

        Collection<List<CoreLabel>> docs = this.makeObjectBankFromString(conll_string, this.defaultReaderAndWriter());
        this.makeAnswerArraysAndTagIndex(docs);

        List<String> result = new ArrayList<>();

        for (List<CoreLabel> doc :docs) {
            result.add(getFeatureString(doc));
        }

        return result.stream().collect(Collectors.joining("\n"));
    }
}

public class CoreNLPFeaturizer{

    Featurizer featurizer;

    private SeqClassifierFlags get_flags() {

        SeqClassifierFlags flags = new SeqClassifierFlags();
        flags.wordFunction = new AmericanizeFunction();
        //flags.useDistSim=true;
        //flags.distSimLexicon="/u/nlp/data/pos_tags_are_useless/egw4-reut.512.clusters";
        flags.numberEquivalenceDistSim = true;
        flags.unknownWordDistSimClass = "0";
        // This is the default mapping of the CoNLL formatted strings
        flags.map = "word=0,answer=1";
        flags.saveFeatureIndexToDisk = false;
        flags.useTitle = true;
        flags.useClassFeature = true;
        flags.useWord = true;
        flags.useNGrams = true;
        flags.noMidNGrams = true;
        flags.usePrev = true;
        flags.useNext = true;
        flags.useLongSequences = true;
        flags.useSequences = true;
        flags.usePrevSequences = true;
        flags.maxLeft = 1;
        flags.useTypeSeqs = true;
        flags.useTypeSeqs2 = true;
        flags.useTypeySequences = true;
        flags.useOccurrencePatterns = true;
        flags.useLastRealWord = true;
        flags.useNextRealWord = true;
        flags.normalize = true;
        flags.wordShape = 3; //dan2useLC
        flags.useDisjunctive = true;
        flags.disjunctionWidth = 4;
        flags.type = "crf";
        flags.readerAndWriter = "edu.stanford.nlp.sequences.ColumnDocumentReaderAndWriter";
        flags.useObservedSequencesOnly = true;
        flags.sigma = 20;
        flags.useQN = true;
        flags.QNsize = 25;
        // makes it go faster
        flags.featureDiffThresh = 0.05;

        return flags;
    }

    public CoreNLPFeaturizer(){

        // This is using the default formatted CoNLL strings - word at 0 and answer at 1
        CRFClassifier<CoreLabel> crf = new CRFClassifier<>(get_flags());

        this.featurizer = new Featurizer(crf);
    }

    public static void main(String[] args){

        String test = "I O\nam O\nin O\nJordan LOC\n";

        CoreNLPFeaturizer f = new CoreNLPFeaturizer();
        System.out.println(f.featurizer.extractFeatures(test));
    }
}