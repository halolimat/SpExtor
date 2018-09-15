package spExtor;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.TypedDependency;
import edu.stanford.nlp.util.ArraySet;
import edu.stanford.nlp.util.CoreMap;
import net.sf.extjwnl.dictionary.Dictionary;
import net.sf.extjwnl.data.POS;
import net.sf.extjwnl.data.Synset;
import org.apache.commons.lang.WordUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import com.google.common.util.concurrent.AtomicLongMap;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.io.StringReader;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class FeatureFactory {

    // #################################################################################################################

    public HashMap<String, String> id_nounphrase_dict = new HashMap<>();
    public HashMap<String, String> nounphrase_id_dict = new HashMap<>();
    public HashMap<Integer, AtomicLongMap> sentence_nounphrases_map = new HashMap<>();
    public HashMap<String, Set<Integer>> nounphrase_sentences_map = new HashMap<>();

    public HashMap<Integer, Set<String>> sentence_features_map = new HashMap<>();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    public static Set<String> feature_set = new ArraySet<>();

    // (1) Lexical Features (LF)
    public HashMap<String, String> id_LF_dict = new HashMap();
    public HashMap<String, String> LF_id_dict = new HashMap();
    public HashMap<String, AtomicLongMap> nounphrase_LF_map = new HashMap();
    public HashMap<String, AtomicLongMap> LF_nounphrase_map = new HashMap();

    // (2) Lexico-Syntactic Features (LS)
    public HashMap<String, String> id_LS_dict = new HashMap();
    public HashMap<String, String> LS_id_dict = new HashMap();
    public HashMap<String, AtomicLongMap> nounphrase_LS_map = new HashMap();
    public HashMap<String, AtomicLongMap> LS_nounphrase_map = new HashMap();

    // (3) Syntactic Features (SF)
    public HashMap<String, String> id_SF_dict = new HashMap();
    public HashMap<String, String> SF_id_dict = new HashMap();
    public HashMap<String, AtomicLongMap> nounphrase_SF_map = new HashMap();
    public HashMap<String, AtomicLongMap> SF_nounphrase_map = new HashMap();

    // (4) Semantic Features (SeF)
    public HashMap<String, String> id_SeF_dict = new HashMap();
    public HashMap<String, String> SeF_id_dict = new HashMap();
    public HashMap<String, AtomicLongMap> nounphrase_SeF_map = new HashMap();
    public HashMap<String, AtomicLongMap> SeF_nounphrase_map = new HashMap();

    // (5) Contextual Features (CF)
    public HashMap<String, HashMap<String, Double>> CF = new HashMap<>();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // will contain the edited sentences to train the word2vec
    String sentences_textblob = "";

    // #################################################################################################################

    Dictionary wordnet_dt;

    // CoreNLP init

    Properties props = new Properties();
    StanfordCoreNLP pipeline;

    String modelPath = DependencyParser.DEFAULT_MODEL;
    String taggerPath = "edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger";

    MaxentTagger tagger = new MaxentTagger(taggerPath);
    DependencyParser parser = DependencyParser.loadFromModelFile(modelPath);

    // #################################################################################################################

    public FeatureFactory(){

        // init
        feature_set.add("LF");
        feature_set.add("LS");
        feature_set.add("SF");
        feature_set.add("SeF");
        feature_set.add("CF");

        this.props.setProperty("annotators", "tokenize, ssplit, pos");
        this.pipeline = new StanfordCoreNLP(props);

        // init WordNet
        try {
            // Download wordnet and make the path points to the dict folder
            String wordnet_db_path = FeatureFactory.class.getResource("wordnet/dict").getPath();
            this.wordnet_dt = Dictionary.getFileBackedInstance(wordnet_db_path);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // #################################################################################################################

    public ArraySet<String> get_nps_using_regex(String text) {

        Annotation document = new Annotation(text);
        pipeline.annotate(document);

        ArraySet<String> nps = new ArraySet<>();

        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
        for (CoreMap sentence : sentences) {

            String tagged_tokens = "";

            for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                String word = token.get(CoreAnnotations.TextAnnotation.class);
                String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
                tagged_tokens += word + " " + pos + " ";
            }

            tagged_tokens = tagged_tokens.trim();

            String JJs = "(?:(?:[A-Z]\\w+ JJ )*)";
            String NNs = "(?:[^\\s]* (?:N[A-Z]*)\\s*)+";
            String CDs = "(?:\\w+ CD)?";

            String np_regex = JJs + NNs + CDs;

            Pattern nps_pattern = Pattern.compile(np_regex);
            Matcher m = nps_pattern.matcher(tagged_tokens);
            while (m.find()) {
                String[] s = m.group(0).trim().split(" ");

                String np = "";

                for (int index = 0; index < s.length; index += 2) {
                    np += s[index] + " ";
                }

                np = np.trim();

                if (np.length() > 1) {
                    nps.add(np);
                }
            }
        }

        return nps;
    }

    private List<TypedDependency> get_dependency_parse(String text){

        ArrayList<List<TypedDependency>> sentences_dep_parses = new ArrayList<>();

        DocumentPreprocessor tokenizer = new DocumentPreprocessor(new StringReader(text));
        for (List<HasWord> sentence : tokenizer) {
            List<TaggedWord> tagged = tagger.tagSentence(sentence);
            GrammaticalStructure gs = parser.predict(tagged);

            sentences_dep_parses.add(gs.typedDependenciesEnhancedPlusPlus());
        }

        return sentences_dep_parses.get(0);
    }

    public Set<String> get_senses(String surfaceName) {

        Set<String> _senses = new ArraySet<>();

        try {

            List<Synset> senses = wordnet_dt.lookupIndexWord(POS.NOUN, surfaceName).getSenses();

            for(Synset sense : senses){
                _senses.add(sense.getLexFileName());
            }

        } catch (Exception e) {
            // pass in case cannot found or any other error
        }

        return _senses;
    }

    private String patternFactory(TypedDependency dependency, boolean governer) {

        String dep_type = dependency.reln().getShortName().replace(" ", "_");

        String gov = dependency.gov().tag();
        String dep = dependency.dep().tag();

        // TODO: check the effect of this abstraction
        if (gov != null)
            gov = gov.substring(0,1);
        if (dep != null)
            dep = dep.substring(0,1);

        if(governer)
            return  dep_type + "+" + gov + "+" + dep;

        else
            return  dep_type + "+" + dep + "+" + gov;
    }

    private void add_dependency_patterns(List<TypedDependency> sentence_parse, String nounphrase, String nounphrase_id, int sentence_id) {

        // edit noun phrase for comparison
        nounphrase = nounphrase.replace(" ", "_");

        Iterator itr = sentence_parse.iterator();

        while (itr.hasNext()) {

            TypedDependency dependency = (TypedDependency) itr.next();

            String govWord = dependency.gov().word();
            String depWord = dependency.dep().word();

            if (govWord == null)
                govWord = "";

            if (depWord == null)
                depWord = "";


            String dep_parse;

            if (nounphrase.equals(govWord) && nounphrase_id_dict.containsKey(govWord.replace("_", " ")))
                dep_parse = patternFactory(dependency, true);

            else if (nounphrase.equals(depWord) && nounphrase_id_dict.containsKey(depWord.replace("_", " ")))
                dep_parse = patternFactory(dependency, false);

            else
                continue;

            String FID;
            if(!SF_id_dict.containsKey(dep_parse)){
                FID = "LSP-"+SF_id_dict.size();
                SF_id_dict.put(dep_parse, FID);
                id_SF_dict.put(FID, dep_parse);
            }

            else
                FID = SF_id_dict.get(dep_parse);

            if(!SF_nounphrase_map.containsKey(FID))
                SF_nounphrase_map.put(FID, AtomicLongMap.create());

            SF_nounphrase_map.get(FID).getAndIncrement(nounphrase_id);

            if(!nounphrase_SF_map.containsKey(nounphrase_id))
                nounphrase_SF_map.put(nounphrase_id, AtomicLongMap.create());

            nounphrase_SF_map.get(nounphrase_id).getAndIncrement(FID);

            sentence_features_map.get(sentence_id).add(FID);
        }
    }

    public Word2Vec trainNp2VecModel(String sentences_blob) {

        //System.out.println("Training Np2Vec");

        Word2Vec vec = new Word2Vec();

        try {
            InputStream is = new ByteArrayInputStream(sentences_blob.getBytes());

            // Strip white space before and after for each line
            SentenceIterator iter = new BasicLineIterator(is);

            // Split on white spaces in the line to get words
            TokenizerFactory t = new DefaultTokenizerFactory();
            t.setTokenPreProcessor(new CommonPreprocessor());

            InMemoryLookupCache cache = new InMemoryLookupCache();
            WeightLookupTable table = new InMemoryLookupTable.Builder()
                    .vectorLength(100)
                    .useAdaGrad(false)
                    .cache(cache)
                    .lr(0.025f).build();

            vec = new Word2Vec.Builder()
                    .minWordFrequency(1)
                    .iterations(1)
                    .batchSize(250)
                    .layerSize(100)
                    .lookupTable(table)
                    //.stopWords(new ArrayList<String>())
                    .vocabCache(cache)
                    .seed(42)
                    .learningRate(0.025)
                    .minLearningRate(0.001)
                    .sampling(0)
                    .windowSize(5)
                    .modelUtils(new BasicModelUtils<>())
                    .iterate(iter)
                    // NOTE: Custom tokenizer can be used here!
                    //.tokenizerFactory(t)
                    .build();

            vec.fit();

        } catch (Exception e) {
            e.printStackTrace();
        }

        return vec;
    }

    // #################################################################################################################

    /*

    1-  Extract noun phrases
    2-  Extract features for each noun phrase:
        2-1-    Dependency Parse
        2-2-    word2vec: top 10 or > 80 similarity
        2-3-    word senses
        2-4-    lexical and orthographic features
        2-5-    lexico-syntactic
     */

    private void extract_nounphrases(String sentence, int sentence_key){
        ArraySet<String> nps = get_nps_using_regex(sentence);

        for(String np: nps){

            String np_id = "NP-"+nounphrase_id_dict.size();

            if(!nounphrase_id_dict.keySet().contains(np)){
                id_nounphrase_dict.put(np_id, np);
                nounphrase_id_dict.put(np, np_id);
            }

            else
                np_id = nounphrase_id_dict.get(np);

            // Mapping sentence to noun phrases --------------------------------------------------------------------
            if(!sentence_nounphrases_map.keySet().contains(sentence_key)){
                sentence_nounphrases_map.put(sentence_key, AtomicLongMap.create());
            }

            sentence_nounphrases_map.get(sentence_key).getAndIncrement(np_id);

            // noun phrase to sentence mapping
            if(!nounphrase_sentences_map.keySet().contains(np_id)){
                nounphrase_sentences_map.put(np_id, new ArraySet<>());
            }

            nounphrase_sentences_map.get(np_id).add(sentence_key);
        }
    }

    private void extract_1_to_4_features(String sentence, int sentence_id){

        /*      (1) Lexical Features (LF)
                (2) Lexico-Syntactic Features (LS)
                (3) Syntactic Features (SF)
                (4) Semantic Features (SeF)
                (5) Contextual Features (CF)         */

        // if the sentence has no noun phrases then ignore it.
        if (!sentence_nounphrases_map.keySet().contains(sentence_id))
            return;

        // sentence_id to feature_id map
        sentence_features_map.put(sentence_id, new HashSet<>());

        for (Object nounphrase_id_obj : sentence_nounphrases_map.get(sentence_id).asMap().keySet()){

            String nounphrase_id = (String) nounphrase_id_obj;

            String nounphrase = id_nounphrase_dict.get(nounphrase_id);

            ////////[ (1) Lexical Features (LF) ]///////////////////////////////////////////////////////////////////////

            if(!nounphrase_LF_map.containsKey(nounphrase_id))
                nounphrase_LF_map.put(nounphrase_id, AtomicLongMap.create());

            // Orthographic Form (OF) ----------------------------------------------------------------------------------

            String LF;

            String numbers_regex = "(:?^|\\s)(?=.)((?:0|(?:[1-9](?:\\d*|\\d{0,2}(?:,\\d{3})*)))?(?:\\.\\d*[1-9])?)(?!\\S)";
            if(Pattern.matches(numbers_regex, nounphrase))
                LF = "numeric";

            else if(Pattern.matches("(([A-Z].*[0-9])|([0-9].*[A-Z]))", nounphrase))
                LF = "alphanumeric";

            else if(!Pattern.matches("[a-zA-Z0-9]+", nounphrase))
                LF = "other";

            else
                LF = "alpha";

            if(!LF.equals("")) {

                // Feature ID
                String FID;
                if (!LF_id_dict.containsKey(LF)) {
                    FID = "TSF-"+LF_id_dict.size();
                    LF_id_dict.put(LF, FID);
                    id_LF_dict.put(FID, LF);

                    LF_nounphrase_map.put(FID, AtomicLongMap.create());
                }

                else{
                    FID = LF_id_dict.get(LF);
                }

                nounphrase_LF_map.get(nounphrase_id).getAndIncrement(FID);
                LF_nounphrase_map.get(FID).getAndIncrement(nounphrase_id);

                sentence_features_map.get(sentence_id).add(FID);
            }

            // ------------------>

            if(nounphrase.equals(nounphrase.toUpperCase()))
                LF = "all_upper";

            else if(nounphrase.equals(nounphrase.toLowerCase()))
                LF = "all_lower";

            else if(nounphrase.equals(WordUtils.capitalizeFully(nounphrase)))
                LF = "title_case";

            else
                LF = "mixed_case";

            String FID;
            if (!LF_id_dict.containsKey(LF)) {
                FID = "CSF-"+LF_id_dict.size();
                LF_id_dict.put(LF, FID);
                id_LF_dict.put(FID, LF);

                LF_nounphrase_map.put(FID, AtomicLongMap.create());
            }

            else
                FID = LF_id_dict.get(LF);

            nounphrase_LF_map.get(nounphrase_id).getAndIncrement(FID);
            LF_nounphrase_map.get(FID).getAndIncrement(nounphrase_id);

            sentence_features_map.get(sentence_id).add(FID);

            // Word Shape (WS) -----------------------------------------------------------------------------------------

            // long word shape (LWS) >>>
            ArrayList<Character> token_pattern = new ArrayList<>();
            for (char c: nounphrase.toCharArray()){
                if(Character.isDigit(c))
                    token_pattern.add('D');

                else if (Character.isLetter(c))
                    token_pattern.add('L');

                else
                    token_pattern.add(c);

            }

            String WS = token_pattern.stream().map(c->c.toString()).collect(Collectors.joining(""));

            if (!LF_id_dict.containsKey(WS)) {
                FID = "LWS-"+LF_id_dict.size();
                LF_id_dict.put(WS, FID);
                id_LF_dict.put(FID, WS);

                LF_nounphrase_map.put(FID, AtomicLongMap.create());
            }

            else
                FID = LF_id_dict.get(WS);

            nounphrase_LF_map.get(nounphrase_id).getAndIncrement(FID);
            LF_nounphrase_map.get(FID).getAndIncrement(nounphrase_id);

            sentence_features_map.get(sentence_id).add(FID);

            //----------------------------------------------------------
            // long word shape (SWS) >>>

            // add also a compact version of the pattern
            WS = WS.replaceAll("([a-zA-Z])\\1{2,}", "$1");

            // Short Surface Form
            if (!LF_id_dict.containsKey(WS)) {
                FID = "SWS-"+LF_id_dict.size();
                LF_id_dict.put(WS, FID);
                id_LF_dict.put(FID, WS);

                LF_nounphrase_map.put(FID, AtomicLongMap.create());
            }

            else
                FID = LF_id_dict.get(WS);

            nounphrase_LF_map.get(nounphrase_id).getAndIncrement(FID);
            LF_nounphrase_map.get(FID).getAndIncrement(nounphrase_id);

            sentence_features_map.get(sentence_id).add(FID);

            ////////[ (2) Lexico-Syntactic Features (LS) ]//////////////////////////////////////////////////////////////

            // NOTE: the tokenizer here is white space, since conll is already tokenized, use other function later!
            List<String> tokens = Arrays.asList(sentence.split(" "));
            List<String> np_tokens = Arrays.asList(nounphrase.split(" "));

            int np_start_idx = tokens.indexOf(np_tokens.get(0));
            int np_end_idx = tokens.indexOf(np_tokens.get(np_tokens.size()-1));

            String beg;
            if(np_start_idx > 0)
                beg = tokens.get(np_start_idx-1);
            else
                beg = "_";

            String end;
            if(np_end_idx < tokens.size()-1)
                end = tokens.get(np_end_idx+1);
            else
                end = "_";

            String LS = beg+"#NP#"+end;

            if(!LS_id_dict.containsKey(LS)){
                FID = "LSF-"+LS_id_dict.size();
                LS_id_dict.put(LS, FID);
                id_LS_dict.put(FID, LS);
            }

            else
                FID = LS_id_dict.get(LS);


            if(!LS_nounphrase_map.containsKey(FID))
                LS_nounphrase_map.put(FID, AtomicLongMap.create());

            LS_nounphrase_map.get(FID).getAndIncrement(nounphrase_id);

            if(!nounphrase_LS_map.containsKey(nounphrase_id))
                nounphrase_LS_map.put(nounphrase_id, AtomicLongMap.create());

            nounphrase_LS_map.get(nounphrase_id).getAndIncrement(FID);

            sentence_features_map.get(sentence_id).add(FID);

            ////////[ (4) Semantic Features (SeF) ]/////////////////////////////////////////////////////////////////////

            Set<String> senses = get_senses(nounphrase);

            for(String sense: senses){

                if(!SeF_id_dict.containsKey(sense)){
                    FID = "WSF-"+SeF_id_dict.size();
                    SeF_id_dict.put(sense, FID);
                    id_SeF_dict.put(FID, sense);
                }

                else
                    FID = SeF_id_dict.get(sense);

                if (!nounphrase_SeF_map.containsKey(nounphrase_id))
                    nounphrase_SeF_map.put(nounphrase_id, AtomicLongMap.create());
                nounphrase_SeF_map.get(nounphrase_id).getAndIncrement(FID);

                if (!SeF_nounphrase_map.containsKey(FID))
                    SeF_nounphrase_map.put(FID, AtomicLongMap.create());
                SeF_nounphrase_map.get(FID).getAndIncrement(nounphrase_id);

                sentence_features_map.get(sentence_id).add(FID);
            }
        }

        ////////[ (3) Syntactic Features (SF) ]/////////////////////////////////////////////////////////////////////////

        // Dependency Parsing

        for (Object nounphrase_id_obj : sentence_nounphrases_map.get(sentence_id).asMap().keySet()) {
            String nounphrase_id = (String) nounphrase_id_obj;
            String nounphrase = id_nounphrase_dict.get(nounphrase_id);

            sentence = sentence.replace(nounphrase, nounphrase.replace(" ","_"));
        }

        // TODO: add dot at the end of the sentence if there is none.

        List<TypedDependency> sentence_parse = get_dependency_parse(sentence);

        for (Object nounphrase_id_obj : sentence_nounphrases_map.get(sentence_id).asMap().keySet()) {

            String nounphrase_id = (String) nounphrase_id_obj;
            String nounphrase = id_nounphrase_dict.get(nounphrase_id);

            add_dependency_patterns(sentence_parse, nounphrase, nounphrase_id, sentence_id);
        }

        // to be used to train the word2vec later
        sentences_textblob += sentence + "\n";
    }

    public void featurize(HashMap<Integer, String> raw_sentences){

        Set<Integer> raw_sentences_ids = raw_sentences.keySet();

        int counter = 0;
        for(int sentence_id: raw_sentences_ids){

            String sentence = raw_sentences.get(sentence_id);

            // 1
            extract_nounphrases(sentence, sentence_id);

            // 2
            extract_1_to_4_features(sentence, sentence_id);

            counter+=1;

            //System.out.println(raw_sentences_ids.size() - counter);
        }

        ////////[ (5) Contextual Features (CF) ]////////////////////////////////////////////////////////////////////////

        // 3 Train word2vec
        Word2Vec nounphrase2vec = trainNp2VecModel(sentences_textblob);

        //word2vec_features
        for(String nounphrase: nounphrase_id_dict.keySet()){

            String nounphrase_id = nounphrase_id_dict.get(nounphrase);

            // init position
            CF.put(nounphrase_id, new HashMap<>());

            nounphrase = nounphrase.replace(" ", "_");

            Collection<String> nearest_words = nounphrase2vec.wordsNearest(nounphrase, 100);

            int top_x = 20;
            for(String word: nearest_words){

                String word_id = nounphrase_id_dict.get(word.replace("_", " "));

                // only noun phrases
                if(nounphrase_id_dict.keySet().contains(word)) {

                    double similarity = nounphrase2vec.similarity(nounphrase, word);

                    CF.get(nounphrase_id).put(word_id, similarity);

                    top_x -= 1;
                    if(top_x <= 0){
                        break;
                    }
                }
            }
        }
    }

    // #################################################################################################################
}