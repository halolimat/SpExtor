import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;

import java.util.*;

public class ActiveLearning {

    HashMap<String, Double> all_entities_probabilities = new HashMap<>();

    ArrayList<String> temp = new ArrayList<>();

    // set up pipeline
    Properties props;
    StanfordCoreNLP pipeline;

    public ActiveLearning(){
        this.props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma");
        this.pipeline = new StanfordCoreNLP(props);
    }

    public double _get_NSE(CRFClassifier<CoreLabel> crf, String str){

        // wrap sentence in an Annotation object
        Annotation annotation = new Annotation(str);
        // tokenize sentence
        pipeline.annotate(annotation);
        // get the list of tokens
        List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);

        // get the n best sequences from your abstract sequence classifier
        Counter<List<CoreLabel>> nBestSequences = crf.classifyKBest(tokens,
                CoreAnnotations.NamedEntityTagAnnotation.class,
                3);

        // sort the k-best examples
        List<List<CoreLabel>> sortedKBest = Counters.toSortedList(nBestSequences);

        double NSE = 0;

        // TODO: sort based on a threshold with margin

        for (int i = 0 ; i < sortedKBest.size() ; i ++){
            List<CoreLabel> seq = sortedKBest.get(i);

            double logProb = nBestSequences.getCount(seq);
            double prob = Math.exp(logProb);

            NSE += logProb * prob;

            add_entities_probabilities(NSE, seq);

        }

        // = - sum (.)
        return NSE * -1;
    }

    public List<Map.Entry<Integer, Double>> _get_sentences_sorted_based_on_nse(
            CRFClassifier<CoreLabel> crf_model,
            HashMap<Integer, String> sentences){

        all_entities_probabilities = new HashMap<>();

        HashMap<Integer, Double> sentence_id_nse_map = new HashMap<>();

        for(int sentence_id: sentences.keySet()){
            sentence_id_nse_map.put(sentence_id, _get_NSE(crf_model, sentences.get(sentence_id)));
            //sentence_id_nse_map.put(sentence.getKey(), get_ENSE(crf_model, "I am in New York Airport"));
        }

        // sort based on nse then take the top 20 sentences

        //List<Map.Entry<Integer, Double>> sorted = sentence_id_nse_map.entrySet().stream().
        //        sorted((k2, k1) -> -k1.getValue().compareTo(k2.getValue())).collect(Collectors.toList());


        Set<Map.Entry<Integer, Double>> set = sentence_id_nse_map.entrySet();
        List<Map.Entry<Integer, Double>> list = new ArrayList<>(set);
        Collections.sort( list, new Comparator<Map.Entry<Integer, Double>>()
        {
            public int compare( Map.Entry<Integer, Double> o1, Map.Entry<Integer, Double> o2 )
            {
                if (o1.getValue() < o2.getValue()) return 1;
                if (o1.getValue() > o2.getValue()) return -1;
                return 0;
            }
        } );

        return list;
    }

    public ArrayList<Double> get_NSE(CRFClassifier<CoreLabel> crf, String str){

        StringJoiner sent_temp = new StringJoiner("\t");
        sent_temp.add(str);

        // wrap sentence in an Annotation object
        Annotation annotation = new Annotation(str);
        // tokenize sentence
        pipeline.annotate(annotation);
        // get the list of tokens
        List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);

        // get the n best sequences from your abstract sequence classifier
        Counter<List<CoreLabel>> nBestSequences = crf.classifyKBest(tokens,
                CoreAnnotations.NamedEntityTagAnnotation.class,
                3); // 3 here is the n in nSE

        // sort the k-best examples
        List<List<CoreLabel>> sortedKBest = Counters.toSortedList(nBestSequences);

        ArrayList<Double> SE = new ArrayList<>();

        for (int i = 0 ; i < sortedKBest.size() ; i ++){
            List<CoreLabel> seq = sortedKBest.get(i);

            double logProb = nBestSequences.getCount(seq);
            double prob = Math.exp(logProb);

            SE.add(logProb * prob);

            ArrayList<String> sequence_entities = add_entities_probabilities(logProb * prob, seq);
            sent_temp.add(String.join("--", sequence_entities));

            sent_temp.add(Double.toString(logProb * prob));

        }

        return SE;
    }

    public List<Map.Entry<Integer, ArrayList<Double>>> get_sentences_sorted_based_on_nse(
            CRFClassifier<CoreLabel> crf_model,
            HashMap<Integer, String> sentences){

        all_entities_probabilities = new HashMap<>();

        HashMap<Integer, ArrayList<Double>> sentence_id_nse_map = new HashMap<>();

        for(int sentence_id: sentences.keySet()){

            // (#1 SE, #2 SE, ..., NSE)
            ArrayList<Double> SE = get_NSE(crf_model, sentences.get(sentence_id));

            double NSE = SE.stream().mapToDouble(a->a).sum();
            SE.add(NSE*-1);

            sentence_id_nse_map.put(sentence_id, SE);
        }

        Set<Map.Entry<Integer, ArrayList<Double>>> set = sentence_id_nse_map.entrySet();
        List<Map.Entry<Integer, ArrayList<Double>>> list = new ArrayList<>(set);
        Collections.sort( list, new Comparator<Map.Entry<Integer, ArrayList<Double>>>()
        {
            public int compare( Map.Entry<Integer, ArrayList<Double>> o1, Map.Entry<Integer, ArrayList<Double>> o2 )
            {
                if (o1.getValue().get(o1.getValue().size()-1) < o2.getValue().get(o2.getValue().size()-1)) return 1;
                if (o1.getValue().get(o1.getValue().size()-1) > o2.getValue().get(o2.getValue().size()-1)) return -1;
                return 0;
            }
        } );

        return list;
    }

    //##################################################################################################################
    //##################################################################################################################
    //##################################################################################################################

    public double get_ENSE(CRFClassifier<CoreLabel> crf, String str, int n_in_NSE){

        // wrap sentence in an Annotation object
        Annotation annotation = new Annotation(str);
        // tokenize sentence
        pipeline.annotate(annotation);
        // get the list of tokens
        List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);

        // get the k best sequences from your abstract sequence classifier
        Counter<List<CoreLabel>> kBestSequences = crf.classifyKBest(tokens,
                CoreAnnotations.NamedEntityTagAnnotation.class,
                n_in_NSE);

        // sort the k-best examples
        List<List<CoreLabel>> sortedKBest = Counters.toSortedList(kBestSequences);

        HashMap<String, Double> entities = new HashMap<>();

        for (int i = 0 ; i < sortedKBest.size() ; i ++){
            List<CoreLabel> seq = sortedKBest.get(i);

            double logProb = kBestSequences.getCount(seq);
            double prob = Math.exp(logProb);

            StringJoiner entity = new StringJoiner(" ");
            // Print token and the tag of it as was assigned in the current sequence (e.g., Hussein/Person)
            for (int j = 0 ; j < seq.size() ; j++){
                String word = seq.get(j).word();
                String tag = seq.get(j).get(CoreAnnotations.NamedEntityTagAnnotation.class);

                String entity_string = entity.toString();

                if (tag != "O"){
                    entity.add(word);
                }
                else{
                    if (entity_string.length() > 0) {

                        if(!entities.keySet().contains(entity_string)){
                            entities.put(entity_string, .0);
                        }

                        entities.put(entity.toString(), entities.get(entity_string)+logProb * prob * -1);

                        // To calculate the estimated coverage later ---------------------------------------------------
                        if (!all_entities_probabilities.containsKey(entity_string)){
                            all_entities_probabilities.put(entity_string, .0);
                        }

                        all_entities_probabilities.put(entity_string,
                                all_entities_probabilities.get(entity_string) + logProb * prob * -1);
                        // ---------------------------------------------------------------------------------------------

                        entity = new StringJoiner(" ");
                    }
                }
            }

            if (entity.toString().length() > 0) {
                if(!entities.keySet().contains(entity.toString())){
                    entities.put(entity.toString(), .0);
                }

                entities.put(entity.toString(), entities.get(entity.toString())+logProb * prob);
            }
        }

        double NSE = 0;

        for(String key : entities.keySet()){
            NSE += entities.get(key);
        }

        return NSE;
    }

    // Entity N-Sequence Entropy -- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2586757/
    public List<Map.Entry<Integer, Double>> get_sentences_sorted_based_on_ense(
            CRFClassifier<CoreLabel> crf_model,
            HashMap<Integer, String> sentences,
            int n_in_nse){

        HashMap<Integer, Double> sentence_id_nse_map = new HashMap<>();

        for(int sentence_id: sentences.keySet()){
            sentence_id_nse_map.put(sentence_id, get_ENSE(crf_model, sentences.get(sentence_id), n_in_nse));
        }

        Set<Map.Entry<Integer, Double>> set = sentence_id_nse_map.entrySet();
        List<Map.Entry<Integer, Double>> list = new ArrayList<>(set);
        Collections.sort( list, new Comparator<Map.Entry<Integer, Double>>()
        {
            public int compare( Map.Entry<Integer, Double> o1, Map.Entry<Integer, Double> o2 )
            {
                if (o1.getValue() < o2.getValue()) return 1;
                if (o1.getValue() > o2.getValue()) return -1;
                return 0;
            }
        } );

        return list;
    }

    private ArrayList<String> add_entities_probabilities(double SE, List<CoreLabel> seq){

        ArrayList<String> sequence_entities = new ArrayList<>();

        StringJoiner entity = new StringJoiner(" ");
        // Print token and the tag of it as was assigned in the current sequence (e.g., Hussein/Person)
        for (int j = 0 ; j < seq.size() ; j++){
            String word = seq.get(j).word();
            String tag = seq.get(j).get(CoreAnnotations.NamedEntityTagAnnotation.class);

            String entity_string = entity.toString();

            if (tag != "O"){
                entity.add(word);
            }
            else{
                if (entity_string.length() > 0) {
                    // To calculate the estimated coverage later ---------------------------------------------------
                    if (!all_entities_probabilities.containsKey(entity_string)){
                        all_entities_probabilities.put(entity_string, .0);

                        sequence_entities.add(entity_string);
                    }

                    all_entities_probabilities.put(entity_string,
                            all_entities_probabilities.get(entity_string) + SE * -1);

                    entity = new StringJoiner(" ");
                }
            }
        }

        if (entity.toString().length() > 0) {
            String entity_string = entity.toString();
            // To calculate the estimated coverage later ---------------------------------------------------
            if (!all_entities_probabilities.containsKey(entity_string)){
                all_entities_probabilities.put(entity_string, .0);
                sequence_entities.add(entity_string);
            }

            all_entities_probabilities.put(entity_string,
                    all_entities_probabilities.get(entity_string) + SE * -1);
        }

        return sequence_entities;
    }
}