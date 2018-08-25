import com.google.common.collect.Sets;
import com.google.common.util.concurrent.AtomicLongMap;

import java.util.*;
import java.util.stream.Collectors;

public class EntitySetExpansion {

    // #################################################################################################################

    private List<Set<String>> get_combinations(Set<String> set, long size) {

        Set<Set<String>> powerset = Sets.powerSet(set);

        List<Set<String>> ensemble_features = powerset.stream()
                .filter(se -> se.size() == size)
                .collect(Collectors.toList());

        return ensemble_features;
    }

    // #################################################################################################################

    public double context_dependent_similarity(HashMap<String, Double> kv1, HashMap<String, Double> kv2){

        double numerator = .0;
        double denominator = .0;

        Set<String> union = new HashSet<>(kv1.keySet());
        union.addAll(kv2.keySet());

        for(String f_pattern : union){
            Double kv1_v = kv1.get(f_pattern);
            Double kv2_v = kv2.get(f_pattern);

            if(kv1_v == null){
                kv1_v = 0.;
            }

            if(kv2_v == null){
                kv2_v = 0.;
            }

            numerator += Math.min(kv1_v, kv2_v);
            denominator += Math.max(kv1_v, kv2_v);
        }

        return numerator/denominator;
    }

    // #################################################################################################################

    private HashMap<String, HashMap<String, Double>> vectorize_nps_features(
            HashMap<String, HashMap<String, HashMap<String, Double[]>>> feature_nps_map, Set<String> f_set){
        //-----------------------------------------------
        //System.out.println("Vectorizing NPs Features");

        HashMap<String, HashMap<String, Double>> nounphrase_feature_vectors = new HashMap<>();

        // coarse features
        for(String fn: feature_nps_map.keySet()){

            // fine features
            for(String f: feature_nps_map.get(fn).keySet()){

                // ignore the ones that we are not considering now.. this is useful when we ensemble features
                if(!f_set.contains(f)){
                    continue;
                }

                // System.out.print(fn + " > " + f + " > ");
                for(String np: feature_nps_map.get(fn).get(f).keySet()){
                    //System.out.print(np + ":" + Arrays.toString(feature_nps_map.get(fn).get(f).get(np))+"\t");

                    if(!nounphrase_feature_vectors.containsKey(np)){
                        nounphrase_feature_vectors.put(np, new HashMap<>());
                    }

                    if(!nounphrase_feature_vectors.get(np).containsKey(f)){
                        // [0-2] > 0: raw count , 1: tfidf , 2: tfidf_sum
                        int np_features_vector_weights = 1;
                        nounphrase_feature_vectors.get(np).put(f, feature_nps_map.get(fn).get(f).get(np)[np_features_vector_weights]);
                    }
                }
            }
        }

//        String search_term_id = ff.nounphrase_id_dict.get(search_term);
//        ff.nounphrase_wordsense_features_map.get(search_term_id);
//        ff.nounphrase_shape_features_map.get(search_term_id);
//        ff.nounphrase_lexico_syntactic_features_map.get(search_term_id);
//        ff.word2vec_features.get(search_term_id);

        //System.out.println(nounphrase_feature_vectors.get("American Airlines"));

        // this code is to help in creating the chart as in EgoSet!... didn't work
//        HashMap<String, Integer> scatter_plot = new HashMap<>();
//
//        for(String np : nounphrase_feature_vectors.keySet()){
//
//            if(!scatter_plot.containsKey(np)){
//                scatter_plot.put(np, nounphrase_feature_vectors.get(np).size());
//            }
//
//            for(String f : nounphrase_feature_vectors.get(np).keySet()){
//                if(!scatter_plot.containsKey(f)){
//                    scatter_plot.put(f, 0);
//                }
//
//                scatter_plot.put(f, scatter_plot.get(f) + nounphrase_feature_vectors.get(np).get(f).intValue());
//            }
//        }
//
//        for(String x : scatter_plot.keySet()){
//            System.out.println(x + "\t" + scatter_plot.get(x));
//        }
//
//        System.exit(0);

        return nounphrase_feature_vectors;
    }

    // #################################################################################################################

    private HashMap<String, HashMap<String, HashMap<String, Double[]>>> embed_feature(
            FeatureFactory ff,
            String search_term_id,
            HashMap<String, HashMap<String, HashMap<String, Double[]>>> feature_nps_map,
            String feature_name,
            HashMap<String, AtomicLongMap> nounphrase_feature_map,
            HashMap<String, AtomicLongMap> feature_nounphrase_map ){

        // vocabulary size in equation 1 in http://mickeystroller.github.io/resources/ECMLPKDD2017.pdf
        int E = ff.nounphrase_id_dict.size();

        try {

            Set<String> Fs = nounphrase_feature_map.get(search_term_id).asMap().keySet();

            for (String F : Fs) {

                Set<String> similar_nps = feature_nounphrase_map.get(F).asMap().keySet();

                double sum = 0;
                for (String snp : similar_nps) {
                    sum += feature_nounphrase_map.get(F).get(snp);
                }

                int number_of_np_with_feature = feature_nounphrase_map.get(F).size();

                for (String np_id : similar_nps) {
                    String np = ff.id_nounphrase_dict.get(np_id);

                    if (!feature_nps_map.get(feature_name).containsKey(F)) {
                        feature_nps_map.get(feature_name).put(F, new HashMap<>());
                    }

                    if (!feature_nps_map.get(feature_name).get(F).keySet().contains(np)) {

                        double tfidf = Math.log(1 + feature_nounphrase_map.get(F).get(np_id)) *
                                (Math.log(E) - Math.log(number_of_np_with_feature));

                        double tfidf_sum = Math.log(1 + feature_nounphrase_map.get(F).get(np_id)) *
                                (Math.log(E) - Math.log(sum));

                        feature_nps_map.get(feature_name).get(F).put(np,
                                new Double[]{(double) feature_nounphrase_map.get(F).get(np_id),
                                        tfidf,
                                        tfidf_sum});
                    }
                }
            }
        }

        catch(Exception e){}

        return feature_nps_map;
    }

    // #################################################################################################################

    public HashMap<String, HashMap<String, HashMap<String, Double[]>>> get_similar_nps_features(FeatureFactory ff,
                                                                                                String search_term){

        // feature_nps_map: Coarse Feature -> Fine Feature -> Noun Phrase -> [count, tfidf, tfidf_sum]
        HashMap<String, HashMap<String, HashMap<String, Double[]>>> feature_nps_map = new HashMap<>();
        for(String coarse_feature_name : FeatureFactory.feature_set){
            feature_nps_map.put(coarse_feature_name, new HashMap<>());
        }

        String search_term_id = ff.nounphrase_id_dict.get(search_term);

        // (1) Lexical Features (LF) ///////////////////////////////////////////////////////////////////////////////////
        feature_nps_map = embed_feature(ff, search_term_id, feature_nps_map, "LF", ff.nounphrase_LF_map,
                ff.LF_nounphrase_map);

        // (2) Lexico-Syntactic Features (LS) //////////////////////////////////////////////////////////////////////////
        feature_nps_map = embed_feature(ff, search_term_id, feature_nps_map, "LS", ff.nounphrase_LS_map,
                ff.LS_nounphrase_map);

        // (3) Syntactic Features (SF) /////////////////////////////////////////////////////////////////////////////////
        feature_nps_map = embed_feature(ff, search_term_id, feature_nps_map, "SF", ff.nounphrase_SF_map,
                ff.SF_nounphrase_map);

        // (4) Semantic Features (SeF) /////////////////////////////////////////////////////////////////////////////////
        feature_nps_map = embed_feature(ff, search_term_id, feature_nps_map, "SeF", ff.nounphrase_SeF_map,
                ff.SeF_nounphrase_map);

        // (5) Contextual Features (CF) ////////////////////////////////////////////////////////////////////////////////
        Set<String> W2VFs = ff.CF.get(search_term_id).keySet();
        String feature_name = "CF";
        for (String np_id : W2VFs) {
            String np = ff.id_nounphrase_dict.get(np_id);

            if(!feature_nps_map.get(feature_name).containsKey(search_term_id)){
                feature_nps_map.get(feature_name).put(search_term_id, new HashMap<>());
            }

            if(!feature_nps_map.get(feature_name).get(search_term_id).keySet().contains(np)){
                feature_nps_map.get(feature_name).get(search_term_id).put(np,
                        new Double[]{1.,
                                ff.CF.get(search_term_id).get(np_id),
                                ff.CF.get(search_term_id).get(np_id)});
            }
        }

        return feature_nps_map;
    }

    // #################################################################################################################

    private List<Map.Entry<String, Double>> rank_nps(HashMap<String, HashMap<String, HashMap<String, Double[]>>> feature_nps_map,
                                                     Set<String> f_set, String search_term) {

        HashMap<String, HashMap<String, Double>> nounphrase_feature_vectors = vectorize_nps_features(feature_nps_map, f_set);

        HashSet<String> all_features = new HashSet<>();
        for(String np: nounphrase_feature_vectors.keySet()){
            all_features.addAll(nounphrase_feature_vectors.get(np).keySet());
        }

        //System.out.println("Calculating Similarities!");

        HashMap<String, Double> similarities = new HashMap<>();

        HashMap<String, Double> search_term_fv = nounphrase_feature_vectors.get(search_term);

        for(String np : nounphrase_feature_vectors.keySet()){

            if(np.equals(search_term)) {
                continue;
            }

            HashMap<String, Double> np_fv = nounphrase_feature_vectors.get(np);

            similarities.put(np, context_dependent_similarity(np_fv, search_term_fv));
        }

        List<Map.Entry<String, Double>> sorted = similarities.entrySet().stream().
                sorted((k1, k2) -> -k1.getValue().compareTo(k2.getValue())).collect(Collectors.toList());

        return sorted;
    }

    public HashSet<String> get_fine_features_names(Set<String> f_set,
                                                   HashMap<String, HashMap<String, HashMap<String, Double[]>>> feature_nps_map) {

        HashSet<String> fine_features = new HashSet<>();

        for(String f: feature_nps_map.keySet()){
            if(!f_set.contains(f)){
                continue;
            }

            for(String f_fine : feature_nps_map.get(f).keySet()){
                fine_features.add(f_fine);
            }
        }

        return fine_features;
    }

    private List<Map.Entry<String, Double>> ensemble_ranking(HashMap<String, HashMap<String, HashMap<String, Double[]>>> feature_nps_map, String search_term) {

        Set<Set<String>> powerset = Sets.powerSet(feature_nps_map.keySet());

        List<Set<String>> ensemble_features = powerset.stream()
                .filter(s -> s.size() == feature_nps_map.keySet().size() - 1)
                .collect(Collectors.toList());

//        System.out.println(ensemble_features);
//        System.exit(0);

//        HashSet<String> fine_features_set = get_fine_features_names(feature_nps_map.keySet(), feature_nps_map);
//
//        long set_size = Math.round(fine_features_set.size()-1);
//        List<Set<String>> ensemble_features = get_combinations(fine_features_set, set_size);

        HashMap<String, Double> all_nps = new HashMap<>();

        for (Set<String> f_set : ensemble_features) {

            //System.out.println(f_set);

            // ranked list of nps based on the set of features chosen f_set
            //List<Map.Entry<String, Double>> ranked_sorted_nps = rank_nps(feature_nps_map, f_set, search_term);

            List<Map.Entry<String, Double>> ranked_sorted_nps;

            ranked_sorted_nps = rank_nps(feature_nps_map, get_fine_features_names(f_set, feature_nps_map), search_term);

            List<String> ranked_sorted_nps_list = ranked_sorted_nps.stream().map(Map.Entry::getKey).collect(Collectors.toList());

            for (String np : ranked_sorted_nps_list) {
                if (!all_nps.containsKey(np)) {
                    all_nps.put(np, 1. / ranked_sorted_nps_list.indexOf(np) + 1);
                } else {
                    all_nps.put(np, all_nps.get(np) + 1. / ranked_sorted_nps_list.indexOf(np) + 1);
                }
            }
        }

        List<Map.Entry<String, Double>> sorted = all_nps.entrySet().stream().
                sorted((k1, k2) -> -k1.getValue().compareTo(k2.getValue())).collect(Collectors.toList());

        return sorted;
    }

    // #################################################################################################################

    // Germany => IN+NP | ***** | ***** | = 3 (raw counts)

    private List<Map.Entry<String, Double>> rank_using_atomicincrement(
            HashMap<String, HashMap<String, HashMap<String, Double[]>>> feature_nps_map) {

        HashMap<String, Double> nps_ranked = new HashMap();

        for(String coarse_grained_feature : feature_nps_map.keySet()){
            for(String fine_grained_feature : feature_nps_map.get(coarse_grained_feature).keySet()){
                for(String np : feature_nps_map.get(coarse_grained_feature).get(fine_grained_feature).keySet()){

                    if (!nps_ranked.containsKey(np)) {
                        nps_ranked.put(np, 1.);
                    } else {
                        nps_ranked.put(np, nps_ranked.get(np) + 1);
                    }
                }
            }
        }

        List<Map.Entry<String, Double>> sorted = new ArrayList<>(nps_ranked.entrySet());
        Collections.sort(sorted, Collections.reverseOrder(Map.Entry.comparingByValue()));

        return sorted;
    }

    // #################################################################################################################

    public List<Map.Entry<String, Double>> expand_set(FeatureFactory ff, Set<String> seed_set){

        // TODO: edit this if we want to use more than one seed entity
        String search_term = seed_set.iterator().next();

        HashMap<String, HashMap<String, HashMap<String, Double[]>>> feature_nps_map = get_similar_nps_features(ff, search_term);

        // rank based on ensambling of feature sets and combining the ranks based on mean reciprocal rank
        return ensemble_ranking(feature_nps_map, search_term);
    }

    // using ESE for the next step of extractions after finishing saving the model - > outside the pipeline
    public List<Map.Entry<String, Double>> expand_set(FeatureFactory ff, FeatureFactory test_ff, Set<String> seed_set){

        // TODO: edit this if we want to use more than one seed entity
        String search_term = seed_set.iterator().next();

        // TODO: what if the training dataset do not have an example of the search term!
        HashMap<String, HashMap<String, HashMap<String, Double[]>>> feature_nps_map = get_similar_nps_features(ff, search_term);
        HashMap<String, HashMap<String, HashMap<String, Double[]>>> feature_nps_map_testing = get_similar_nps_features(test_ff, search_term);

        // TODO: this needs to be improved

        // merge nps and features

        for(String coarse_grained_feature : feature_nps_map_testing.keySet()) {
            for (String fine_grained_feature : feature_nps_map_testing.get(coarse_grained_feature).keySet()) {
                for (String np : feature_nps_map_testing.get(coarse_grained_feature).get(fine_grained_feature).keySet()) {

                    if(!feature_nps_map.containsKey(coarse_grained_feature)){
                        feature_nps_map.put(coarse_grained_feature, new HashMap<>());
                    }

                    if (!feature_nps_map.get(coarse_grained_feature).containsKey(fine_grained_feature)){
                        feature_nps_map.get(coarse_grained_feature).put(fine_grained_feature, new HashMap<>());
                    }

                    if(!feature_nps_map.get(coarse_grained_feature).get(fine_grained_feature).containsKey(np)){
                        feature_nps_map.get(coarse_grained_feature).get(fine_grained_feature).put(np, new Double[]{0.,0.,0.});
                    }

                    // This should mimic the line marked as ($*$)
                    feature_nps_map.get(coarse_grained_feature).get(fine_grained_feature).get(np)[0] += 1;
                }
            }
        }

        // rank based on ensambling of feature sets and combining the ranks based on mean reciprocal rank
        return ensemble_ranking(feature_nps_map, search_term);
    }

    public List<Map.Entry<Integer, Double>> rank_features_of_entity_set(FeatureFactory ff, Set<String> seed_set){

        HashMap<Integer, Double> rank_sentences = new HashMap<>();

        for(String entity_name : seed_set){

            HashMap<String, HashMap<String, HashMap<String, Double[]>>> feature_nps_map = get_similar_nps_features(ff, entity_name);

            for(String coarse_grained_feature : feature_nps_map.keySet()){

                Set<String> fine_grained_features = feature_nps_map.get(coarse_grained_feature).keySet();

                for(int sent_id : ff.sentence_nounphrases_map.keySet()){

                    Set<String> sentence_fine_grained_features = ff.sentence_features_map.get(sent_id);
                    Set<String> intersection = new HashSet<>(sentence_fine_grained_features);
                    intersection.retainAll(fine_grained_features);

                    if(!rank_sentences.containsKey(sent_id)){
                        rank_sentences.put(sent_id, 0.);
                    }

                    rank_sentences.put(sent_id, rank_sentences.get(sent_id)+intersection.size());
                }
            }
        }

        List<Map.Entry<Integer, Double>> sentences_mrr_sorted = rank_sentences.entrySet().stream()
                .sorted(Collections.reverseOrder(Map.Entry.comparingByValue())).collect(Collectors.toList());


//        // This counts the f @ k for the sentences ranked using this feature. If the sentence has a positive entity.
//        int counter = 0;
//        int e_counter = 0;
//        HashMap<Integer, Integer> f_at_k = new HashMap<>();
//        for(Map.Entry<Integer, Double> sent : sentences_mrr_sorted){
//            counter++;
//            if(dataset.sentence_entities.containsKey(sent.getKey())){
//                e_counter++;
//            }
//
//            if(counter%10==0){
//                f_at_k.put(counter, e_counter);
//                e_counter=0;
//            }
//        }
//
//        List<Map.Entry<Integer, Integer>> f_at_k_sorted = f_at_k.entrySet().stream()
//                .sorted(Map.Entry.comparingByKey()).collect(Collectors.toList());
//
//
//        for(Map.Entry<Integer, Integer> x : f_at_k_sorted){
//            System.out.println(x.getKey() + "\t" + x.getValue());
//        }
//
//        System.exit(0);

        return sentences_mrr_sorted;
    }

    public List<Map.Entry<Integer, Double>> sim_rank_features_of_entity_set(FeatureFactory ff,
                                                                            Set<String> seed_set,
                                                                            HashMap<Integer, String> conll_train_sentences){

        HashSet<Integer> all_sentences_with_entities_from_set_set = new HashSet<>();

        for(String entity_name : seed_set){
            Set<Integer> entity_sentences = ff.nounphrase_sentences_map.get(ff.nounphrase_id_dict.get(entity_name));
            all_sentences_with_entities_from_set_set.addAll(entity_sentences);
        }

        HashMap<Integer, Double> sentences_similarities = new HashMap<>();

        // Now, calculate the similarity between each sentence and these sentences based on Jaccard Sim
        for(int en_sent_id: all_sentences_with_entities_from_set_set){

            Set<String> en_sent_features = ff.sentence_features_map.get(en_sent_id);

            for(int sentence_id: conll_train_sentences.keySet()){
                Set<String> sent_features = ff.sentence_features_map.get(sentence_id);

                if(sent_features == null){
                    continue;
                }

                Set<String> intersection = new HashSet<>(en_sent_features);
                intersection.retainAll(sent_features);

                Set<String> union = new HashSet<>(en_sent_features);
                union.addAll(sent_features);

                if (!sentences_similarities.containsKey(sentence_id)) {
                    sentences_similarities.put(sentence_id, 0.);
                }
                sentences_similarities.put(sentence_id, sentences_similarities.get(sentence_id) + (intersection.size() / (double) union.size()));
            }
        }

        for(int sentence_id: sentences_similarities.keySet()){
            sentences_similarities.put(sentence_id, sentences_similarities.get(sentence_id)/all_sentences_with_entities_from_set_set.size());
        }

        List<Map.Entry<Integer, Double>> sentences_similarities_sorted = sentences_similarities.entrySet().stream()
                .sorted(Collections.reverseOrder(Map.Entry.comparingByValue())).collect(Collectors.toList());

        return sentences_similarities_sorted;
    }
}