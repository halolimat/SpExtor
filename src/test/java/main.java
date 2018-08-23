import com.google.common.util.concurrent.AtomicLongMap;
import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.ArraySet;
import edu.stanford.nlp.util.Triple;
import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Logger;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class main {

    static Logger log = Logger.getLogger(main.class.getName());

    public static void main(String args[]) {

        System.out.println("Running!");

        System.exit(0);

        BasicConfigurator.configure();

        double auto_annotation_se_margin = 0.2;
        String dataset_name = "CoNLL-2003-LOC";
        String search_term = "Germany";

        Parameters.auto_annotation_se_margin = auto_annotation_se_margin;
        Parameters.dataset_name = dataset_name;

        Datasets dataset = new Datasets(Parameters.data_dir, dataset_name);

        // ---------------------------------------------------------------------------

        EntityLearning el = new EntityLearning();



        EntitySetExpansion ese = new EntitySetExpansion();

        FeatureFactory ff = new FeatureFactory(dataset.root_dir);
        ff.featurize(dataset.raw_train_sentences);
        System.out.println("Done Featurizing !!!");

        if (Parameters.USE_ESE_PIPELINE) {


            // TODO: make the starting point work with a set

            Set<String> seed_set = new ArraySet<>();
            seed_set.add(search_term);

            // remove the ones that are not noun phrases / the system was not able to extract them as NPs ----------
            Set<String> original_seed_set = new ArraySet<>();
            original_seed_set.addAll(seed_set);

            for (String seed_entity : original_seed_set) {
                if (!ff.nounphrase_id_dict.containsKey(seed_entity)) {
                    seed_set.remove(seed_entity);
                }
            }

            // -------------------------------------------------------------------------------------------------

            List<Map.Entry<String, Double>> new_ranked_set = ese.expand_set(ff, seed_set);

            // top 20 ranked nps
            List<String> new_ranked_list = new_ranked_set.stream().map(Map.Entry::getKey).collect(Collectors.toList()).subList(0, Parameters.ese_candidates_number);

            // simulating user input >> only choosing the TP entities
            seed_set.addAll(new_ranked_list.stream().filter(a -> dataset.train_entities.contains(a)).collect(Collectors.toSet()));

            el.start_active_learning(dataset, seed_set, ff, ese.sim_rank_features_of_entity_set(ff, seed_set, dataset));

        }

        // NOTE: in the final system the pipeline should also contain this, but this is for testing only.
        else {

            //----------------------------------------------------------------------------------------------------------
            //----------------------------------------------------------------------------------------------------------
            //----------------------------------------------------------------------------------------------------------
            // (1) Use trained model to tag testing data and extract entities

            CRFClassifier<CoreLabel> crf_model = train_crf_all_training_sentences(dataset, el);

            ActiveLearning al = new ActiveLearning();
            // calculate the avg nse value for all the unlabeled sentences
            Set<Map.Entry<Integer, String>> unannotated_sentences = dataset.raw_test_sentences.entrySet().stream()
                    .collect(Collectors.toSet());
            List<Map.Entry<Integer, Double>> sentence_id_nse_map = al.get_sentences_sorted_based_on_nse(crf_model,
                    unannotated_sentences);
            double average_NSE = sentence_id_nse_map.stream().mapToDouble(a -> a.getValue()).average().orElse(-1);

            //----------------------------------------------------------------------------------------------------------

            System.out.println("#################################################################################");
            System.out.println("#################################################################################");

            AtomicLongMap crf_model_entities = AtomicLongMap.create();

            for (int test_sen_id : dataset.raw_test_sentences.keySet()) {

                String sentence_text = dataset.raw_test_sentences.get(test_sen_id);

                List<Triple<String, Integer, Integer>> entities = crf_model.classifyToCharacterOffsets(sentence_text);

                if (entities.size() == 0) {
                    continue;
                }

                int start_idx = entities.iterator().next().second;
                int end_idx = entities.iterator().next().third;

                for (Triple<String, Integer, Integer> entity : entities.subList(1, entities.size())) {

                    if (end_idx + 1 == entity.second) {
                        end_idx = entity.third;
                    } else {

                        String e_name = sentence_text.substring(start_idx, end_idx);

                        crf_model_entities.getAndIncrement(e_name);

                        start_idx = entity.second;
                        end_idx = entity.third;
                    }
                }

                String e_name = sentence_text.substring(start_idx, end_idx);

                crf_model_entities.getAndIncrement(e_name);
            }

            // now, filter out the correct entities extracted using CRF --- this is done in the loop below

            List<Map.Entry<String, Long>> sorted = new ArrayList<>(crf_model_entities.asMap().entrySet());
            Collections.sort(sorted, Collections.reverseOrder(Map.Entry.comparingByValue()));

            // ----------- Interactive Learning

            Set<String> new_train = new ArraySet<>();

            int counter = 20;
            for (Map.Entry<String, Long> entity : sorted) {

                // get the sentences with those entities and train the model with them also and then test the model again

                if (dataset.test_entity_sentence_map.containsKey(entity.getKey())) {

                    Set<Integer> sentences_with_entity = dataset.test_entity_sentence_map.get(entity.getKey());

                    for (int sen_id : sentences_with_entity) {
                        new_train.add(dataset.conll_test_sentences.get(sen_id));
                    }
                }

                counter--;
                if (counter == 0) {
                    break;
                }
            }

            // (2) ----------- ESE

            FeatureFactory test_ff = new FeatureFactory(dataset.root_dir);
            // Start Extraction from Test Dataset
            test_ff.featurize(dataset.raw_test_sentences);
            System.out.println("Done Featurizing Testing Data!!!");

            //Set<String> new_train = new ArraySet<>();

            Set<String> seed_entity = new ArraySet<>();

            for (Map.Entry<String, Long> entity : sorted) {
                if (test_ff.nounphrase_id_dict.containsKey(entity.getKey())) {
                    seed_entity.add(entity.getKey());
                    // only add one for now!
                    break;
                }
            }

            List<Map.Entry<String, Double>> new_ranked_set = ese.expand_set(ff, test_ff, seed_entity);
            // top 20 ranked nps
            List<String> new_ranked_list = new_ranked_set.stream().map(Map.Entry::getKey)
                    // only keep the ones which are part of the new dataset (i.e., testing)
                    .filter(a -> test_ff.nounphrase_id_dict.containsKey(a))
                    .collect(Collectors.toList()).subList(0, 20);

            // simulating user input >> only choosing the TP entities
            Set<String> seed_entities = new_ranked_list.stream().filter(a -> dataset.test_entity_sentence_map.keySet().contains(a)).collect(Collectors.toSet());

            for (String entity : seed_entities) {

                Set<Integer> sentences_with_entity = dataset.test_entity_sentence_map.get(entity);

                for (int sen_id : sentences_with_entity) {
                    new_train.add(dataset.conll_test_sentences.get(sen_id));
                }
            }

            // -----------

            String training_sentences_conll = dataset.conll_train_sentences.entrySet().stream()
                    .map((entry) -> entry.getValue()).collect(Collectors.joining("\n\n"));

            String testing_sentences_conll_added = new_train.stream().collect(Collectors.joining("\n\n"));

            String all_training_sentences = training_sentences_conll + "\n\n" + testing_sentences_conll_added;
            CRFClassifier<CoreLabel> crf_m = el.train_CRF(all_training_sentences, dataset.meta.get(2));

            String testing_sentences_conll = dataset.conll_test_sentences.entrySet().stream()
                    .map((entry) -> entry.getValue()).collect(Collectors.joining("\n\n"));
            el.test_CRF(crf_m, testing_sentences_conll);

            ///        -------------

            // calculate the avg nse value for all thxe unlabeled sentences

            sentence_id_nse_map = al.get_sentences_sorted_based_on_nse(crf_m, unannotated_sentences);
            double average_NSE_after = sentence_id_nse_map.stream().mapToDouble(a -> a.getValue()).average().orElse(-1);

            System.out.println(average_NSE);
            System.out.println(average_NSE_after);

        }
    }
}