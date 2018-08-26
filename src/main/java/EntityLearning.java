import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.objectbank.ObjectBank;
import edu.stanford.nlp.sequences.SeqClassifierFlags;
import edu.stanford.nlp.process.AmericanizeFunction;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.nio.file.StandardOpenOption.*;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class EntityLearning {

    public SeqClassifierFlags get_flags(String map) {

        SeqClassifierFlags flags = new SeqClassifierFlags();
        flags.wordFunction = new AmericanizeFunction();
        //flags.useDistSim=true;
        //flags.distSimLexicon="/u/nlp/data/pos_tags_are_useless/egw4-reut.512.clusters";
        flags.numberEquivalenceDistSim = true;
        flags.unknownWordDistSimClass = "0";
        // "word=0,pos=1,chunk=2,answer=3"
        flags.map = map;
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

    public CRFClassifier<CoreLabel> train_CRF(String training_text, String map) {
        SeqClassifierFlags flags = get_flags(map);
        CRFClassifier<CoreLabel> crf = new CRFClassifier<>(flags);
        ObjectBank trainObjBank = crf.makeObjectBankFromString(training_text, crf.defaultReaderAndWriter());
        crf.train(trainObjBank);
        return crf;
    }

    public CRFClassifier<CoreLabel> start_active_learning(HashMap<Integer, String> conll_train_sentences,
                                                          HashMap<Integer, String> raw_train_sentences,
                                                          String data_map,
                                                          Set<String> seed_set,
                                                          FeatureFactory ff,
                                                          int sample_size,
                                                          double auto_annotation_se_margin,
                                                          String seed_entity) {

        // number of docs -> nse
        List<String> sigma = new ArrayList<>();
        sigma.add("# Sentences\tSigma");

        // annotate sentences with those entities and train a model to start Active Learning
        Set<String> annotated_sentences = new HashSet<>();
        Set<Integer> annotated_sentences_ids = new HashSet<>();

        String training_sentences_conll = "";
        CRFClassifier<CoreLabel> first_crf_model = null;

        // bi-sequence entropy!
        ActiveLearning al = new ActiveLearning();

        // using seed entities from the Entity Set Expansion method
        if (seed_set.size() > 0) {

            // Query users to annotated the sentences sampled using ESE.
            while(true){
                int size_before = annotated_sentences_ids.size();

                for (String entity : seed_set) {

                    for (int sentence_id : ff.nounphrase_sentences_map.get(ff.nounphrase_id_dict.get(entity))) {
                        if (!annotated_sentences_ids.contains(sentence_id) && annotated_sentences_ids.size()!=sample_size) {
                            annotated_sentences_ids.add(sentence_id);
                            annotated_sentences.add(conll_train_sentences.get(sentence_id));
                            // break to iteratively get sentences from all seeds
                            break;
                        }
                    }
                }

                // to make sure that we are breaking the while loop when we are done!
                if(annotated_sentences_ids.size() == sample_size || size_before == annotated_sentences_ids.size()){
                    break;
                }
            }

            training_sentences_conll = annotated_sentences.stream().collect(Collectors.joining("\n\n"));

            // 1st CRF Model ###########################################################################################
            first_crf_model = train_CRF(training_sentences_conll, data_map);

            // #########################################################################################################
            // ####################################################################################### Auto Annotation #

            if(auto_annotation_se_margin > 0) {

                // calculate the avg nse value for all the unlabeled sentences
                List<Map.Entry<Integer, ArrayList<Double>>> sentence_id_nse_map = al.get_sentences_sorted_based_on_nse(
                        first_crf_model,
                        raw_train_sentences);

                for (Map.Entry<Integer, ArrayList<Double>> s : sentence_id_nse_map) {

                    double margin_percentage = s.getValue().get(0) / s.getValue().get(1);

                    // Auto annotating sentences with margin percentage difference of <= 50%
                    if (margin_percentage <= auto_annotation_se_margin) {
                        annotated_sentences_ids.add(s.getKey());

                        // annotate based on the output of the model
                        List<CoreLabel> seq = first_crf_model.classify(raw_train_sentences.get(s.getKey())).get(0);

                        String conll_sentence = "";
                        for (CoreLabel cl : seq) {

                            String line = "";

                            if (data_map.charAt(data_map.length() - 1) == '3')
                                line = cl.word() + " " + "-" + " " + "-" + " ";
                            else if (data_map.charAt(data_map.length() - 1) == '2')
                                line = cl.word() + " " + "-" + " ";
                            else
                                line = cl.word() + " ";

                            String tag = cl.get(CoreAnnotations.AnswerAnnotation.class);
                            if (tag != null) {
                                line += tag + "\n";
                            } else {
                                line += "O\n";
                            }

                            conll_sentence += line;
                        }

                        annotated_sentences.add(conll_sentence);
                    }
                }
            }

            // ####################################################################################### Auto Annotation #
            // #########################################################################################################

            // #########################################################################################################
            // Calculate Estimated Coverage as in equation 1 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2586757/

            // use the new trained model using auto annotations
            List<Map.Entry<Integer, Double>> _sentence_id_nse_map = al._get_sentences_sorted_based_on_nse(first_crf_model, raw_train_sentences);

            double average_NSE = _sentence_id_nse_map.stream().mapToDouble(a -> a.getValue()).average().orElse(-1);

            sigma.add(Integer.toString(annotated_sentences_ids.size())+ "\t" + Double.toString(1-average_NSE));
        }

        // Start Active Learning >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        int annotated_sentences_size = annotated_sentences_ids.size();

        CRFClassifier<CoreLabel> new_crf_model = first_crf_model;

        try {

            while (annotated_sentences_size < raw_train_sentences.size()) {

                // TODO: change this.. ugly!!!!
                // calculate the nse value for all the unlabeled sentences
                Map<Integer, String> unannotated_sentences = raw_train_sentences.entrySet().stream()
                        .filter(a -> !annotated_sentences_ids.contains(a.getKey())).collect(Collectors.toMap(Map.Entry::getKey,
                                Map.Entry::getValue,
                                (a, b) -> b));

                // NSE
                List<Map.Entry<Integer, ArrayList<Double>>> sentence_id_nse_map = al.get_sentences_sorted_based_on_nse(
                        new_crf_model,
                        new HashMap<>(unannotated_sentences));

                //******************************************************************************************************

                // NOW ---- rank sentences based on NSE or NSE+features ************************************************

                HashMap<Integer, Double> sentences_rank = new HashMap<>();

                for(Map.Entry<Integer, ArrayList<Double>> s : sentence_id_nse_map){
                    sentences_rank.put(s.getKey(), s.getValue().get(s.getValue().size()-1));
                }

                List<Map.Entry<Integer, Double>> sentences_rank_sorted = sentences_rank.entrySet().stream()
                        .sorted(Collections.reverseOrder(Map.Entry.comparingByValue())).collect(Collectors.toList());

                int count_sentences = sample_size;
                for (Map.Entry<Integer, Double> sentence : sentences_rank_sorted) {

                    if(!annotated_sentences_ids.contains(sentence.getKey())) {
                        annotated_sentences_ids.add(sentence.getKey());
                        annotated_sentences.add(conll_train_sentences.get(sentence.getKey()));

                        count_sentences--;
                        if (count_sentences == 0) {
                            break;
                        }
                    }
                }

                training_sentences_conll = annotated_sentences.stream().collect(Collectors.joining("\n\n"));

                //******************************************************************************************************

                System.out.println("#########################################################################################");
                System.out.println("#########################################################################################");
                System.out.println("Number of ann sentences = " +
                        annotated_sentences_ids.size() + "/" +
                        conll_train_sentences.size());
                System.out.println("#########################################################################################");
                System.out.println("#########################################################################################");

                // CRF model ###########################################################################################
                new_crf_model = train_CRF(training_sentences_conll, data_map);

                // #########################################################################################################
                // ####################################################################################### Auto Annotation #

                if(auto_annotation_se_margin > 0) {

                    sentence_id_nse_map = al.get_sentences_sorted_based_on_nse(new_crf_model, new HashMap<>(unannotated_sentences));

                    for (Map.Entry<Integer, ArrayList<Double>> s : sentence_id_nse_map) {

                        double margin_percentage = s.getValue().get(0) / s.getValue().get(1);

                        // Auto annotating sentences with margin percentage difference of <= 50%
                        if (margin_percentage <= auto_annotation_se_margin) {
                            annotated_sentences_ids.add(s.getKey());

                            // the below line makes the annotation automatic
                            //annotated_sentences.add(dataset.conll_train_sentences.get(s.getKey()));

                            // annotate based on the output of the model
                            List<CoreLabel> seq = new_crf_model.classify(raw_train_sentences.get(s.getKey())).get(0);

                            String conll_sentence = "";
                            for (CoreLabel cl : seq) {

                                String line = "";

                                if (data_map.charAt(data_map.length() - 1) == '3')
                                    line = cl.word() + " " + "-" + " " + "-" + " ";
                                else if (data_map.charAt(data_map.length() - 1) == '2')
                                    line = cl.word() + " " + "-" + " ";
                                else
                                    line = cl.word() + " ";

                                String tag = cl.get(CoreAnnotations.AnswerAnnotation.class);
                                if (tag != null) {
                                    line += tag + "\n";
                                } else {
                                    line += "O\n";
                                }

                                conll_sentence += line;
                            }

                            annotated_sentences.add(conll_sentence);
                        }
                    }
                }

                // ####################################################################################### Auto Annotation #
                // #########################################################################################################

                // add the average NSE result
                List<Map.Entry<Integer, Double>> new_sentence_id_nse_map = al._get_sentences_sorted_based_on_nse(new_crf_model,
                        raw_train_sentences);
                double average_NSE = new_sentence_id_nse_map.stream().mapToDouble(a -> a.getValue()).average().orElse(-1);
                sigma.add(Integer.toString(annotated_sentences_ids.size())+ "\t" + Double.toString(1-average_NSE));

                // loop
                annotated_sentences_size = annotated_sentences_ids.size();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        try {
            Files.write(Paths.get("out/seMargin_" + Double.toString(auto_annotation_se_margin) + "_sampleSize_" + sample_size + "_seeds_" + seed_entity + ".output"), sigma, UTF_8, TRUNCATE_EXISTING, CREATE);
        }
        catch(Exception e){
            e.printStackTrace();
        }

        return new_crf_model;
    }
}