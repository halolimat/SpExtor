import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.objectbank.ObjectBank;
import edu.stanford.nlp.sequences.SeqClassifierFlags;
import edu.stanford.nlp.process.AmericanizeFunction;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.nio.file.StandardOpenOption.*;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class EntityLearning {

    // #################################################################################################################
    // #################################################################################################################
    // #################################################################################################################

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

    public HashMap<String, Double> test_CRF(CRFClassifier<CoreLabel> crf, String testing_text) {

        ObjectBank testObjBank = crf.makeObjectBankFromString(testing_text, crf.defaultReaderAndWriter());
        StringWriter strOut = new StringWriter();

        try {
            crf.classifyAndWriteAnswers(testObjBank, new PrintWriter(strOut), crf.defaultReaderAndWriter(), true);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }

        return crf.get_testing_results();
    }

    public CRFClassifier<CoreLabel> start_active_learning(Datasets dataset,
                                                          Set<String> seed_set,
                                                          FeatureFactory ff,
                                                          List<Map.Entry<Integer, Double>> sentences_mrr_rank) {


        // estimated coverage vs. number of sentences
        HashMap<Integer, Double> estimated_coverages = new HashMap<>();

        // number of docs -> F-score
        HashMap<Integer, HashMap<String, Double>> evaluation_result = new HashMap<>();
        // this will contain the random samples indexed by the F-Scores
        TreeMap<Double, HashSet<String>> f_scores_samples_mapping = new TreeMap<>();
        TreeMap<Double, HashSet<Integer>> f_scores_samples_mapping_ids = new TreeMap<>();

        String all_training_sentences = dataset.conll_train_sentences.entrySet()
                .stream().map(e -> e.getValue()).collect(Collectors.joining("\n\n"));

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
            // get at most the number of sentences defined as sample size in parameters
            while(true){
                int size_before = annotated_sentences_ids.size();

                for (String entity : seed_set) {

                    for (int sentence_id : ff.nounphrase_sentences_map.get(ff.nounphrase_id_dict.get(entity))) {
                        if (!annotated_sentences_ids.contains(sentence_id) && annotated_sentences_ids.size()!=Parameters.sample_size) {
                            annotated_sentences_ids.add(sentence_id);
                            annotated_sentences.add(dataset.conll_train_sentences.get(sentence_id));
                            // break to iteratively get sentences from all seeds
                            break;
                        }
                    }
                }

                // to make sure that we are breaking the while loop when we are done!
                if(annotated_sentences_ids.size() == Parameters.sample_size || size_before == annotated_sentences_ids.size()){
                    break;
                }
            }

            training_sentences_conll = annotated_sentences.stream().collect(Collectors.joining("\n\n"));

            //String testing_sentences_conll = dataset.conll_train_sentences.entrySet().stream().map(e->e.getValue())
            //        .collect(Collectors.joining("\n\n"));

            // 1st CRF Model ###########################################################################################
            first_crf_model = train_CRF(training_sentences_conll, dataset.meta.get(2));

            // #########################################################################################################
            // ####################################################################################### Auto Annotation #

            if(Parameters.auto_annotation_se_margin > 0) {

                // calculate the avg nse value for all the unlabeled sentences
                //Set<Map.Entry<Integer, String>> unannotated_sentences = dataset.raw_train_sentences.entrySet().stream()
                //        .filter(a -> !annotated_sentences_ids.contains(a.getKey())).collect(Collectors.toSet());
                List<Map.Entry<Integer, ArrayList<Double>>> sentence_id_nse_map = al.get_sentences_sorted_based_on_nse(
                        first_crf_model,
                        dataset.raw_train_sentences);


                for (Map.Entry<Integer, ArrayList<Double>> s : sentence_id_nse_map) {

                    double margin_percentage = s.getValue().get(0) / s.getValue().get(1);

                    // Auto annotating sentences with margin percentage difference of <= 50%
                    if (margin_percentage <= Parameters.auto_annotation_se_margin) {
                        annotated_sentences_ids.add(s.getKey());

                        // the below line makes the annotation automatic
                        //annotated_sentences.add(dataset.conll_train_sentences.get(s.getKey()));

                        // annotate based on the output of the model
                        List<CoreLabel> seq = first_crf_model.classify(dataset.raw_train_sentences.get(s.getKey())).get(0);

                        String conll_sentence = "";
                        for (CoreLabel cl : seq) {

                            String line = "";

                            if (dataset.meta.get(2).charAt(dataset.meta.get(2).length() - 1) == '3')
                                line = cl.word() + " " + "-" + " " + "-" + " ";
                            else if (dataset.meta.get(2).charAt(dataset.meta.get(2).length() - 1) == '2')
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

////                    System.out.println(first_crf_model.classifyWithInlineXML(dataset.raw_train_sentences.get(sid)));
////
////                    System.out.println("-------------");
////                    System.out.println(conll_sentence);
////                    System.out.println("-------------");
////                    System.out.println(dataset.conll_train_sentences.get(sid));
////                    System.out.println("-------------");

                    }
                }

                //training_sentences_conll = annotated_sentences.stream().collect(Collectors.joining("\n\n"));

                //String testing_sentences_conll = dataset.conll_train_sentences.entrySet().stream().map(e->e.getValue())
                //        .collect(Collectors.joining("\n\n"));

                // ESE Auto annotation model model ###########################################################################################
                //first_crf_model = train_CRF(training_sentences_conll, dataset.meta.get(2));
            }

            // ####################################################################################### Auto Annotation #
            // #########################################################################################################

            // result of ESE
            evaluation_result.put(annotated_sentences_ids.size(), test_CRF(first_crf_model, all_training_sentences));

            // #########################################################################################################
            // Calculate Estimated Coverage as in equation 1 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2586757/

            // use the new trained model using auto annotations --- just run to update the estimated coverage using all_entities_probabilities
            List<Map.Entry<Integer, Double>> _sentence_id_nse_map = al._get_sentences_sorted_based_on_nse(first_crf_model, dataset.raw_train_sentences);

            double average_NSE = _sentence_id_nse_map.stream().mapToDouble(a -> a.getValue()).average().orElse(-1);
            evaluation_result.get(annotated_sentences_ids.size()).put("AVG-NSE", average_NSE);

            // Number of Entities in annotated sentences.
            int number_of_entities_m = annotated_sentences_ids.stream().map(a->dataset.train_sentence_entities
                    .get(a)).filter(Objects::nonNull).map(a->a.size()).mapToInt(a->a).sum();

            double expected_number_of_entities =  al.all_entities_probabilities.keySet().stream()
                    .collect(Collectors.summingDouble(a->al.all_entities_probabilities.get(a)));

            double estimated_coverage = number_of_entities_m/(number_of_entities_m+expected_number_of_entities);

            estimated_coverages.put(annotated_sentences_ids.size(),estimated_coverage);

//            System.out.println(evaluation_result);
//            System.exit(0);

            // #########################################################################################################
        }

        // random sampling of the first few sentences to replace the ESE method
        else {

            // TODO: make this routine use the random sampling function below!

            // random sampling for 100 times then take the average scores
            int sampling_times_counter = Parameters.sample_size;

            List<Integer> conll_train_sentences_keys = new ArrayList<>(dataset.conll_train_sentences.keySet());

            // random sample 100 sentences
            Random random = new Random(1);

            HashMap<String, Double> results = new HashMap<>();

            while (sampling_times_counter > 0) {
                sampling_times_counter--;

                // just to keep only 100 sentences at most!
                Set<Integer> annotated_sentences_ids_temp = new HashSet<>();

                while (true) {
                    int random_sentence_id = conll_train_sentences_keys.get(random.nextInt(conll_train_sentences_keys.size()));
                    annotated_sentences_ids_temp.add(random_sentence_id);

                    if (annotated_sentences_ids_temp.size() % Parameters.sample_size == 0){
                        break;
                    }
                }

                List<String> annotated_sentences_temp = new ArrayList<>();

                for(int sen_id : annotated_sentences_ids_temp){
                    annotated_sentences_temp.add(dataset.conll_train_sentences.get(sen_id));
                }

                training_sentences_conll = annotated_sentences_temp.stream().collect(Collectors.joining("\n\n"));

                //String testing_sentences_conll = dataset.conll_train_sentences.entrySet().stream().map(e->e.getValue())
                //        .collect(Collectors.joining("\n\n"));

                // random CRF Model ####################################################################################
                CRFClassifier<CoreLabel> temp_crf_model = train_CRF(training_sentences_conll, dataset.meta.get(2));

                // result of ESE
                HashMap<String, Double> res = test_CRF(temp_crf_model, all_training_sentences);

                for (String r : res.keySet()) {
                    if (!results.keySet().contains(r)) {
                        results.put(r, res.get(r));
                    } else {
                        results.put(r, results.get(r) + res.get(r));
                    }
                }

                f_scores_samples_mapping.put(res.get("f"), new HashSet<>(annotated_sentences_temp));
                f_scores_samples_mapping_ids.put(res.get("f"), new HashSet<>(annotated_sentences_ids_temp));
            }

            // ---------------------------------------------------

            for (String r : results.keySet()) {
                results.put(r, results.get(r) / Parameters.sample_size);
            }

            // add the average of the 100 runs
            evaluation_result.put(Parameters.sample_size, results);

            // adding the medoid sample as the starting sample of active learning... which represents the mean

            double mean_f_score = evaluation_result.get(Parameters.sample_size).get("f");

            Double floor = f_scores_samples_mapping.floorKey(mean_f_score);
            Double ceiling = f_scores_samples_mapping.ceilingKey(mean_f_score);

            double medoid_key = 0.;

            if (floor == null && ceiling != null) {
                medoid_key = ceiling;
            } else {
                medoid_key = floor;
            }

            // add the medoid sample sentences
            annotated_sentences.addAll(f_scores_samples_mapping.get(medoid_key));
            annotated_sentences_ids.addAll(f_scores_samples_mapping_ids.get(medoid_key));

            // change the value of average to the medoid sample value
            training_sentences_conll = annotated_sentences.stream().collect(Collectors.joining("\n\n"));

            // 1st CRF Model (medoid) ##################################################################################
            first_crf_model = train_CRF(training_sentences_conll, dataset.meta.get(2));
            evaluation_result.put(annotated_sentences_ids.size(), test_CRF(first_crf_model, all_training_sentences));

            // calculate the avg nse value for all the unlabeled sentences
            //Set<Map.Entry<Integer, String>> unannotated_sentences = dataset.raw_train_sentences.entrySet().stream()
            //        .filter(a -> !annotated_sentences_ids.contains(a.getKey())).collect(Collectors.toSet());
            List<Map.Entry<Integer, Double>> sentence_id_nse_map = al._get_sentences_sorted_based_on_nse(first_crf_model,
                    dataset.raw_train_sentences);
            double average_NSE = sentence_id_nse_map.stream().mapToDouble(a -> a.getValue()).average().orElse(-1);
            evaluation_result.get(annotated_sentences_ids.size()).put("AVG-NSE", average_NSE);

            // #########################################################################################################
            // Calculate Estimated Coverage as in equation 1 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2586757/

            // Number of Entities in annotated sentences.
            int number_of_entities_m = annotated_sentences_ids.stream().map(a->dataset.train_sentence_entities
                    .get(a)).filter(Objects::nonNull).map(a->a.size()).mapToInt(a->a).sum();

            double expected_number_of_entities =  al.all_entities_probabilities.keySet().stream()
                    .collect(Collectors.summingDouble(a->al.all_entities_probabilities.get(a)));

            double estimated_coverage = number_of_entities_m/(number_of_entities_m+expected_number_of_entities);

            estimated_coverages.put(annotated_sentences_ids.size(),estimated_coverage);
        }

        // Start Active Learning >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        int annotated_sentences_size = annotated_sentences_ids.size();

        CRFClassifier<CoreLabel> new_crf_model = first_crf_model;

        try {

            while (annotated_sentences_size < dataset.raw_train_sentences.size()) {

                // TODO: change this.. ugly!!!!
                // calculate the nse value for all the unlabeled sentences
                Map<Integer, String> unannotated_sentences = dataset.raw_train_sentences.entrySet().stream()
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

                if(Parameters.USE_NSE) {
                    for(Map.Entry<Integer, ArrayList<Double>> s : sentence_id_nse_map){
                        sentences_rank.put(s.getKey(), s.getValue().get(s.getValue().size()-1));
                    }
                }

//                // NSE+Jaccard
//                if(Parameters.USE_NSE_and_Features){
//                    Map<Integer, Double> mapFromList = sentences_mrr_rank.stream()
//                            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
//
//                    for(Map.Entry<Integer, Double> sen : sentence_id_nse_map){
//
//                        // This is the jaccard similarity between the sentence and all sentences with entities found by ESE
//                        Double x = mapFromList.get(sen.getKey());
//
//                        if(x == null){
//                            sentences_rank.put(sen.getKey(), sen.getValue());
//                        }
//
//                        else{
//                            sentences_rank.put(sen.getKey(), sen.getValue()*Math.pow(x,Parameters.BETA));
//                        }
//                    }
//                }

                List<Map.Entry<Integer, Double>> sentences_rank_sorted = sentences_rank.entrySet().stream()
                        .sorted(Collections.reverseOrder(Map.Entry.comparingByValue())).collect(Collectors.toList());

//                // #####################################################################################################
//                // #####################################################################################################
//                // #####################################################################################################
//                System.out.println("HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE");
//
//                // This counts the f @ k for the sentences ranked using this feature. If the sentence has a positive entity.
//                int c = 0;
//                int e_counter = 0;
//                HashMap<Integer, Double> f_at_k = new HashMap<>();
//                for(Map.Entry<Integer, Double> sent : sentences_rank_sorted){
//                    c++;
//                    if(dataset.sentence_entities.containsKey(sent.getKey())){
//                        e_counter++;
//                    }
//
//                    if(c%10==0){
//                        f_at_k.put(c, (e_counter+.0));
//                        e_counter=0;
//                    }
//                }
//
//                List<Map.Entry<Integer, Double>> f_at_k_sorted = f_at_k.entrySet().stream()
//                        .sorted(Map.Entry.comparingByKey()).collect(Collectors.toList());
//
//
//                for(Map.Entry<Integer, Double> x : f_at_k_sorted){
//                    System.out.println(x.getKey() + "\t" + x.getValue());
//                }
//
//
//                System.exit(0);
//                // #####################################################################################################
//                // #####################################################################################################
//                // #####################################################################################################

                int count_sentences = Parameters.sample_size;
                for (Map.Entry<Integer, Double> sentence : sentences_rank_sorted) {

                    if(!annotated_sentences_ids.contains(sentence.getKey())) {
                        //training_sentences_conll += "\n\n" + dataset.conll_train_sentences.get(sentence.getKey());
                        annotated_sentences_ids.add(sentence.getKey());
                        annotated_sentences.add(dataset.conll_train_sentences.get(sentence.getKey()));

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
                        dataset.conll_train_sentences.size());
                System.out.println("#########################################################################################");
                System.out.println("#########################################################################################");

                // CRF model ###########################################################################################
                new_crf_model = train_CRF(training_sentences_conll, dataset.meta.get(2));

                // #########################################################################################################
                // ####################################################################################### Auto Annotation #

                if(Parameters.auto_annotation_se_margin > 0) {

                    sentence_id_nse_map = al.get_sentences_sorted_based_on_nse(new_crf_model, new HashMap<>(unannotated_sentences));

                    for (Map.Entry<Integer, ArrayList<Double>> s : sentence_id_nse_map) {

                        double margin_percentage = s.getValue().get(0) / s.getValue().get(1);

                        // Auto annotating sentences with margin percentage difference of <= 50%
                        if (margin_percentage <= Parameters.auto_annotation_se_margin) {
                            annotated_sentences_ids.add(s.getKey());

                            // the below line makes the annotation automatic
                            //annotated_sentences.add(dataset.conll_train_sentences.get(s.getKey()));

                            // annotate based on the output of the model
                            List<CoreLabel> seq = new_crf_model.classify(dataset.raw_train_sentences.get(s.getKey())).get(0);

                            String conll_sentence = "";
                            for (CoreLabel cl : seq) {

                                String line = "";

                                if (dataset.meta.get(2).charAt(dataset.meta.get(2).length() - 1) == '3')
                                    line = cl.word() + " " + "-" + " " + "-" + " ";
                                else if (dataset.meta.get(2).charAt(dataset.meta.get(2).length() - 1) == '2')
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

                    //training_sentences_conll = annotated_sentences.stream().collect(Collectors.joining("\n\n"));

                    // after auto annotation
                    //new_crf_model = train_CRF(training_sentences_conll, dataset.meta.get(2));

                }

                // ####################################################################################### Auto Annotation #
                // #########################################################################################################

                HashMap<String, Double> eval_results = test_CRF(new_crf_model, all_training_sentences);

                // result of 1st iteration of active learning
                evaluation_result.put(annotated_sentences_ids.size(), eval_results);

                // add the average NSE result
                List<Map.Entry<Integer, Double>> new_sentence_id_nse_map = al._get_sentences_sorted_based_on_nse(new_crf_model,
                        dataset.raw_train_sentences);
                double average_NSE = new_sentence_id_nse_map.stream().mapToDouble(a -> a.getValue()).average().orElse(-1);
                evaluation_result.get(annotated_sentences_ids.size()).put("AVG-NSE", average_NSE);

                // #########################################################################################################
                // Calculate Estimated Coverage as in equation 1 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2586757/

                // Number of Entities in annotated sentences.
                int number_of_entities_m = annotated_sentences_ids.stream().map(a->dataset.train_sentence_entities
                        .get(a)).filter(Objects::nonNull).map(a->a.size()).mapToInt(a->a).sum();

                double expected_number_of_entities =  al.all_entities_probabilities.keySet().stream()
                        .collect(Collectors.summingDouble(a->al.all_entities_probabilities.get(a)));

                double estimated_coverage = number_of_entities_m/(number_of_entities_m+expected_number_of_entities);

                estimated_coverages.put(annotated_sentences_ids.size(),estimated_coverage);

                // #########################################################################################################

                // loop
                annotated_sentences_size = annotated_sentences_ids.size();

//                if(annotated_sentences_ids.size() >= 3500){
//                    break;
//                }

//                System.out.println(evaluation_result);
//                System.exit(0);

                if (Parameters.stop_on_100F && 1 == (Math.round(eval_results.get("f") * 100) / 100)){
                    break;
                }

            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        List<String> output_lines = new ArrayList<>();

        output_lines.add("# Sentences\tPrecision\tRecall\tF-1\tTP\tFP\tFN\tAVG-NSE\tEstimated-Coverage\tSigma");

        Map<Integer, HashMap> map = new TreeMap<>(evaluation_result);

        // AVG-NSE values should be shifted one row above and the last row should be 0
        // This is because we don't want to run NSE again after training the model, we just take the value when we
        // calculate it in the next iteration!
        for (int number_of_sentences : map.keySet()) {

            double sigma = 1-evaluation_result.get(number_of_sentences).get("AVG-NSE");

            if (Parameters.auto_annotation_se_margin > .1)
                sigma -= Parameters.auto_annotation_se_margin-sigma;

            output_lines.add( number_of_sentences + "\t" +
                    evaluation_result.get(number_of_sentences).get("p") + "\t" +
                    evaluation_result.get(number_of_sentences).get("r") + "\t" +
                    evaluation_result.get(number_of_sentences).get("f") + "\t" +
                    evaluation_result.get(number_of_sentences).get("tp") + "\t" +
                    evaluation_result.get(number_of_sentences).get("fp") + "\t" +
                    evaluation_result.get(number_of_sentences).get("fn") + "\t" +
                    evaluation_result.get(number_of_sentences).get("AVG-NSE") + "\t" +
                    estimated_coverages.get(number_of_sentences) + "\t" +
                    Double.toString(sigma)
            );
        }

        try {
            Files.write(Paths.get("Output/" + Parameters.dataset_name+"_"+Double.toString(Parameters.auto_annotation_se_margin)), output_lines, UTF_8, TRUNCATE_EXISTING, CREATE);
        }
        catch(Exception e){
            System.out.println("Cannot write to file");
            System.exit(0);
        }

        return new_crf_model;
    }

    // #################################################################################################################
    // #################################################################################################################
    // #################################################################################################################

    public HashMap<Integer, HashMap<String, Double>> start_random_sampling(Datasets dataset, int iterations) {

        // number of docs -> F-score
        HashMap<Integer, HashMap<String, Double>> evaluation_result = new HashMap<>();

        String all_training_sentences = dataset.conll_train_sentences.entrySet()
                .stream().map(e -> e.getValue()).collect(Collectors.joining("\n\n"));

        Set<Integer> annotated_sentences_ids = new HashSet<>();

        List<Integer> conll_train_sentences_keys = new ArrayList<>(dataset.conll_train_sentences.keySet());
        int ann_sentences_number = annotated_sentences_ids.size();

        String training_sentences_conll = "";

        Random random = new Random(1);

        // random sample and annotate sentences for the number of iterations passed to this function
        while (iterations > 0) {
            iterations--;

            while (true) {

                int random_sentence_id = conll_train_sentences_keys.get(random.nextInt(conll_train_sentences_keys.size()));
                annotated_sentences_ids.add(random_sentence_id);

                if (annotated_sentences_ids.size() % Parameters.sample_size == 0 && ann_sentences_number != annotated_sentences_ids.size()){
                    ann_sentences_number = annotated_sentences_ids.size();
                    break;
                }
            }

            List<String> annotated_sentences = new ArrayList<>();

            for(int sen_id : annotated_sentences_ids){
                annotated_sentences.add(dataset.conll_train_sentences.get(sen_id));
            }

            training_sentences_conll = annotated_sentences.stream().collect(Collectors.joining("\n\n"));

            //String testing_sentences_conll = dataset.conll_train_sentences.entrySet().stream().map(e->e.getValue())
            //        .collect(Collectors.joining("\n\n"));

            // CRF Model random sampling all ###########################################################################
            CRFClassifier<CoreLabel> crf_model = train_CRF(training_sentences_conll, dataset.meta.get(2));

            // result of ESE
            HashMap<String, Double> res = test_CRF(crf_model, all_training_sentences);

            HashMap<String, Double> results = new HashMap<>();

            for (String r : res.keySet()) {
                if (!results.keySet().contains(r)) {
                    results.put(r, res.get(r));
                } else {
                    results.put(r, results.get(r) + res.get(r));
                }
            }

            evaluation_result.put(annotated_sentences_ids.size(), results);

            System.out.println("#####################################################################################");
            System.out.println("#####################################################################################");
            System.out.println("Number of Annotated Docs = " +
                    annotated_sentences_ids.size() + "/" +
                    dataset.conll_train_sentences.size());
            System.out.println("#####################################################################################");
            System.out.println("#####################################################################################");

            if(annotated_sentences_ids.size() >= 11300){
                break;
            }

        }

        List<String> output_lines = new ArrayList<>();

        output_lines.add("# Sentences\tPrecision\tRecall\tF-1\tTP\tFP\tFN\tAVG-NSE\tEstimated-Coverage\tSigma");

        Map<Integer, HashMap> map = new TreeMap<>(evaluation_result);

        // AVG-NSE values should be shifted one row above and the last row should be 0
        // This is because we don't want to run NSE again after training the model, we just take the value when we
        // calculate it in the next iteration!
        for (int number_of_sentences : map.keySet()) {

            double sigma = 1-evaluation_result.get(number_of_sentences).get("AVG-NSE");

            if (Parameters.auto_annotation_se_margin > .1)
                sigma -= Parameters.auto_annotation_se_margin-sigma;

            output_lines.add( number_of_sentences + "\t" +
                    evaluation_result.get(number_of_sentences).get("p") + "\t" +
                    evaluation_result.get(number_of_sentences).get("r") + "\t" +
                    evaluation_result.get(number_of_sentences).get("f") + "\t" +
                    evaluation_result.get(number_of_sentences).get("tp") + "\t" +
                    evaluation_result.get(number_of_sentences).get("fp") + "\t" +
                    evaluation_result.get(number_of_sentences).get("fn")
            );
        }

        try {
            Files.write(Paths.get("Output/" + Parameters.dataset_name + "_AR"), output_lines, UTF_8, TRUNCATE_EXISTING, CREATE);
        }
        catch(Exception e){
            System.out.println("Cannot write to file");
            System.exit(0);
        }

        return evaluation_result;
    }
}