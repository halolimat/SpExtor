import edu.stanford.nlp.util.ArraySet;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class main {

    public static double ZEROMARGIN = 0.;
    public static double FAST = 0.1;
    public static double HYPERFAST = 0.15;
    public static double ULTRAFAST = 0.20;

    public static void main(String args[]) throws IOException {

        String training_data = "CoNLL_2003_LOC.Train";
        String data_map = "word=0,pos=1,chunk=2,answer=3";
        String seed_entity = "Germany";
        int k = 30; // the number of candidate noun phrases from ESE
        int sample_size = 100; // the batch size
        double auto_annotation_se_margin = ZEROMARGIN;

        Dataset ds = new Dataset(main.class.getResource(training_data).getFile(), data_map);

        EntitySetExpansion ese = new EntitySetExpansion();

        FeatureFactory ff = new FeatureFactory();
        ff.featurize(ds.raw_train_sentences);
        System.out.println("Done Featurizing !!!");

        // TODO: make the starting point work with a set
        Set<String> seed_set = new ArraySet<>();
        seed_set.add(seed_entity);

        List<Map.Entry<String, Double>> entity_set = ese.expand_set(ff, seed_set).subList(0, k);
        // Filter out noun phrases which are not entities - you can get only the top 30 as in the paper.
        entity_set = entity_set.stream().filter(e -> ds.train_entities.contains(e.getKey())).collect(Collectors.toList());
        for (Map.Entry<String, Double> entity: entity_set){
            seed_set.add(entity.getKey());
        }

        EntityLearning el = new EntityLearning();
        el.start_active_learning(   ds.conll_train_sentences,
                                    ds.raw_train_sentences,
                                    data_map,
                                    seed_set,
                                    ff,
                                    sample_size,
                                    auto_annotation_se_margin);
    }
}