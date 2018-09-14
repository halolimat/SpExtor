import edu.stanford.nlp.util.ArraySet;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class Test_SpExtor {

    public static double ZEROMARGIN = 0.;
    public static double FAST = 0.1;
    public static double HYPERFAST = 0.15;
    public static double ULTRAFAST = 0.20;

    static String training_data;
    static String data_map;
    static String seed_entity;
    static int k; // the number of candidate noun phrases from ESE
    static int batch_size;
    static double auto_annotation_se_margin;
    static int sample_size = 0; // size of the sample from the oroginal pool of sentences

    private static void start(){

        Dataset ds;

        // if no size was specified then read all the sentences from the pool
        if (sample_size == 0){
            ds = new Dataset(Test_SpExtor.class.getResource(training_data).getFile(), data_map);
        } else{
            ds = new Dataset(Test_SpExtor.class.getResource(training_data).getFile(), data_map, sample_size);
        }

        // Extract Noun Phrases and featurize them for the Entity Set Expansion method
        FeatureFactory ff = new FeatureFactory();
        ff.featurize(ds.getRaw_train_sentences());
        System.out.println("Done Featurizing !!!");

        Set<String> seed_set = new ArraySet<>();
        seed_set.add(seed_entity);

        EntitySetExpansion ese = new EntitySetExpansion();
        List<Map.Entry<String, Double>> entity_set = ese.expand_set(ff, seed_set).subList(0, k);
        // Filter out noun phrases which are not entities - you can get only the top 30 as in the paper.
        entity_set = entity_set.stream().filter(e -> ds.getTrain_entities().contains(e.getKey())).collect(Collectors.toList());
        for (Map.Entry<String, Double> entity: entity_set){
            seed_set.add(entity.getKey());
        }

        Core core = new Core();

        // This returns a final CRF model. You can serialize it to your local desk!
        core.start_active_learning( ds.getConll_train_sentences(),
                                    ds.getRaw_train_sentences(),
                                    data_map,
                                    seed_set,
                                    ff,
                                    batch_size,
                                    auto_annotation_se_margin,
                                    seed_entity);
    }

    public static void main(String args[]) throws IOException {

        training_data = "CoNLL_2003_LOC.Train";
        data_map = "word=0,pos=1,chunk=2,answer=3";
        seed_entity = "Germany";
        k = 30;
        sample_size = 100;
        auto_annotation_se_margin = ZEROMARGIN;
        batch_size = 500;

        start();
    }
}