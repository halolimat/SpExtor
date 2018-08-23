import com.google.common.util.concurrent.AtomicLongMap;
import edu.stanford.nlp.util.ArraySet;

import java.io.BufferedReader;
import java.io.FileReader;
import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.Collectors;

public class Datasets {

    public String root_dir = "";
    public List<String> meta = new ArrayList<>();

    public HashMap<Integer, String> conll_train_sentences = new HashMap<>();
    public HashMap<Integer, String> conll_test_sentences = new HashMap<>();

    public HashMap<Integer, String> raw_train_sentences = new HashMap<>();
    public HashMap<Integer, String> raw_test_sentences = new HashMap<>();

    public Set<String> train_entities = new ArraySet<>();
    public AtomicLongMap train_entities_counts = AtomicLongMap.create();

    public HashMap<String, Set<Integer>> test_entity_sentence_map = new HashMap<>();
    public Set<Integer> test_sentences_with_entities = new HashSet<>();

    public HashMap<Integer, Set<String>> train_sentence_entities = new HashMap<>();
    public Set<Integer> train_sentences_with_no_entities = new HashSet<>();

    public Set<String> search_entities = new ArraySet<>();

    // #################################################################################################################

    private String get_raw_text_from_conll(List<String> file_text_lines){

        StringBuilder sb = new StringBuilder();
        for(String line: file_text_lines){
            sb.append(line.split(" ")[0]+ " ");
        }

        String raw_text = sb.toString().trim();

        // Remove space before punctuations
        //raw_text = raw_text.replaceAll("\\s+(?=\\p{Punct})", "");

        return raw_text;
    }

    private String get_file_text(String absPath, boolean raw) {

        try {
            BufferedReader reader = new BufferedReader(new FileReader(absPath));

            List<String> file_text_lines = reader.lines().collect(Collectors.toList());

            if (raw){
                return get_raw_text_from_conll(file_text_lines);
            }

            else{
                String file_text = String.join("\n", file_text_lines);
                return file_text;
            }

        } catch (Exception e) {
            System.out.println(e.getStackTrace());
            return "";
        }
    }

    public void CoNLL_file_to_sentences(String abspath, boolean train) {

        String conll = get_file_text(abspath, false);
        String raw = get_file_text(abspath, true);

        // Tokenize to sentences
        String[] cs = conll.split("\n\n");
        String[] rs = raw.split("  ");

        assert cs.length == rs.length;

        if(train)
            for(int i = 0 ; i < cs.length ; i++){
                conll_train_sentences.put(i, cs[i]);
                raw_train_sentences.put(i, rs[i]);
            }
        else
            for(int i = 0 ; i < cs.length ; i++){
                conll_test_sentences.put(i, cs[i]);
                raw_test_sentences.put(i, rs[i]);
            }
    }

    // TODO: this is not correctly getting all sentences with entities from CoNLL-2003-LOC getting only 977 from 1000
    private void get_entities_from_train_conll() {

        for (int sentence_id : conll_train_sentences.keySet()) {
            String conll_sentence = conll_train_sentences.get(sentence_id);

            StringJoiner entity = new StringJoiner(" ");

            for (String line : conll_sentence.split("\n")) {

                String word = line.split(" ")[0];

                int eclass_index = Integer.parseInt(meta.get(2).substring(meta.get(2).length() - 1));

                String eclass = "";
                try{
                    eclass = line.split(" ")[eclass_index];
                }

                catch (Exception e){
                    System.out.println("####");
                    System.out.println(conll_sentence);
                    System.out.println(sentence_id);
                    System.out.println("####");
                    //System.exit(0);
                }


                if (!eclass.equals("O")) {
                    entity.add(word);
                } else if (!entity.toString().equals("")) {
                    train_entities.add(entity.toString());
                    train_entities_counts.incrementAndGet(entity.toString());

                    if (!train_sentence_entities.keySet().contains(sentence_id)) {
                        train_sentence_entities.put(sentence_id, new HashSet<>());
                    }

                    train_sentence_entities.get(sentence_id).add(entity.toString());

                    entity = new StringJoiner(" ");
                }
            }

//            if(raw_sentences.get(sentence_id).contains("United States")){
//                System.out.println("----");
//                System.out.println(entities);
//                System.exit(0);
//            }

            if (!train_sentence_entities.keySet().contains(sentence_id))
                train_sentences_with_no_entities.add(sentence_id);

        }
    }

    private void get_entities_from_test_conll() {

        for (int sentence_id : conll_test_sentences.keySet()) {
            String conll_sentence = conll_test_sentences.get(sentence_id);

            StringJoiner entity = new StringJoiner(" ");

            for (String line : conll_sentence.split("\n")) {

                if(line.equals(""))
                    continue;

                String word = line.split(" ")[0];

                int eclass_index = Integer.parseInt(meta.get(2).substring(meta.get(2).length() - 1));

                String eclass = "";
                try{
                    eclass = line.split(" ")[eclass_index];
                }

                catch (Exception e){
                    System.out.println("####");
                    System.out.println(conll_sentence);
                    System.out.println("####");
                    System.exit(0);
                }


                if (!eclass.equals("O")) {
                    entity.add(word);
                } else if (!entity.toString().equals("")) {

                    if(!test_entity_sentence_map.containsKey(entity.toString())){
                        test_entity_sentence_map.put(entity.toString(), new ArraySet<>());
                    }

                    test_entity_sentence_map.get(entity.toString()).add(sentence_id);

                    test_sentences_with_entities.add(sentence_id);

                    entity = new StringJoiner(" ");
                }
            }

        }
    }

    // #################################################################################################################

    public Datasets(String root_dir, String ds_name) {

        this.root_dir = root_dir;

        switch (ds_name) {

            /*
                CoNLL > LOC, MISC, ORG, PER
                GENIA > DNA, RNA, cell_line, cell_type

                ATIS_Airline.Train <
                # sentences > 3983
                # sentences with entities > 544 >> 0.13
                # sentences w/out entities > 3439

                ATIS_LOC.Train
                # sentences > 3983
                # sentences with entities > 3760 >> 0.94
                # sentences w/out entities > 223

                CoNLL_2003_LOC.Train <
                # sentences > 13519
                # sentences with entities > 4878 >> 0.36
                # sentences w/out entities > 8641

                CoNLL_2003_PER.Train <
                # sentences > 13519
                # sentences with entities > 4228 >> 0.31
                # sentences w/out entities > 9291

                GENIA_3.02_G#protein_molecule.Train
                # sentences > 14838
                # sentences with entities > 8284 >> 0.55
                # sentences w/out entities > 6554

                GENIA_3.02_G#virus.Train <
                # sentences > 14838
                # sentences with entities > 1210 >> 0.08
                # sentences w/out entities > 13628

                GMB-2.2.0-geo.Train
                # sentences > 41518
                # sentences with entities > 22567 >> 0.54
                # sentences w/out entities > 18951

                GMB-2.2.0-eve.Train <
                # sentences > 41518
                # sentences with entities > 294 >> 0.007
                # sentences w/out entities > 41224

                 */

            // >> NP extraction full > 0.5071353318647639
            // # sentences > 12500
            // # sentences with entities > 6381 >>> 0.51
            // # sentences w/out entities > 6119
            case "bc2gm":

                if (Parameters.use_proportional) {
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/bc2gm.Train",
                            root_dir + "/",
                            "word=0,answer=1"
                    ));
                }
                else{
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/bc2gm.Train",
                            root_dir + "/",
                            "word=0,answer=1"
                    ));
                }

                // 64
                search_entities.add("insulin");
                // 1
                search_entities.add("GrpE");

                break;

            // Entity	    P	    R	    F1	    TP	FP	FN
            // AirlineName	1.0000	0.9474	0.9730	36	0	2
            // >> NPs > 0.875
            case "ATIS-Airline":

                if (Parameters.use_proportional) {
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/ATIS_Airline_proportional.Train",
                            root_dir + "/ATIS_Airline_proportional.Test",
                            "word=0,answer=1"
                    ));
                }
                else{
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/ATIS_Airline.Train",
                            root_dir + "/ATIS_Airline.Test",
                            "word=0,answer=1"
                    ));
                }

                // 92
                search_entities.add("American Airlines");
                //search_entities.add("Us Air");
                //search_entities.add("Delta");
                // 7
                //search_entities.add("Eastern Airlines");
                // 3
                search_entities.add("Air Canada");

                break;

            // Entity	P	    R	    F1	    TP	    FP	FN
            // LOC	    0.9905	0.8739	0.9286	104	    1	15
            // >> NPs > 0.8909090909090909
            case "ATIS-LOC":

                if (Parameters.use_proportional) {
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/ATIS_LOC_proportional.Train",
                            root_dir + "/ATIS_LOC_proportional.Test",
                            "word=0,answer=1"
                    ));
                }
                else{
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/ATIS_LOC.Train",
                            root_dir + "/ATIS_LOC.Test",
                            "word=0,answer=1"
                    ));
                }

                // 19
                search_entities.add("San Francisco");
                // 1
                search_entities.add("Tacoma");

                //search_entities.add("Salt Lake City");
                //search_entities.add("Tacoma");
                //search_entities.add("New York");
                //search_entities.add("Denver Airport");
                //search_entities.add("St. Petersburg");

                break;


            // ALL > NP extraction accuracy = 0.84
            // Entity	P	    R	    F1	    TP	    FP	FN
            // LOC	    0.9645	0.8218	0.8874	272	    10	59
            // LOC+ff	0.9695	0.8640	0.9137	286	    9	45
            // >> NPs > 0.8445807770961146
            case "CoNLL-2003-LOC":

                if (Parameters.use_proportional) {
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/CoNLL_2003_LOC_proportional.Train",
                            root_dir + "/CoNLL_2003_LOC_proportional.TestA",
                            "word=0,pos=1,chunk=2,answer=3"));
                }
                else{
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/CoNLL_2003_LOC.Train",
                            root_dir + "/CoNLL_2003_LOC.TestA",
                            "word=0,pos=1,chunk=2,answer=3"));
                }

                // 296
                search_entities.add("U.S.");

                // 25
                //search_entities.add("Germany");

                //search_entities.add("Atlanta");
                //search_entities.add("South Dakota");
                //search_entities.add("Western Ukraine");

                // 5
                //search_entities.add("Ohio");

                // 26
                search_entities.add("Washington");

                break;


            // ALL NP extraction accuracy = 0.76
            // Entity	P	    R	    F1	    TP	    FP	FN
            // PER	    0.9607	0.8072	0.8772	293	    12	70
            // >> NPs > 0.739093242087254
            case "CoNLL-2003-PER":

                if (Parameters.use_proportional) {
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/CoNLL_2003_PER_proportional.Train",
                            root_dir + "/CoNLL_2003_PER_proportional.TestA",
                            "word=0,pos=1,chunk=2,answer=3"));
                }
                else{
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/CoNLL_2003_PER.Train",
                            root_dir + "/CoNLL_2003_PER.TestA",
                            "word=0,pos=1,chunk=2,answer=3"));
                }

                //search_entities.add("Bill Clinton");
                //search_entities.add("Boris Yeltsin");
                //search_entities.add("Tom Lehman");
                //search_entities.add("Mark Kennedy");
                //search_entities.add("Tom Daschle");

                // 70
                search_entities.add("Clinton");
                // 1
                search_entities.add("Von Heesen");

                break;


            // # Entities = 829
            // NPs extraction accuracy = 0.41857659831121835
            case "CoNLL-2003-MISC":

                if (Parameters.use_proportional) {}
                else{
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/CoNLL_2003_MISC.Train",
                            root_dir + "/CoNLL_2003_MISC.TestA",
                            "word=0,pos=1,chunk=2,answer=3"));
                }

                // >>>>>
                // 75
                search_entities.add("Russian");
                // 1
                search_entities.add("Jordanians");

                break;


            // # sentences with Entities = 8284
            // # Entities = 3413
            // NPs extraction accuracy = 0.5994726047465573

            // Entity	                    P	    R	    F1	    TP	    FP	    FN
            // G#protein_molecule	        0.7540	0.5791	0.6551	377	    123	    274
            // G#protein_molecule+CRF+ESE	0.8303	0.6989	0.7590	455	    93	    196
            // G#protein_molecule+CRF+ESE+features	0.8237	0.6820	0.7462	444	95	207
            // >> NPs > 0.6585365853658537
            case "GENIA-3.02-protein_molecule":
                if (Parameters.use_proportional) {
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/GENIA_3.02_G#protein_molecule_proportional.Train",
                            root_dir + "/GENIA_3.02_G#protein_molecule_proportional.Test",
                            "word=0,answer=1"));
                }
                else{
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/GENIA_3.02_G#protein_molecule.Train",
                            root_dir + "/GENIA_3.02_G#protein_molecule.Test",
                            "word=0,answer=1"));
                }

                //search_entities.add("BCL-6");
                //search_entities.add("CD40");
                //search_entities.add("tumour suppressor p53");
                //search_entities.add("TCR alpha");
                //search_entities.add("3H-TdR");

                // 552
                search_entities.add("NF-kappa B");
                // 58
                //search_entities.add("IL-2");
                // 34
                //search_entities.add("AP-1");
                // 1
                search_entities.add("IL-1RA");
                // 1
                //search_entities.add("cyclin D3");

                break;


            // # sentences with Entities = 4065
            // # Entities = 1569
            // NPs extraction accuracy = 0.24410452517527087
            case "GENIA_3.02_G#cell_type":
                if (Parameters.use_proportional) {

                }
                else{
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/GENIA_3.02_G#cell_type.Train",
                            root_dir + "/GENIA_3.02_G#cell_type.Test",
                            "word=0,answer=1"));
                }

                // 489
                search_entities.add("T cells");
                // 2
                search_entities.add("MGC");

                break;

            case "GENIA_3.02_RNA_family_or_group":
                if (Parameters.use_proportional) {

                }
                else{
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/GENIA_3.02_RNA_family_or_group.Train",
                            root_dir + "/GENIA_3.02_RNA_family_or_group.Test",
                            "word=0,answer=1"));
                }

                search_entities.add("mRNA");
                search_entities.add("EBNA-2");

                break;

            // # sentences with Entities = 1210
            // # Entities = 324
            // NPs extraction accuracy = 0.49382716049382713

            // Entity	        P	    R	    F1	    TP	FP	FN
            // G#virus	        0.7636	0.8400	0.8000	42	13	8
            // G#virus+CRF+ESE	0.8000	0.8800	0.8381	44	11	6
            // G#virus+ff   	0.8000	0.8800	0.8381	44	11	6
            // > NPs > 0.4948453608247423
            case "GENIA-3.02-virus":
                if (Parameters.use_proportional) {
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/GENIA_3.02_G#virus_proportional.Train",
                            root_dir + "/GENIA_3.02_G#virus_proportional.Test",
                            "word=0,answer=1"));
                }
                else{
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/GENIA_3.02_G#virus.Train",
                            root_dir + "/GENIA_3.02_G#virus.Test",
                            "word=0,answer=1"));
                }

                // 336
                search_entities.add("HIV-1");
                //search_entities.add("HTLV-I");
                // 89
                //search_entities.add("Epstein-Barr virus");
                // 36
                //search_entities.add("human immunodeficiency virus");
                // 28
                //search_entities.add("HIV-2");

                // 2
                //search_entities.add("murine leukemia virus");

                // 1
                search_entities.add("adenovirus E1A");

                break;


            // Entity	    P	    R	    F1	    TP	FP	FN
            // eve	        0.8103	0.6184	0.7015	47	11	29
            // eve+m	    0.9825	0.7368	0.8421	56	1	20
            // eve+ESE      0.8136	0.6316	0.7111	48	11	28
            // eve+m+ESE    0.9828	0.7500	0.8507	57	1	19
            // eve+ff   	1.0000	0.7500	0.8571	57	0	19
            // >> NPs > 0.6206896551724138
            case "GMB-2.2.0-eve":
                if (Parameters.use_proportional) {
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/GMB-2.2.0-eve_proportional.Train",
                            root_dir + "/GMB-2.2.0-eve_proportional.Test",
                            "word=0,answer=2"
                    ));
                }
                else{
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/GMB-2.2.0-eve.Train",
                            root_dir + "/GMB-2.2.0-eve.Test",
                            "word=0,answer=2"
                    ));
                }

                // 16
                search_entities.add("World War I");

                //search_entities.add("Summit of the Americas");
                //search_entities.add("Memorial Day");
                //search_entities.add("World Cup");
                //search_entities.add("Tropical Storm Tomas");

                // 1
                //search_entities.add("2012 Summer Olympics");

                // 5
                search_entities.add("World Cup");

                break;


            // Entity	    P	    R	    F1	    TP	    FP	    FN
            // geo	        0.8884	0.7576	0.8178	422	    53	    135
            // geo+CRF+ESE	0.9385	0.8761	0.9062	488	    32	    69
            // geo+ff   	0.9364	0.8725	0.9033	486	    33	    71
            // >> NPs > 0.722027972027972
            case "GMB-2.2.0-geo":
                if (Parameters.use_proportional) {
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/GMB-2.2.0-geo_proportional.Train",
                            root_dir + "/GMB-2.2.0-geo_proportional.Test",
                            "word=0,answer=2"
                    ));
                }
                else{
                    meta = new ArrayList<>(Arrays.asList(
                            root_dir + "/GMB-2.2.0-geo.Train",
                            root_dir + "/GMB-2.2.0-geo.Test",
                            "word=0,answer=2"
                    ));
                }

                // 51
                search_entities.add("United States");

                //search_entities.add("Germany");
                //search_entities.add("Atlanta");
                //search_entities.add("South Dakota");
                //search_entities.add("Western Ukraine");
                //search_entities.add("Ohio");

                // 1
                search_entities.add("Oxford");

                break;
        }

        // -------------------

        // read files and tokenize to sentences
        CoNLL_file_to_sentences(meta.get(0), true);
        //CoNLL_file_to_sentences(meta.get(1), false);

        get_entities_from_train_conll();
        //get_entities_from_test_conll();

        //sample_sentences_with_third_entity_class();
    }

    // #################################################################################################################

    public void test_entities_in_dataset(){
        System.out.println("# sentences with entities > " + train_sentence_entities.size());
        System.out.println("# sentences w/out entities > " + train_sentences_with_no_entities.size());

        ArrayList<Integer> lengths = new ArrayList<>();
        for(int sentence_id : raw_train_sentences.keySet()){
            lengths.add(raw_train_sentences.get(sentence_id).split(" ").length);
        }

        Double average = lengths.stream().mapToInt(val -> val).average().getAsDouble();

        System.out.println("Average # Tokens in Sentences > "+average);
        System.out.println();

        System.out.println(train_entities_counts);

        List<Map.Entry<Object, Long>> sorted =
                new ArrayList<>(train_entities_counts.asMap().entrySet());
        Collections.sort(sorted, Map.Entry.comparingByValue());

        int count = 0;
        for (Map.Entry<Object, Long> entry : sorted) {
            System.out.println(entry); // Or something more useful

            count++;

            if (count>10){
                break;
            }
        }

        sorted = new ArrayList<>(train_entities_counts.asMap().entrySet());
        Collections.sort(sorted, Collections.reverseOrder(Map.Entry.comparingByValue()));

        for (Map.Entry<Object, Long> entry : sorted) {
            System.out.println(entry); // Or something more useful

            count--;

            if (count==0){
                break;
            }
        }
    }
}