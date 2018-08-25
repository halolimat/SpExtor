import edu.stanford.nlp.util.ArraySet;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import java.util.stream.Collectors;

public class Dataset {

    public HashMap<Integer, String> conll_train_sentences = new HashMap<>();
    public HashMap<Integer, String> conll_test_sentences = new HashMap<>();

    public HashMap<Integer, String> raw_train_sentences = new HashMap<>();
    public HashMap<Integer, String> raw_test_sentences = new HashMap<>();

    public Set<String> train_entities = new ArraySet<>();
    public HashMap<Integer, Set<String>> train_sentence_entities = new HashMap<>();

    private void get_entities_from_train_conll(String data_map) {
        // Note: this is not correctly getting all sentences with entities from CoNLL-2003-LOC getting only 977 from 1000

        for (int sentence_id : conll_train_sentences.keySet()) {
            String conll_sentence = conll_train_sentences.get(sentence_id);

            StringJoiner entity = new StringJoiner(" ");

            for (String line : conll_sentence.split("\n")) {

                String word = line.split(" ")[0];

                int eclass_index = Integer.parseInt(data_map.substring(data_map.length() - 1));

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

                    if (!train_sentence_entities.keySet().contains(sentence_id)) {
                        train_sentence_entities.put(sentence_id, new HashSet<>());
                    }

                    train_sentence_entities.get(sentence_id).add(entity.toString());

                    entity = new StringJoiner(" ");
                }
            }
        }
    }

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

    public Dataset(String abspath, String data_map) {

        String conll = get_file_text(abspath, false);
        String raw = get_file_text(abspath, true);

        // Tokenize to sentences
        String[] cs = conll.split("\n\n");
        String[] rs = raw.split("  ");

        assert cs.length == rs.length;

        for(int i = 0 ; i < cs.length ; i++){
            conll_train_sentences.put(i, cs[i]);
            raw_train_sentences.put(i, rs[i]);
        }

        for(int i = 0 ; i < cs.length ; i++){
            conll_test_sentences.put(i, cs[i]);
            raw_test_sentences.put(i, rs[i]);
        }

        get_entities_from_train_conll(data_map);
    }
}
