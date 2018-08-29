/**------------------------------------------------------------------------
 *
 * Prepares the testing data to be used in the full process of the incremental learning procedure
 *
 * @author Hussein S. Al-Olimat github.com/halolimat
 *
 *------------------------------------------------------------------------
 *
 * This file is part of the Sparse Entity Extraction tool (SpExtor) which is the implementation of the proposed method
 * in our COLING 2018 publication titled: "A practical incremental learning framework for sparse entity extraction".
 * The full COLING 2018 paper can be found @ https://aclanthology.info/papers/C18-1059/c18-1059
 *
 * SpExtor is a free tool: you can redistribute it and/or modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * SpExtor is distributed WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with SpExtor.
 * If not, see <http://www.gnu.org/licenses/> and https://github.com/halolimat/SpExtor/blob/master/LICENSE.
 *
 *------------------------------------------------------------------------
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import java.util.stream.Collectors;

public class Dataset {

    private HashMap<Integer, String> conll_train_sentences;
    private HashMap<Integer, String> conll_test_sentences;

    private HashMap<Integer, String> raw_train_sentences;
    private HashMap<Integer, String> raw_test_sentences;

    private Set<String> train_entities;
    private HashMap<Integer, Set<String>> train_sentence_entities;

    // *****************************************************************************************************************
    // Private methods
    // *****************************************************************************************************************

    /**
     * Extract annotated entities from the CoNLL formatted annotated file.
     *
     * @param data_map: a CSV tuple of the semantics of each column in CoNLL formatted data.
     */
    private void extract_entities_from_train_conll(String data_map) {
        // Note: this is not correctly getting all sentences with entities from CoNLL-2003-LOC, getting only 977 from 1k

        HashMap<Integer, String> conll_train_sentences = getConll_train_sentences();
        HashMap<Integer, Set<String>> train_sentence_entities = new HashMap<>();
        Set<String> train_entities = new HashSet<>();

        for (int sentence_id : conll_train_sentences.keySet()) {
            String conll_sentence = conll_train_sentences.get(sentence_id);
            StringJoiner entity = new StringJoiner(" ");
            for (String line : conll_sentence.split("\n")) {
                String word = line.split(" ")[0];
                int eclass_index = Integer.parseInt(data_map.substring(data_map.length() - 1));
                String eclass = "";
                try{
                    eclass = line.split(" ")[eclass_index];
                } catch (Exception e){
                    System.out.println("####");
                    System.out.println(conll_sentence);
                    System.out.println(sentence_id);
                    System.out.println("####");
                    e.printStackTrace();
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

        setTrain_sentence_entities(train_sentence_entities);
        setTrain_entities(train_entities);
    }

    /**
     * Returns the raw text from the CoNLL formatted annotated file.
     *
     * @param file_text_lines: the path from which you want to read the dataset
     */
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

    /**
     * Returns the textual content of a CoNLL annotated dataset as raw or CoNLL formatted text
     *
     * @param absPath: the path from which you want to read the dataset
     * @param raw: boolean value to choose between returning a raw or CoNLL formatted string
     */
    private String get_file_text(String absPath, boolean raw) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(absPath));
            List<String> file_text_lines = reader.lines().collect(Collectors.toList());
            if (raw){
                return get_raw_text_from_conll(file_text_lines);
            } else{
                String file_text = String.join("\n", file_text_lines);
                return file_text;
            }
        } catch (Exception e) {
            System.out.println(e.getStackTrace());
            return "";
        }
    }

    // *****************************************************************************************************************
    // Public methods
    // *****************************************************************************************************************

    /**
     * Preparing test data
     *
     * @param abspath: the path from which you want to read the dataset
     * @param data_map: a CSV tuple of the semantics of each column in the CoNLL training data.
     */
    public Dataset(String abspath, String data_map) {

        String conll = get_file_text(abspath, false);
        String raw = get_file_text(abspath, true);

        // Tokenize to sentences
        String[] cs = conll.split("\n\n");
        String[] rs = raw.split("  ");

        assert cs.length == rs.length;

        HashMap<Integer, String> conll_train_sentences = new HashMap<>();
        HashMap<Integer, String> raw_train_sentences = new HashMap<>();
        for(int i = 0 ; i < cs.length ; i++){
            conll_train_sentences.put(i, cs[i]);
            raw_train_sentences.put(i, rs[i]);
        }
        setConll_train_sentences(conll_train_sentences);
        setRaw_train_sentences(raw_train_sentences);

        HashMap<Integer, String> conll_test_sentences = new HashMap<>();
        HashMap<Integer, String> raw_test_sentences = new HashMap<>();
        for(int i = 0 ; i < cs.length ; i++){
            conll_test_sentences.put(i, cs[i]);
            raw_test_sentences.put(i, rs[i]);
        }
        setConll_test_sentences(conll_test_sentences);
        setRaw_test_sentences(raw_test_sentences);

        extract_entities_from_train_conll(data_map);
    }

    /**
     * This is for testing to have a subset of the data only
     *
     * @param abspath: the path from which you want to read the dataset
     * @param data_map: a CSV tuple of the semantics of each column in the CoNLL training data.
     * @param size: the size of the pool of sentences
     */
    public Dataset(String abspath, String data_map, int size) {
        String conll = get_file_text(abspath, false);
        String raw = get_file_text(abspath, true);

        // Tokenize to sentences
        String[] cs = conll.split("\n\n");
        String[] rs = raw.split("  ");

        assert cs.length == rs.length;

        HashMap<Integer, String> conll_train_sentences = new HashMap<>();
        HashMap<Integer, String> raw_train_sentences = new HashMap<>();
        for(int i = 0 ; i < cs.length ; i++){
            conll_train_sentences.put(i, cs[i]);
            raw_train_sentences.put(i, rs[i]);

            if (i >= size-1)
                break;
        }
        setConll_train_sentences(conll_train_sentences);
        setRaw_train_sentences(raw_train_sentences);

        HashMap<Integer, String> conll_test_sentences = new HashMap<>();
        HashMap<Integer, String> raw_test_sentences = new HashMap<>();
        for(int i = 0 ; i < cs.length ; i++){
            conll_test_sentences.put(i, cs[i]);
            raw_test_sentences.put(i, rs[i]);

            if (i >= size-1)
                break;
        }
        setConll_test_sentences(conll_test_sentences);
        setRaw_test_sentences(raw_test_sentences);

        extract_entities_from_train_conll(data_map);
    }

    /**
     * Returns all train sentences from the train dataset CoNLL formatted
     *
     * @return map of sentence-id to sentence CoNLL text
     */
    public HashMap<Integer, String> getConll_train_sentences() {
        return conll_train_sentences;
    }

    public void setConll_train_sentences(HashMap<Integer, String> conll_train_sentences) {
        this.conll_train_sentences = conll_train_sentences;
    }

    /**
     * Returns all raw train sentences from the train dataset
     *
     * @return map of sentence-id to raw sentence text
     */
    public HashMap<Integer, String> getRaw_train_sentences() {
        return raw_train_sentences;
    }

    public void setRaw_train_sentences(HashMap<Integer, String> raw_train_sentences) {
        this.raw_train_sentences = raw_train_sentences;
    }

    /**
     * Returns all the annotated entities from the train sentences
     *
     * @return set of entities
     */
    public Set<String> getTrain_entities() {
        return train_entities;
    }

    public void setTrain_entities(Set<String> train_entities) {
        this.train_entities = train_entities;
    }

    /**
     * Returns all test sentences from the test dataset CoNLL formatted
     *
     * @return map of sentence-id to sentence CoNLL text
     */
    public HashMap<Integer, String> getConll_test_sentences() {
        return conll_test_sentences;
    }

    public void setConll_test_sentences(HashMap<Integer, String> conll_test_sentences) {
        this.conll_test_sentences = conll_test_sentences;
    }

    /**
     * Returns all raw test sentences from the test dataset
     *
     * @return map of sentence-id to raw sentence text
     */
    public HashMap<Integer, String> getRaw_test_sentences() {
        return raw_test_sentences;
    }

    public void setRaw_test_sentences(HashMap<Integer, String> raw_test_sentences) {
        this.raw_test_sentences = raw_test_sentences;
    }

    /**
     * Returns all the annotated entities for each sentence from the train dataset
     *
     * @return map of sentence-id to entities set in each sentence
     */
    public HashMap<Integer, Set<String>> getTrain_sentence_entities() {
        return train_sentence_entities;
    }

    public void setTrain_sentence_entities(HashMap<Integer, Set<String>> train_sentence_entities) {
        this.train_sentence_entities = train_sentence_entities;
    }
}