public class Parameters {

    static String[] datasets_names = new String[]{  "bc2gm",
            //"ATIS-Airline",
            //"ATIS-LOC",
            "CoNLL-2003-LOC",
            "CoNLL-2003-PER",
            //"CoNLL-2003-MISC",
            "GENIA-3.02-protein_molecule",
            "GENIA_3.02_G#cell_type",
            "GENIA-3.02-virus"};
    //"GMB-2.2.0-eve"};
    //"GMB-2.2.0-geo"};

    public static String data_dir = System.getProperty("user.home") + "/work/NERData";

    public static String dataset_name;// = datasets_names[2];

    public static double auto_annotation_se_margin = 0;

    public static boolean use_proportional = false;

    public static boolean USE_ACTIVE_LEARNING = true;

    public static boolean USE_NSE = true;
    public static boolean USE_NSE_and_Features = false;

    public static int ese_candidates_number = 30;

    public static int n_in_NSE = 3;
    public static boolean use_ENSE = false;

    public static double BETA = 1;

    public static int sample_size = 100;

    public static boolean stop_on_100F = false;

    // 1
    public static boolean USE_ESE = true; // true always to test ESE only
    public static boolean USE_ESE_PIPELINE = true; // true always to test ESE only

    // 2 >>>
    public static boolean USE_FEATURES_ENSEMBLE = true;

    // false means use the graph embedding technique
    public static boolean atomic_increment_ranking = false; // false means use feature embedding
    // use the weights of a feature as the counts // atomic increment will then aggregate all counts
    public static boolean atomic_increment_ranking_useWeights = false;

    // ---------------------------------------------------- if atomic_increment_ranking = false

    // [0-2] > 0: raw count , 1: tfidf , 2: tfidf_sum
    public static int np_features_vector_weights = 1;

    public static boolean use_cosine_sim = false; // if false > use context_dependent_similarity
}