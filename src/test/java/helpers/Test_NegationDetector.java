package helpers;

public class Test_NegationDetector {

    public static void main(String[] args){
        String txt="There were no problems found.";
        txt="D-mib and Neur have overlapping rather than identical molecular activities.";
        txt="There isn't any problem.";

        NegationDetection nd = new NegationDetection();
        System.out.println(nd.get_negations(txt, true));
    }
}
