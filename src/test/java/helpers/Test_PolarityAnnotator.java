package helpers;

public class Test_PolarityAnnotator {

    public static void main(String[] args){
        String txt="There were no problems found.";
        txt="D-mib and Neur have overlapping rather than identical molecular activities.";
        txt="There isn't any problem.";

        PolarityAnnoation nd = new PolarityAnnoation();
        System.out.println(nd.get_polarityAnnotation(txt, true));
    }
}
