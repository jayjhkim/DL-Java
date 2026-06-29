package dl.linearregression;

public class Main {
    public static void main(String[] args) {
        double[] xData={1,2,3};
        double[] yData={1,2,3};

        LinearRegressionModel model=new LinearRegressionModel(0,0);
        model.train(xData,yData);
    }
}