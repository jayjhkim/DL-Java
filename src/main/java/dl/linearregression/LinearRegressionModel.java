package dl.linearregression;

public class LinearRegressionModel {
    private double W;
    private double b;

    public LinearRegressionModel(double W, double b){
        this.W=W;
        this.b=b;
    }

    // linear 가설 공식
    public double hypo(double x){
        return W*x;
    }

    // cost function 공식
    public double cost(double[] xData, double[] yData){
        double sum=0.0;
        int m=xData.length;

        for(int i=1; i<=m; i++){
            double hypothesis=hypo(xData[i]);
            double error=hypothesis-yData[i];
            sum+=error*error;
        }

        return sum/(2*m);
    }

    // gradient 계산
    public double gradient(double[] xData, double[] yData){
        int m=xData.length;
        double sum=0.0;

        for(int i=1; i<=m; i++){
            double error = hypo(xData[i]) - yData[i];
            sum += error * xData[i];
        }

        return sum/m;
    }

    public void train(double[] xData, double[] yData, double learningRate){
        int operationCnt=0;

        // cost 구하기
        while (true){
            operationCnt++;
            double prevCost=cost(xData, yData);
            double prevW=W;

            double grad=gradient(xData, yData);
            W=W-learningRate*grad;

            double newCost = cost(xData, yData);

            if(operationCnt % 1000 == 1) {
                System.out.println("op " + operationCnt + ", W = " + W + ", cost = " + newCost);
            }

            // cost가 더 이상 줄지 않으면 멈춤
            if(prevCost <= newCost) {
                W = prevW;  // 원위치
                break;
            }
        }

        System.out.println("Result - W: " + W + ", cost: " + cost(xData, yData));
    }
}
