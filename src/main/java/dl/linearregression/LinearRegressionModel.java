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
        return W*x+b;
    }

    // cost function 공식
    public double cost(double[] xData, double[] yData){
        double sum=0.0;

        for(int i=1; i<xData.length; i++) {
            double hypothesis = hypo(xData[i]);
            double error = hypothesis - yData[i];
            sum += error * error;
        }

        return sum/xData.length;
    }

    public void train(double[] xData, double[] yData){

        while(true){
            double currentCost=cost(xData, yData);
            double step=currentCost*0.1;  // cost에 비례하여 변수의 값의 조정 비율을 결정

            if (step < 0.000001) break;  // 충분히 작은 경우 멈춤

            W += step;  // 증가시켜보기
            if (cost(xData, yData) < currentCost) continue;
            W -= step * 2;  // 감소시켜보기
            if (cost(xData, yData) < currentCost) continue;
            W += step;  // 증감 모두 별로면 원래대로 두기

            currentCost = cost(xData, yData);

            b += step;  // 증가시켜보기
            if (cost(xData, yData) < currentCost) continue;
            b -= step * 2;  // 감소시켜보기
            if (cost(xData, yData) < currentCost) continue;
            b += step;  // 증감 모두 별로면 원래대로 두기

            break;
        }

        System.out.println("W: " + W + ", b: " + b + ", cost: " + cost(xData, yData));
    }
}
