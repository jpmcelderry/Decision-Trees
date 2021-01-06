import java.util.*;

public class regressTreeNode {

	ArrayList<regressionSample> samples;
	double mean;
	int feature;
	boolean isCategorical;
	double midpoint;
	regressTreeNode[] children = new regressTreeNode[0];
	
	regressTreeNode(ArrayList<regressionSample> samples){
		this.samples = samples;
		this.mean = calcMean();
	}
	
	regressTreeNode(double mean){
		this.mean=mean;
		this.samples = new ArrayList<regressionSample>();
	}
	
	public double calcMean() {
		double sum = 0;
		for(regressionSample sample:this.samples) {
			sum += sample.value;
		}
		return (double) sum/samples.size();
	}
}
