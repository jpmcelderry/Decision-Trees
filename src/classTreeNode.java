import java.util.*;

public class classTreeNode {

	ArrayList<classificationSample> samples;
	String classPrediction;
	int feature;
	boolean isCategorical;
	double midpoint;
	classTreeNode[] children = new classTreeNode[0];
	
	classTreeNode(ArrayList<classificationSample> samples){
		this.samples = samples;
		predictClass();
	}
	
	classTreeNode(String classPrediction){
		this.classPrediction=classPrediction;
		this.samples = new ArrayList<classificationSample>();
	}
	
	//method to predict what the predicted class for this node should be
	public void predictClass(){
		HashMap<String,Integer> classCounts = new HashMap<String,Integer>();
		for(classificationSample sample: this.samples) {
			String classs = sample.classification;
			if(classCounts.get(classs)==null) {
				classCounts.put(classs,1);
			}
			else {
				classCounts.put(classs,classCounts.get(classs)+1);
			}
		}

		//Iterate over the map to find the plurality vote
		for(Map.Entry<String,Integer> entry: classCounts.entrySet()) {
			if (this.classPrediction==null || entry.getValue()>classCounts.get(classPrediction)) {
				this.classPrediction=entry.getKey();
			}
		}
	}
}
