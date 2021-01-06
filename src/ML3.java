import java.util.*;
import java.io.*;

public class ML3 {

	public static void main(String[] args) throws Exception {
		ArrayList<ArrayList<classificationSample>> brca = new ArrayList<ArrayList<classificationSample>>();
		ArrayList<ArrayList<classificationSample>> cars = new ArrayList<ArrayList<classificationSample>>();		
		ArrayList<ArrayList<classificationSample>> images = new ArrayList<ArrayList<classificationSample>>();
		ArrayList<ArrayList<regressionSample>> abalone = new ArrayList<ArrayList<regressionSample>>();		
		ArrayList<ArrayList<regressionSample>> machines = new ArrayList<ArrayList<regressionSample>>();
		ArrayList<ArrayList<regressionSample>> fires = new ArrayList<ArrayList<regressionSample>>();
		for(String file:args) {
			switch(file) {
			case "breast-cancer-wisconsin.data":
				brca = partitionData.partitionClassificationData(readData.readClassificationData(file));
				fiveFoldClassify(brca, new int[][]{{}},true,file);
				break;
			case "car.data":
				cars = partitionData.partitionClassificationData(readData.readClassificationData(file));
				fiveFoldClassify(cars, new int[][]{ {0,1,2,3,4,5} , {4,4,4,3,3,3} },false,file);
				break;
			case "segmentation.data":
				images = partitionData.partitionClassificationData(readData.readClassificationData(file));
				fiveFoldClassify(images, new int[][]{{}},false,file);
				break;
			case "abalone.data":
				abalone = partitionData.partitionRegressionData(readData.readRegressionData(file));
				fiveFoldRegress(abalone, new int[][]{ {0} , {3} }, 1.5,false,file);
				break;
			case "machine.data":
				machines = partitionData.partitionRegressionData(readData.readRegressionData(file));
				fiveFoldRegress(machines, new int[][]{ {0} , {30} }, 500,false,file);
				break;
			case "forestfires.csv":
				fires = partitionData.partitionRegressionData(readData.readRegressionData(file));
				fiveFoldRegress(fires, new int[][]{{2,3},{12,7}}, 1.85,true,file);
				break;
			}
		}
	}
	
	/*
	 * Driver method to classify an input dataset with five-fold classification
	 * Most input variables are self-explanatory, the categorical features 2d array must include two arrays: the first is the index of any categorical variables and the second is the
	 * number of categories possible (this can't be simply inferred in the case that a categorical feature is used late in a tree where few samples remain)
	 */
	public static void fiveFoldClassify(ArrayList<ArrayList<classificationSample>> samples, int[][] categoricalFeatures, boolean printOutput, String fileName) throws IOException {
		BufferedWriter buffWriter = null;
		try{
			buffWriter = new BufferedWriter(new FileWriter(fileName + ".out.txt"));
			double[] prePrune = new double[5];
			double[] postPrune = new double[5];

			//iterate for all 5 folds
			for (int holdOut=0;holdOut<5;holdOut++) {
				ArrayList<classificationSample> trainingData = new ArrayList<classificationSample>();
				for(int trainingFold=0;trainingFold<5;trainingFold++) {
					if(trainingFold != holdOut) {
						for(classificationSample sample: samples.get(trainingFold)) {
							trainingData.add(sample);
						}
					}
				}
				boolean printOutput2=false;
				if(printOutput && holdOut==0) {
					printOutput2=true;
				}
				//test without pruning, print outputs on the first fold
				decisionTree classTree = new decisionTree();
				classTree.buildClassTree(trainingData,categoricalFeatures,printOutput2);
				if(printOutput2) {decisionTree.printClassTree(classTree.classTreeHead,"root","");}
				String[] resultsNoPrune = classTree.classifySamples(samples.get(holdOut), printOutput2);
				prePrune[holdOut]=calc01Loss(resultsNoPrune,samples.get(holdOut));
				if(printOutput2) {
					for(int index=0;index<resultsNoPrune.length;index++) {
						System.out.println("PREDICTED: " + resultsNoPrune[index] + ", ACTUAL: " + samples.get(holdOut).get(index).classification);
					}
				}
				
				//prune and test, print ouputs on the first fold
				classTree.recursivePrune(classTree.classTreeHead,samples.get(5),printOutput2);
				if(printOutput2) {decisionTree.printClassTree(classTree.classTreeHead,"root","");}
				String[] resultsWithPrune = classTree.classifySamples(samples.get(holdOut),printOutput2);
				postPrune[holdOut]=calc01Loss(resultsWithPrune,samples.get(holdOut));
				if(printOutput2) {
					for(int index=0;index<resultsWithPrune.length;index++) {
						System.out.println("PREDICTED: " + resultsWithPrune[index] + ", ACTUAL: " + samples.get(holdOut).get(index).classification);
					}
				}
				
				buffWriter.write("--------------HOLD OUT SET: " + (holdOut+1) + "---------------\n");
				buffWriter.write("Error % without pruning: " + (prePrune[holdOut]*100) + "\nError % with pruning: " + (postPrune[holdOut]*100) + "\n\n");
			}
			double preError = 0;
			double postError = 0;
			for(int index=0;index<5;index++) {
				preError += prePrune[index]/5.0;
				postError += postPrune[index]/5.0;
			}
			buffWriter.write("--------------AVERAGE PERFORMANCE---------------\n");
			buffWriter.write("Error % without pruning: " + (preError*100) + "\nError % with pruning: " + (postError*100));
		}
		finally {
			if(buffWriter!=null) {buffWriter.close();}
		}
	}
	
	/*
	 * Driver method to classify an input dataset with five-fold classification
	 * Most input variables are self-explanatory, the categorical features 2d array must include two arrays: the first is the index of any categorical variables and the second is the
	 * number of categories possible (this can't be simply inferred in the case that a categorical feature is used late in a tree where few samples remain)
	 */
	public static void fiveFoldRegress(ArrayList<ArrayList<regressionSample>> samples, int[][] categoricalFeatures, double stoppingPoint, boolean printOutput, String fileName) throws IOException {
		BufferedWriter buffWriter = null;
		try{
			buffWriter = new BufferedWriter(new FileWriter(fileName + ".out.txt"));
			double[] withoutStop = new double[5];
			double[] withStop = new double[5];
			
			//iterate for all 5 folds
			for (int holdOut=0;holdOut<5;holdOut++) {
				ArrayList<regressionSample> trainingData = new ArrayList<regressionSample>();
				for(int trainingFold=0;trainingFold<5;trainingFold++) {
					if(trainingFold != holdOut) {
						for(regressionSample sample: samples.get(trainingFold)) {
							trainingData.add(sample);
						}
					}
				}
				boolean printOutput2=false;
				if(printOutput && holdOut==0) {
					printOutput2=true;
				}
				//test without early stop
				decisionTree regressionTree = new decisionTree();
				regressionTree.buildRegressTree(trainingData,categoricalFeatures,0.0,printOutput2);
				if(printOutput2) {decisionTree.printRegressTree(regressionTree.regressTreeHead,"root","");}
				double[] valuesNoStop = regressionTree.regressSamples(samples.get(holdOut),printOutput2);
				withoutStop[holdOut]=calcMSE(valuesNoStop,samples.get(holdOut));
				if(printOutput2) {
					for(int index=0;index<valuesNoStop.length;index++) {
						System.out.println("PREDICTED: " + valuesNoStop[index] + ", ACTUAL: " + samples.get(holdOut).get(index).value);
					}
				}
				
				//test with early stop
				regressionTree.buildRegressTree(trainingData,categoricalFeatures,stoppingPoint,printOutput2);
				if(printOutput2) {decisionTree.printRegressTree(regressionTree.regressTreeHead,"root","");}
				double[] valuesWithStop = regressionTree.regressSamples(samples.get(holdOut),printOutput2);
				withStop[holdOut]=calcMSE(valuesWithStop,samples.get(holdOut));
				if(printOutput2) {
					for(int index=0;index<valuesWithStop.length;index++) {
						System.out.println("PREDICTED: " + valuesWithStop[index] + ", ACTUAL: " + samples.get(holdOut).get(index).value);
					}
				}
				
				buffWriter.write("--------------HOLD OUT SET: " + (holdOut+1) + "---------------\n");
				buffWriter.write("MSE without early stop: " + withoutStop[holdOut] + "\nMSE with early stop: " + withStop[holdOut] + "\n\n");
			}
			double nostopError = 0;
			double stopError = 0;
			for(int index=0;index<5;index++) {
				nostopError += withoutStop[index]/5.0;
				stopError += withStop[index]/5.0;
			}
			buffWriter.write("--------------AVERAGE PERFORMANCE---------------\n");
			buffWriter.write("Error % without early stop: " + (nostopError) + "\nError % with early stop: " + (stopError));
		}
		finally {
			if(buffWriter!=null) {buffWriter.close();}
		}
	}
	
	/*
	 * Method to calculate mean squared error between predictions and real data
	 */
	public static double calc01Loss(String[] predictions, ArrayList<classificationSample> actual) {
		int errors = 0;
		for(int index=0;index<predictions.length;index++) {
			if(!predictions[index].equals(actual.get(index).classification)) {
				errors++;
			}
		}
		return (double) errors/predictions.length;
	}
	
	/*
	 * Method to calculate mean squared error between predictions and real data
	 */
	public static double calcMSE(double[] predictions, ArrayList<regressionSample> actual) {
		double squaredError = 0;
		for(int index=0;index<predictions.length;index++) {
			squaredError += (predictions[index]-actual.get(index).value)*(predictions[index]-actual.get(index).value);
		}
		return (double) squaredError/predictions.length;
	}
}