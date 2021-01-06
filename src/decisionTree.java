import java.util.*;
import java.io.*;

public class decisionTree {

	regressTreeNode regressTreeHead;
	classTreeNode classTreeHead;
	
	/*
	 * Driver method for building an ID3 tree, no pruning
	 */
	public void buildClassTree(ArrayList<classificationSample> trainingSet, int[][] categoricalFeatures, boolean printOutput) {
		this.classTreeHead = null;
		this.classTreeHead = new classTreeNode(trainingSet);
		ArrayList<Integer> features = new ArrayList<Integer>();
		for(int index=0;index<trainingSet.get(0).features.length;index++) {
			features.add(index);
		}
		recursiveClassTreeBuild(this.classTreeHead,features,categoricalFeatures, printOutput);
	}

	/*
	 * Recursive Method to build an ID3 tree
	 */
	public void recursiveClassTreeBuild(classTreeNode head, ArrayList<Integer> features, int[][] categoricalFeatures, boolean printOutput) {
		//stop when all samples match in class or there's no features left
		if(features.size()==0 || head.samples.size()<=1 || calcEntropy(head.samples)==0) {
			return;
		}
		//calculate initial entropy
		double currentMax=-1;
		double initEntropy = calcEntropy(head.samples);
		
		//Iterate by feature
		for (Integer feature: features) {
			//Sort on feature being considered
			Collections.sort(head.samples, new Comparator<classificationSample>() {
				public int compare(classificationSample row1, classificationSample row2) {
					return Double.compare(row1.features[feature],row2.features[feature]);
				}
			});
			//Check if feature is categorical
			int temp = arrayContains(categoricalFeatures[0],feature);
			//Logic for splitting on a categorical feature
			if(temp>-1) {
				//create a child for each category value
				ArrayList<classificationSample>[] children = (ArrayList<classificationSample>[]) new ArrayList[categoricalFeatures[1][temp]];
				for(classificationSample sample: head.samples) {
					if(children[(int)sample.features[feature]]==null) {
						children[(int)sample.features[feature]] = new ArrayList<classificationSample>();
					}
					children[(int)sample.features[feature]].add(sample);
				}
				//calculate gain ratio
				double entropy = 0;
				for(int index=0;index<children.length;index++) {
					if(children[index]==null) {
						children[index] = new ArrayList<classificationSample>();
					}
					else {
						ArrayList<classificationSample> child = children[index]; 
						entropy += ((double) calcEntropy(child)*child.size())/head.samples.size();
					}
				}
				double gainRatio = calcGainRatio(entropy, initEntropy, head.samples.size(),children);
				//save split if this is a new best
				if(gainRatio>currentMax) {
					if(printOutput) {System.out.println("Info Gain=" + (initEntropy-entropy) + ", Gain Ratio=" + gainRatio + ", previous max=" + currentMax);}
					head.feature=feature;
					head.isCategorical=true;
					head.children = new classTreeNode[children.length];
					for(int index=0;index<children.length;index++) {
						if(children[index].size()==0) {
							head.children[index]=new classTreeNode(head.classPrediction);
						}
						else {
							head.children[index]=new classTreeNode(children[index]);
						}
					}
					currentMax=gainRatio;
				}
			}
			//Logic for splitting on a continuous feature
			else {
				for(int index=1;index<head.samples.size();index++) {
					//only consider splits after class changes
					classificationSample current=head.samples.get(index);
					classificationSample previous=head.samples.get(index-1);
					if(!current.classification.equals(previous.classification)) {
						//split along the midpoint
						double midpoint = (previous.features[feature] + current.features[feature])/2.0;
						ArrayList<classificationSample> lessThan = new ArrayList<classificationSample>();
						ArrayList<classificationSample> greaterThan = new ArrayList<classificationSample>();
						for(classificationSample sample: head.samples) {
							if(sample.features[feature]<midpoint) {
								lessThan.add(sample);
							}
							else {
								greaterThan.add(sample);
							}
						}
						//calc gain ratio
						double entropy = ((double)lessThan.size()*calcEntropy(lessThan))/head.samples.size() + ((double)greaterThan.size()*calcEntropy(greaterThan))/head.samples.size();
						ArrayList<classificationSample>[] children = (ArrayList<classificationSample>[]) new ArrayList[2];
						children[0] = lessThan;
						children[1] = greaterThan;
						double gainRatio = calcGainRatio(entropy, initEntropy, head.samples.size(),children);
						//if best split, save it
						if(gainRatio>currentMax && lessThan.size()>0 && greaterThan.size()>0) {
							if(printOutput) {System.out.println("Info Gain=" + (initEntropy-entropy) + ", Gain Ratio=" + gainRatio + ", previousMax=" + currentMax);}
							head.feature=feature;
							head.isCategorical=false;
							head.midpoint=midpoint;
							head.children = new classTreeNode[2];
							head.children[0] = new classTreeNode(lessThan);
							head.children[1] = new classTreeNode(greaterThan);
							currentMax=gainRatio;
						}
					}
				}
			}
		}
		//recurse
		ArrayList<Integer> updatedFeatures = new ArrayList<Integer>(features);
		updatedFeatures.remove(Integer.valueOf(head.feature));
		for(classTreeNode subtree: head.children) {
			recursiveClassTreeBuild(subtree,updatedFeatures,categoricalFeatures,printOutput);
		}
	}
	
	/*
	 * Method for pruning an ID3 tree
	 */
	public void recursivePrune(classTreeNode head, ArrayList<classificationSample> verificationSet, boolean printOutput) {
		//base case, leaves have no subtrees
		if(head.children.length==0) {
			return;
		}
		
		//depth-first recursion
		for(classTreeNode subtree: head.children) {
			recursivePrune(subtree,verificationSet, printOutput);
		}
		
		//check error before pruning
		int prePruneError = 0;
		String[] classPredictions = classifySamples(verificationSet, false);
		for(int index=0;index<classPredictions.length;index++) {
			if(!classPredictions[index].equals(verificationSet.get(index).classification)) {
				prePruneError++;
			}
		}
		//save the subtrees to a temp variable
		classTreeNode[] children = head.children;
		head.children = new classTreeNode[] {};
		//check error after pruning
		int postPruneError = 0;
		String[] classPredictions2 = classifySamples(verificationSet,false);
		for(int index=0;index<classPredictions2.length;index++) {
			if(!classPredictions2[index].equals(verificationSet.get(index).classification)) {
				postPruneError++;
			}
		}
		//restore the subtree if error increased
		if(printOutput==true) {System.out.println("Error Pre Prune: " + prePruneError + ", Post Prune:" + postPruneError);}
		if(postPruneError>prePruneError) {
			head.children = children;
		}
		else if (head.isCategorical==false && printOutput==true) {
			System.out.println("SUBTREE PRUNED: FEATURE " + head.feature + " - " + head.midpoint);
		}
		else if(printOutput==true) {
			System.out.println("SUBTREE PRUNED: CATEGORICAL FEATURE " + head.feature);
		}
	}
	
	/*
	 * Method for classifying new samples using an ID3 tree
	 */
	public String[] classifySamples(ArrayList<classificationSample> samples, boolean printOutput) {
		
		String[] classifications = new String[samples.size()];
		
		for(int index=0;index<samples.size();index++) {
			classTreeNode curNode=this.classTreeHead;
			classificationSample sample = samples.get(index);
			//while current node is not a leaf, traverse tree
			while(curNode.children.length>0) {
				//if current node is split on a categorical feature, move by value
				if(printOutput==true && index==0) {System.out.print("CHECKING FEATURE '" + curNode.feature + "', SAMPLE VALUE: " + sample.features[curNode.feature]);}
				if(curNode.isCategorical==true) {
					if(printOutput==true && index==0) {System.out.print(": MOVING ALONG BRANCH " + (int)sample.features[curNode.feature] + "\n");}
					curNode=curNode.children[(int)sample.features[curNode.feature]];
				}
				//if current node is split on a continuous feature, test vs midpoint
				else {
					if(printOutput==true && index==0) {System.out.print(", AGAINST: " + curNode.midpoint);}
					if(sample.features[curNode.feature]<curNode.midpoint) {
						if(printOutput==true && index==0) {System.out.print(": MOVING LEFT\n" );}
						curNode=curNode.children[0];
					}
					else {
						if(printOutput==true && index==0) {System.out.print(": MOVING RIGHT\n" );}
						curNode=curNode.children[1];
					}
				}
			}
			classifications[index] = curNode.classPrediction;
			if(printOutput==true && index==0) {System.out.println("PREDICTING CLASS '" + curNode.classPrediction + "'");}
		}
		return classifications;
	}
	
	/*
	 * Driver method for building a CART tree
	 */
	public void buildRegressTree(ArrayList<regressionSample> trainingSet, int[][] categoricalFeatures, double stoppingPoint, boolean printOutput) {
		this.regressTreeHead = null;
		this.regressTreeHead = new regressTreeNode(trainingSet);
		ArrayList<Integer> features = new ArrayList<Integer>();
		for(int index=0;index<trainingSet.get(0).features.length;index++) {
			features.add(index);
		}
		recursiveRegressTreeBuild(this.regressTreeHead,features,categoricalFeatures, stoppingPoint, printOutput);
	}
	
	/*
	 * Method to build a CART tree
	 */
	public void recursiveRegressTreeBuild(regressTreeNode head, ArrayList<Integer> features, int[][] categoricalFeatures, double stoppingPoint, boolean printOutput) {
		//stop when all samples match in class or there's no features left
		if(features.size()==0 || head.samples.size()<=1) {
			return;
		}
		double MSE = (double) calcSE(head.samples)/head.samples.size();
		if(MSE<stoppingPoint) {
			if(printOutput==true) {System.out.println("STOPPING EARLY, MSE=" + MSE + "<" + stoppingPoint);}
			return;
		}
		double currentMin=Double.MAX_VALUE;
		
		//Iterate by feature
		for (Integer feature: features) {
			//Sort on feature being considered
			Collections.sort(head.samples, new Comparator<regressionSample>() {
				public int compare(regressionSample row1, regressionSample row2) {
					return Double.compare(row1.features[feature],row2.features[feature]);
				}
			});
			//Logic for splitting on a categorical feature
			int temp = arrayContains(categoricalFeatures[0],feature);
			if(temp>-1) {
				//create a child for each category value
				ArrayList<regressionSample>[] children = (ArrayList<regressionSample>[]) new ArrayList[categoricalFeatures[1][temp]];
				for(regressionSample sample: head.samples) {
					if(children[(int)sample.features[feature]]==null) { 
						children[(int)sample.features[feature]] = new ArrayList<regressionSample>();
					}
					children[(int)sample.features[feature]].add(sample);
				}
				//calculate MSE
				MSE = 0;
				for(int index=0;index<children.length;index++) {
					if(children[index]==null) { //create an empty child where null to avoid a NPExpection
						children[index] = new ArrayList<regressionSample>();
					}
					else {
						ArrayList<regressionSample> child = children[index]; 
						MSE += (double) calcSE(child)/head.samples.size();
					}
				}
				//save split if this is a new best
				if(MSE<currentMin && !Double.isNaN(MSE)) {
					if(printOutput) {System.out.println("MSE=" + MSE + ", previous min=" + currentMin);}
					head.feature=feature;
					head.isCategorical=true;
					head.children = new regressTreeNode[children.length];
					for(int index=0;index<children.length;index++) {
						if(children[index].size()==0) {
							head.children[index]=new regressTreeNode(head.mean);
						}
						else {
							head.children[index]=new regressTreeNode(children[index]);
						}
					}
					currentMin=MSE;
				}
			}
			//Logic for splitting on a continuous feature
			else {
				//split data in two relative to midpoint
				double midpoint = head.samples.get(head.samples.size()/2).features[feature];
				ArrayList<regressionSample> lessThan = new ArrayList<regressionSample>();
				ArrayList<regressionSample> greaterThan = new ArrayList<regressionSample>();
				for(regressionSample sample: head.samples) {
					if(sample.features[feature]<midpoint) {
						lessThan.add(sample);
					}
					else {
						greaterThan.add(sample);
					}
				}
				//calc MSE
				MSE = (calcSE(lessThan) + calcSE(greaterThan))/(double) head.samples.size();
				//if new best split, save it
				if(MSE<currentMin && lessThan.size()>0 && greaterThan.size()>0) {
					if(printOutput) {System.out.println("MSE=" + MSE + ", previous min=" + currentMin);}
					head.feature=feature;
					head.isCategorical=false;
					head.midpoint=midpoint;
					head.children = new regressTreeNode[2];
					head.children[0] = new regressTreeNode(lessThan);
					head.children[1] = new regressTreeNode(greaterThan);
					currentMin=MSE;
				}
			}
		}
		//recurse
		ArrayList<Integer> updatedFeatures = new ArrayList<Integer>(features);
		updatedFeatures.remove(Integer.valueOf(head.feature));
		for(regressTreeNode subtree: head.children) {
			recursiveRegressTreeBuild(subtree,updatedFeatures,categoricalFeatures, stoppingPoint, printOutput);
		}
	}

	/*
	 * Method to regress a set of test points
	 */
	public double[] regressSamples(ArrayList<regressionSample> samples, boolean printOutput) {

		double[] regressions = new double[samples.size()];
		
		for(int index=0;index<samples.size();index++) {
			regressTreeNode curNode=this.regressTreeHead;
			regressionSample sample = samples.get(index);
			//while the current node is not a leaf, traverse tree
			while(curNode.children.length>0) {
				if(printOutput==true && index==0) {System.out.print("CHECKING FEATURE '" + curNode.feature + "', SAMPLE VALUE: " + sample.features[curNode.feature]);}
				//if current node is split on a categorical feature, move by value 
				if(curNode.isCategorical==true) {
					if(printOutput==true && index==0) {System.out.print(": MOVING ALONG BRANCH " + (int)sample.features[curNode.feature] + "\n");}
					curNode=curNode.children[(int)sample.features[curNode.feature]];
				}
				//if current node is split on a continuous feature, test vs midpoint
				else {
					if(printOutput==true && index==0) {System.out.print(", AGAINST: " + curNode.midpoint);}
					if(sample.features[curNode.feature]<curNode.midpoint) {
						if(printOutput==true && index==0) {System.out.print(": MOVING LEFT\n" );}
						curNode=curNode.children[0];
					}
					else {
						if(printOutput==true && index==0) {System.out.print(": MOVING RIGHT\n" );}
						curNode=curNode.children[1];
					}
				}
			}
			if(printOutput==true && index==0) {System.out.println("PREDICTING VAL '" + curNode.mean + "'");}
			regressions[index] = curNode.mean;
		}
		return regressions;
	}
	
	/*
	 * Method to calculate squared error for the purposes of checking for early stops and finding the optimal split
	 */
	public double calcSE(ArrayList<regressionSample> samples) {
		double total = 0;
		double mean = 0;
		//calculate mean within the node
		for(regressionSample sample: samples) {
			mean += (double) sample.value/samples.size();
		}
		//calculate squared-error of each sample from that mean
		for(regressionSample sample: samples) {
			total += (sample.value-mean)*(sample.value-mean);
		}
		return total;
	}
	
	/*
	 * Method to calculate entropy
	 */
	public double calcEntropy(ArrayList<classificationSample> samples) {
		double entropy = 0;
		//Create a map to store the count of each class
		HashMap<String,Integer> classCounts = new HashMap<String,Integer>();
		for(classificationSample sample: samples) {
			String classs = sample.classification;
			if(classCounts.get(classs)==null) {
				classCounts.put(classs,1);
			}
			else {
				classCounts.put(classs,classCounts.get(classs)+1);
			}
		}
		//Calculate entropy from class counts
		for(Map.Entry<String,Integer> entry: classCounts.entrySet()) {
			double ratio = (double) entry.getValue()/samples.size();
			entropy += -1.0*ratio*Math.log(ratio)/Math.log(2);
		}
		return entropy;
	}
	
	/*
	 * Method to calculate gain ratio
	 */
	public double calcGainRatio(double entropy, double initEntropy, int initialSize, ArrayList<classificationSample>[] splits) {
		double infoGain = initEntropy-entropy;
		double infoValue = 0;
		for(ArrayList<classificationSample> branch: splits) {
			double ratio = (double) branch.size()/initialSize;
			infoValue += -1.0*ratio*Math.log(ratio)/Math.log(2);
		}
		return infoGain/infoValue;
	}
	
	/*
	 * Method to check whether an array contains a given integer, returns the index where that item is found
	 */
	public static int arrayContains(int[] array, int toCheck) {
		for (int index=0;index<array.length;index++) {
			if (array[index]==toCheck) {
				return index;
			}
		}
		return -1;
	}

	/*
	 * Method to print a classification tree
	 */
	public static void printClassTree(classTreeNode head, String branchIndicator, String indents) {
		if(head.children.length==0) {
			System.out.println(indents + "[" + branchIndicator + "]"+ head.classPrediction);
			return;
		}
		
		if(head.isCategorical==true) {
			System.out.println(indents + "[" + branchIndicator + "] FEATURE '"+ head.feature + "', CATEGORICAL");
			for(int index=0;index<head.children.length;index++) {
				printClassTree(head.children[index],Integer.toString(index),(indents+"\t|"));
			}
		}
		else {
			System.out.println(indents + "[" + branchIndicator + "] FEATURE '"+ head.feature + "', MIDPOINT: " + head.midpoint);
			printClassTree(head.children[0],"<",(indents+"\t|"));
			printClassTree(head.children[1],">=",(indents+"\t|"));
		}
	}
	
	/*
	 * Method to print a regression tree
	 */
	public static void printRegressTree(regressTreeNode head, String branchIndicator, String indents) {
		if(head.children.length==0) {
			System.out.println(indents + "[" + branchIndicator + "]"+ head.mean);
			return;
		}
		
		if(head.isCategorical==true) {
			System.out.println(indents + "[" + branchIndicator + "] FEATURE '"+ head.feature + "', CATEGORICAL");
			for(int index=0;index<head.children.length;index++) {
				printRegressTree(head.children[index],Integer.toString(index),(indents+"\t|"));
			}
		}
		else {
			System.out.println(indents + "[" + branchIndicator + "] FEATURE '"+ head.feature + "', MIDPOINT: " + head.midpoint);
			printRegressTree(head.children[0],"<",(indents+"\t|"));
			printRegressTree(head.children[1],">=",(indents+"\t|"));
		}
	}
}
