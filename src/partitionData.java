import java.util.*;

public class partitionData {
	public static ArrayList<ArrayList<regressionSample>> partitionRegressionData(ArrayList<regressionSample> data) {
		ArrayList<regressionSample> trainingSet = new ArrayList<regressionSample>();
		
		//Regression problems need to be sorted first
		Collections.sort(data, new Comparator<regressionSample>() {
			public int compare(regressionSample row1, regressionSample row2) {
				return Double.compare(row1.value,row2.value);
			}
		});
		
		 //First, remove to trainingSet (this is before shuffling a classification problem to keep the training dataset consistent)
		 for(int index=data.size()-1;index>=0;index--) {
			 if(index != 0 && index%10==0) {
				 trainingSet.add(data.remove(index));
			 }
		 }
		 
		 ArrayList<regressionSample> fold1 = new ArrayList<regressionSample>();
		 ArrayList<regressionSample> fold2 = new ArrayList<regressionSample>();
		 ArrayList<regressionSample> fold3 = new ArrayList<regressionSample>();
		 ArrayList<regressionSample> fold4 = new ArrayList<regressionSample>();
		 ArrayList<regressionSample> fold5 = new ArrayList<regressionSample>();
		 
		 //remove every 1 in 5 consecutive samples to each fold
		 for(int index=data.size()-1;index>=0;index--) {
			 switch(index%5) {
			 	case 0:
			 		fold1.add(data.remove(index));
			 		break;
			 	case 1:
			 		fold2.add(data.remove(index));
			 		break;
			 	case 2:
			 		fold3.add(data.remove(index));
			 		break;
			 	case 3:
			 		fold4.add(data.remove(index));
			 		break;
			 	case 4:
			 		fold5.add(data.remove(index));
			 		break;	
			 }
		 }
		 
		 ArrayList<ArrayList<regressionSample>> returnData = new ArrayList<ArrayList<regressionSample>>();
		 returnData.add(fold1);
		 returnData.add(fold2);
		 returnData.add(fold3);
		 returnData.add(fold4);
		 returnData.add(fold5);
		 returnData.add(trainingSet);
		 
		 return returnData;
	}
	
	 public static ArrayList<ArrayList<classificationSample>> partitionClassificationData(ArrayList<classificationSample> data) {
		ArrayList<classificationSample> trainingSet = new ArrayList<classificationSample>();

		 //First, remove to trainingSet (this is before shuffling a classification problem to keep the training dataset consistent)
		 for(int index=data.size()-1;index>=0;index--) {
			 if(index != 0 && index%10==0) {
				 trainingSet.add(data.remove(index));
			 }
		 }
		 
		 //Classification problems are shuffled, then sorted preserving the random order within each class
		 	Collections.shuffle(data);
		 	Collections.sort(data, new Comparator<classificationSample>() {
		 		public int compare(classificationSample row1, classificationSample row2) {
		 			return row1.classification.compareTo(row2.classification);
		 		}
		 	});
		 
		 	ArrayList<classificationSample> fold1 = new ArrayList<classificationSample>();
			 ArrayList<classificationSample> fold2 = new ArrayList<classificationSample>();
			 ArrayList<classificationSample> fold3 = new ArrayList<classificationSample>();
			 ArrayList<classificationSample> fold4 = new ArrayList<classificationSample>();
			 ArrayList<classificationSample> fold5 = new ArrayList<classificationSample>();
			 
			 //remove every 1 in 5 consecutive samples to each fold
			 for(int index=data.size()-1;index>=0;index--) {
				 switch(index%5) {
				 	case 0:
				 		fold1.add(data.remove(index));
				 		break;
				 	case 1:
				 		fold2.add(data.remove(index));
				 		break;
				 	case 2:
				 		fold3.add(data.remove(index));
				 		break;
				 	case 3:
				 		fold4.add(data.remove(index));
				 		break;
				 	case 4:
				 		fold5.add(data.remove(index));
				 		break;	
				 }
			 }
			 
			 ArrayList<ArrayList<classificationSample>> returnData = new ArrayList<ArrayList<classificationSample>>();
			 returnData.add(fold1);
			 returnData.add(fold2);
			 returnData.add(fold3);
			 returnData.add(fold4);
			 returnData.add(fold5);
			 returnData.add(trainingSet);
			 
			 return returnData;
	 }
}
