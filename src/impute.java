import java.util.*;

public class impute {
	/*
	 * This function imputes missing values. It does so by identifying the class that corresponds to the missing value, iterating over the entire array and compiling the non-"?" values for that missing feature
	 * for all other samples which share it's class into a list. Finally, the algorithm randomly selects a value from that list to impute with, then returns the array.
	 */
	public static String[] randomImpute(ArrayList<String[]> data, String[] toImpute, int classIndex, int attributeIndex) {
		List<String> list = new ArrayList<>();
		//iterate over the whole array to find samples with matching class as the data point to impute
		for (String[] sample: data) {
			//If the samples match and the other sample has a value, add it to the list
			if(sample[classIndex].equals(toImpute[classIndex]) && !sample[attributeIndex].equals("?")) {
				list.add(sample[attributeIndex]);
			}
		}
		//Only randomly select a value if the list isn't empty
		if(list.size()>0) {
			Random rand = new Random();
			String randomVal = list.get(rand.nextInt(list.size()));
			toImpute[attributeIndex] = randomVal;
		}
		//If no values were extracted, return the item as is
		else {
			return toImpute;
		}
		return toImpute;
	}
}