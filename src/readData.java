import java.util.*;
import java.io.*;

public class readData {
	
	public static ArrayList<classificationSample> readClassificationData(String file) throws Exception{
		ArrayList<String[]> stringArray = readToStringArray(file);
		switch(file) {
		case "breast-cancer-wisconsin.data":
			return parseBrca(stringArray);
		case "car.data":
			return parseCars(stringArray);
		case "segmentation.data":
			return parseSegmentation(stringArray);
		default:
			throw new IllegalArgumentException("Unrecognized classification file: " + file);
		}
	}
	
	public static ArrayList<regressionSample> readRegressionData(String file) throws Exception{
		ArrayList<String[]> stringArray = readToStringArray(file);
		switch(file) {
		case "abalone.data":
			return parseAbalone(stringArray);
		case "machine.data":
			return parseMachine(stringArray);
		case "forestfires.csv":
			return parseFires(stringArray);
		default:
			throw new IllegalArgumentException("Unrecognized regression file: " + file);
		}
	}
	
	public static ArrayList<String[]> readToStringArray(String file) throws Exception{
		BufferedReader buffReader = null;
		ArrayList<String[]> stringArray;
		try {
			String currentLine;
			stringArray = new ArrayList<String[]>();
			buffReader = new BufferedReader(new FileReader(file));
			//Read file to string array
			while((currentLine=buffReader.readLine()) != null) {	//read until end of file
				if(!currentLine.trim().equals("")) {	//don't read empty lines
					stringArray.add(currentLine.split(","));
				}
			}
		}
		finally {
			if(buffReader != null) {buffReader.close();}
		}
		return stringArray;
	}
	
	/*
	 * The following are methods for taking input ArrayList<String[]> datasets and parsing them to create ArrayList<sampleNode> arrays
	 * This accomplishes separating classes/regression targets into a named variable and features into a named array, as well as parsing all numerical 
	 * features/regression targets into doubles. 
	 * 
	 * All categorical features have been transformed to an integer (which then must be transformed to a double)
	 */
	public static ArrayList<classificationSample> parseBrca(ArrayList<String[]> data){
		ArrayList<classificationSample> parsedData = new ArrayList<classificationSample>();
		for(String[] sample: data) {
			classificationSample newNode;
			
			if(sample[sample.length-1].equals("2")) {
				newNode = new classificationSample("NORMAL",new double[9]);
			}
			else {
				newNode = new classificationSample("MALIGNANT",new double[9]);
			}
			
			for(int feature=1;feature<sample.length-2;feature++) {
				if(sample[feature].equals("?")) {
					sample = impute.randomImpute(data,sample,sample.length-1,feature);
				}
				newNode.features[feature-1] = Integer.parseInt(sample[feature]);
			}
			parsedData.add(newNode);
		}
		return parsedData;
	}
	
	public static ArrayList<classificationSample> parseCars(ArrayList<String[]> data){
		ArrayList<classificationSample> parsedData = new ArrayList<classificationSample>();
		for(String[] sample: data) {
			classificationSample newNode = new classificationSample(sample[sample.length-1],new double[sample.length-1]);
			
			for(int feature=0;feature<sample.length-1;feature++) {
				if(feature==0 || feature==1) {
					switch(sample[feature]) {
					case "v-high":
						newNode.features[feature] = 0; break;
					case "high":
						newNode.features[feature] = 1; break;
					case "med":
						newNode.features[feature] = 2; break;
					case "low":
						newNode.features[feature] = 3; break;
					}
				}
				else if(feature==2) {
					switch(sample[feature]) {
					case "2":
						newNode.features[feature] = 0; break;
					case "3":
						newNode.features[feature] = 1; break;
					case "4":
						newNode.features[feature] = 2; break;
					case "5-more":
						newNode.features[feature] = 3; break;
					}
				}
				else if(feature==3) {
					switch(sample[feature]) {
					case "2":
						newNode.features[feature] = 0; break;
					case "4":
						newNode.features[feature] = 1; break;
					case "more":
						newNode.features[feature] = 2; break;
					}
				}
				else if(feature==4) {
					switch(sample[feature]) {
					case "small":
						newNode.features[feature] = 0; break;
					case "med":
						newNode.features[feature] = 1; break;
					case "big":
						newNode.features[feature] = 2; break;
					}
				}
				else if(feature==5) {
					switch(sample[feature]) {
					case "low":
						newNode.features[feature] = 0; break;
					case "med":
						newNode.features[feature] = 1; break;
					case "high":
						newNode.features[feature] = 2; break;
					}
				}
			}
			
			parsedData.add(newNode);
		}
		return parsedData;
	}
	
	public static ArrayList<classificationSample> parseSegmentation(ArrayList<String[]> data){
		ArrayList<classificationSample> parsedData = new ArrayList<classificationSample>();
		for(String[] sample: data) {
			classificationSample newNode = new classificationSample(sample[0],new double[sample.length-1]);
			
			for(int feature=1;feature<sample.length;feature++) {
				newNode.features[feature-1] = Double.parseDouble(sample[feature]);
			}
			
			parsedData.add(newNode);
		}
		return parsedData;
	}
	
	public static ArrayList<regressionSample> parseAbalone(ArrayList<String[]> data){
		ArrayList<regressionSample> parsedData = new ArrayList<regressionSample>();
		for(String[] sample: data) {
			regressionSample newNode = new regressionSample(Double.parseDouble(sample[sample.length-1]),new double[sample.length-1]);
			
			switch(sample[0]) {
			case "I":
				newNode.features[0] = 0;
				break;
			case "M":
				newNode.features[0] = 1;
				break;
			case "F":
				newNode.features[0] = 2;
				break;
			}
			for(int feature=1;feature<sample.length-1;feature++) {
				newNode.features[feature] = Double.parseDouble(sample[feature]);
			}
			
			parsedData.add(newNode);
		}
		return parsedData;
	}
	
	public static ArrayList<regressionSample> parseMachine(ArrayList<String[]> data){
		ArrayList<regressionSample> parsedData = new ArrayList<regressionSample>();
		for(String[] sample: data) {
			regressionSample newNode = new regressionSample(Double.parseDouble(sample[sample.length-2]),new double[sample.length-3]);
			
			switch(sample[0]) {
			case "adviser":
				newNode.features[0]=0; break;
			case "amdahl":
				newNode.features[0]=1; break;
			case "apollo":
				newNode.features[0]=2; break;
			case "basf":
				newNode.features[0]=3; break;
			case "bti":
				newNode.features[0]=4; break;
			case "burroughs":
				newNode.features[0]=5; break;
			case "c.r.d":
				newNode.features[0]=6; break;
			case "cdc":
				newNode.features[0]=7; break;
			case "cambex":
				newNode.features[0]=8; break;
			case "dec":
				newNode.features[0]=9; break;
			case "dg":
				newNode.features[0]=10; break;
			case "formation":
				newNode.features[0]=11; break;
			case "four-phase":
				newNode.features[0]=12; break;
			case "gould":
				newNode.features[0]=13; break;
			case "hp":
				newNode.features[0]=14; break;
			case "harris":
				newNode.features[0]=15; break;
			case "honeywell":
				newNode.features[0]=16; break;
			case "ibm":
				newNode.features[0]=17; break;
			case "ipl":
				newNode.features[0]=18; break;
			case "magnuson":
				newNode.features[0]=19; break;
			case "microdata":
				newNode.features[0]=20; break;
			case "nas":
				newNode.features[0]=21; break;
			case "ncr":
				newNode.features[0]=22; break;
			case "nixdorf":
				newNode.features[0]=23; break;
			case "perkin-elmer":
				newNode.features[0]=24; break;
			case "prime":
				newNode.features[0]=25; break;
			case "siemens":
				newNode.features[0]=26; break;
			case "sperry":
				newNode.features[0]=27; break;
			case "sratus":
				newNode.features[0]=28; break;
			case "wang":
				newNode.features[0]=29; break;
			}
			for(int feature=2;feature<sample.length-2;feature++) {
				newNode.features[feature-1]=Double.parseDouble(sample[feature]);
			}
			
			parsedData.add(newNode);
		}
		return parsedData;
	}

	public static ArrayList<regressionSample> parseFires(ArrayList<String[]> data){
		ArrayList<regressionSample> parsedData = new ArrayList<regressionSample>();
		
		for(String[] sample: data) {
			regressionSample newNode = new regressionSample(Math.log(Double.parseDouble(sample[sample.length-1])+1),new double[sample.length-1]);
			
			for(int feature=0;feature<sample.length-1;feature++) {
				if (feature==2) {
					switch(sample[feature]) {
					case "jan":
						newNode.features[feature] = 0; break;
					case "feb":
						newNode.features[feature] = 1; break;
					case "mar":
						newNode.features[feature] = 2; break;
					case "apr":
						newNode.features[feature] = 3; break;
					case "may":
						newNode.features[feature] = 4; break;
					case "jun":
						newNode.features[feature] = 5; break;
					case "jul":
						newNode.features[feature] = 6; break;
					case "aug":
						newNode.features[feature] = 7; break;
					case "sep":
						newNode.features[feature] = 8; break;
					case "oct":
						newNode.features[feature] = 9; break;
					case "nov":
						newNode.features[feature] = 10; break;
					case "dec":
						newNode.features[feature] = 11; break;
					}
				}
				else if (feature==3) {
					switch(sample[feature]) {
					case "sun":
						newNode.features[feature] = 0; break;
					case "mon":
						newNode.features[feature] = 1; break;
					case "tue":
						newNode.features[feature] = 2; break;
					case "wed":
						newNode.features[feature] = 3; break;
					case "thu":
						newNode.features[feature] = 4; break;
					case "fri":
						newNode.features[feature] = 5; break;
					case "sat":
						newNode.features[feature] = 6; break;
					}
				}
				else {
					newNode.features[feature]=Double.parseDouble(sample[feature]);
				}
			}
			parsedData.add(newNode);
		}
		
		return parsedData;
	}
}
