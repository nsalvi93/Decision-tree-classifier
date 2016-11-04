/*
 * UTILITY CLASS CREATED FOR CALCULATING ENTROPY AND INFORMATION GAIN
 * 
 * */
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
public class Entropy 
{
	public Map<Integer, Double> entropyMap ;
	public Map<Integer, Double> infoMap;
	public double priorEntropy =0;
	
	public void boostingCalcPrioEntropy(List<ArrayList<Integer>> dataLists, List<Double> weightList)
	{
		// doing dataLists.size() - 2 since that corresponds to the class label 
		// dataLists.size() - 1 corresponds to weights
		
		double positiveCount = dataLists.get(dataLists.size() - 1).stream().filter(number -> number.equals(1)).count();
		double negativeCount = dataLists.get(dataLists.size() - 1).stream().filter(number -> number.equals(0)).count();
		double classCount = dataLists.get(dataLists.size() - 1).size();   // gets count of class label
		
		double sum = weightList.stream().mapToDouble(Double::doubleValue).sum();
		double positiveCountWeight = positiveCount * weightList.get(0);	// since all weights will be the same at the start
		double negativeCountWeight = negativeCount * weightList.get(0);
		
		double normalizedPosWeight = positiveCountWeight/sum;
		double normalizedNegWeight = negativeCountWeight/sum;
		
		
		priorEntropy = -(normalizedPosWeight/classCount) * (Math.log(normalizedPosWeight/classCount)/ Math.log(2)) -(normalizedNegWeight/classCount) * (Math.log(normalizedNegWeight/classCount)/ Math.log(2)) ;
		
		//priorEntropy = -(positiveCount/classCount) * (Math.log(positiveCount/classCount)/ Math.log(2)) -(negativeCount/classCount) * (Math.log(negativeCount/classCount)/ Math.log(2)) ;
		System.out.println("prior entropy is " + priorEntropy );
		
	}
	
	
	
	public void calculatePriorEntropy(List<ArrayList<Integer>> dataLists)
	{
		//System.out.println(dataLists.get(dataLists.size()-1).toString());
		double positiveCount = dataLists.get(dataLists.size() - 1).stream().filter(number -> number.equals(1)).count();
		double negativeCount = dataLists.get(dataLists.size() - 1).stream().filter(number -> number.equals(0)).count();
		double classCount = dataLists.get(dataLists.size() - 1).size();   // gets count of class label
		priorEntropy = -(positiveCount/classCount) * (Math.log(positiveCount/classCount)/ Math.log(2)) -(negativeCount/classCount) * (Math.log(negativeCount/classCount)/ Math.log(2)) ;
		System.out.println("prior entropy is " + priorEntropy );
	}
	public double calcEntropy(double a, double b)  // b corresponds to +ve count i.e. 1 and a corresponds to -ve count i.e. 0
	{
		double total = a +b;
		if(a ==0 || b ==0)
		{
			return 0;
		}
		else
		{
			return((-(b/total) * Math.log(b/total)/ Math.log(2))-(a/total) * (Math.log(a/total)/ Math.log(2)));
		}
	}
	public double[] returnHighestInfoGain()
	{
		double lowestEntropy = Double.MAX_VALUE; int lowestEntropyColumnIndex =0; double[] array = new double[2];
		System.out.println();
		for(Integer key : entropyMap.keySet())
		{
			double compare = entropyMap.get(key);
			if(compare < lowestEntropy )
			{
				lowestEntropy = compare;
				lowestEntropyColumnIndex = key;
			}
		}
		array[0] = lowestEntropy;
		array[1] = lowestEntropyColumnIndex;
		return array ;
	}
	public void calculateTotalEntropy(Map<Integer, Map<Integer, double[]>> featureAttrCountMap, int totalEntries) 
	{
		entropyMap = new LinkedHashMap<>();
		for(Integer feature : featureAttrCountMap.keySet())
		{
			double featureEntropy =0;
			Map<Integer, double[]> attrCountMap = featureAttrCountMap.get(feature);
			for(Integer value : attrCountMap.keySet())
			{
				double[] array = attrCountMap.get(value);
				double totalAttrCount = array[0] + array[1];
				featureEntropy = featureEntropy + (totalAttrCount/ totalEntries) * calcEntropy(array[0], array[1] );
			}
			entropyMap.put(feature, featureEntropy);
		}
	}
	public void calculateInfoGain(double priorEntropy)
	{
		infoMap = new LinkedHashMap<>();
		for(Integer feature : entropyMap.keySet())
		{
			double entropy = entropyMap.get(feature);
			double infogain = priorEntropy - entropy;
			infoMap.put(feature, infogain);
		}
	}
}
