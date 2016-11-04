/* ######### PLEASE READ ###########
 * Arguments: 
Training_data_filename Test_data_filename depth 
Eg: monk1.txt monk1test.txt 2
Training data filenames: monk1.txt, monk2.txt, monk3.txt
Test data filename: monk1test.txt, monk2test.txt, monk3test.txt
Important note: No error checking has been provided in case:
1.	Wrong filename is provided as input or spelling error.
2.	Extra spacing or character in between the arguments.
3.	Incorrect no. of arguments is provided.
4.	Incorrect ordering of arguments.
OUTPUT:
1. Confusion matrix
2. Accuracy
3. Tree
Note:
The format of printing is: feature_name (parent_attribute it has split on).
SAMPLE EXAMPLE:
####### LEVEL 0 #######
a5(0)	
1234	
####### LEVEL 1 #######
a4(2)	a6(3)	a1(4)	
123	12	123	
feature a4 is splitting on attribute 2 of feature a5.
 * */
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
import java.util.Stack;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.Stream;
/**
 * ################ NODE CLASS BELOW AND METHODS ################
 */
class Node
{
	int featureName;
	int depth;
	boolean leaf;
	boolean feature;
	double nodeEntropy;
	List<ArrayList<Integer>> nodeDataset;
	Queue<Node> childNodes;
	Node parent;
	public Node(int feature)
	{
		this.featureName = feature;
		this.leaf = false;
		this.nodeDataset = null;
		this.childNodes = new LinkedList<>();
		this.parent = null;
		this.feature  = false;
	}
	public void setParent(Node parent)
	{
		this.parent = parent;
	}
	public void setDataset(List<ArrayList<Integer>> dataset)
	{
		this.nodeDataset = dataset;
	}
	public void setEntropy(Double entropy)
	{
		this.nodeEntropy = entropy;
	}
	public void setDepth( int depth)
	{
		this.depth = depth;
	}
}
/**
 * ################ DRIVER CLASS ################
 * 
 * **/
public class DecisionTrees 
{
	public Node rootNode = null;
	List<ArrayList<Integer>> dataLists = new ArrayList<ArrayList<Integer>>();		// for capturing data set
	List<ArrayList<Integer>> resultLists = new ArrayList<ArrayList<Integer>>();		// for capturing results
	List<ArrayList<Integer>> testDataLists = new ArrayList<ArrayList<Integer>>();	// for capturing test data set
	List<Integer> featureQueue = new ArrayList<>();									// total no. features in dataset
	
	List<Double> weightList = new ArrayList<>();									// weight list for boosting
	
	static Entropy entropyUtility = new Entropy();									// global object of Entropy class to calculate gain and entropy
	int depth;																		// global variable for depth
	String method;
	
	public static void main(String[] args) 
	{
		
		DecisionTrees tree = new DecisionTrees();
		tree.method = "boosting";
		tree.depth = 10;												// getting depth
		List<String> recordList = tree.readFile("train1.csv");

		tree.populateFeatures(recordList);
		tree.createFeatureList1(recordList, tree.dataLists);

		if(tree.method.equals("bagging"))
		{
			tree.buildEnsembles();
			//tree.buildTree();											// builds decision tree on training data
			//tree.testdata_traversal("monk1test.txt");					// tests on test data
			//tree.print_decision_tree(tree.rootNode);					// prints the decision tree. Comment if not needed
		}
		else
		{
			
			
			double size = tree.dataLists.get(0).size();				// total no. of samples
			System.out.println(tree.dataLists.get(0).size());
			System.out.println(1/size + "&&&&");
			for(int i=0; i< tree.dataLists.get(0).size(); i++)
			{
				tree.weightList.add((double) (1/(size)));				// adding initial weight in the form of list
			}
			tree.learnBoosting((ArrayList<ArrayList<Integer>>) tree.dataLists);
			
			//System.out.println(tree.weightList.toString());
			
		}

		tree.dataLists.clear();
		tree.testDataLists.clear();
		tree.featureQueue.clear();

	}

	private void learnBoosting(ArrayList<ArrayList<Integer>> boostingDataLists) 
	{
		buildTree(boostingDataLists);
		
	}

	private void populateFeatures(List<String> recordList) 
	{
		int featureNo = recordList.get(0).trim().split(",").length;   // getting first line of dataset for features

		System.out.println(featureNo);
		for(int i=0; i< featureNo-1; i++ )			 // not considering feature bruises?-no in the feature queue
		{
			featureQueue.add(i);
		}

		System.out.println("Look here " + featureQueue.size());

	}

	public void buildEnsembles()
	{
		// creating 2 bags and setting bound as 60% of the total size
		int n = 3; Random random = new Random(); int bound = (int) (dataLists.get(0).size() * 0.75);
		System.out.println("*******"+bound);
		List<ArrayList<ArrayList<Integer>>> ensembleList = new ArrayList<ArrayList<ArrayList<Integer>>>();
		while(ensembleList.size() < n)
		{
			ArrayList<ArrayList<Integer>> ensemble = new ArrayList<ArrayList<Integer>>();
			for(int i=0; i< dataLists.size(); i++)								
			{
				ensemble.add(new ArrayList<Integer>());
			}

			while(ensemble.get(0).size() < bound)		// getting the newly added ensemble from the ensemble list
			{
				int dataPoint = random.nextInt(bound);							// since bound is not inclusive
				//System.out.println(dataPoint + " ensemble no. " + ensembleList.size());
				//System.out.println("picked " +dataPoint);
				//ArrayList<ArrayList<Integer>> addedEnsemble = ensembleList.get(ensembleList.size() - 1);
				for(int i =0; i< dataLists.size(); i++)
				{
					ensemble.get(i).add(dataLists.get(i).get(dataPoint));

				}
				//System.out.println("");

			}
			ensembleList.add(ensemble);
			System.out.println("Ended filling the ensemble");
		}
		int i=1;
		for(ArrayList<ArrayList<Integer>> ensemble : ensembleList)
		{
			System.out.println("for ensemble "+ i);
			buildTree(ensemble);
			testdata_traversal("test1.csv");
			i++;
			//print_decision_tree(rootNode);

		}

		/*for(ArrayList<ArrayList<Integer>> ensemble : ensembleList)
		{
			printUtilityFunction(ensemble);
			System.out.println("############### FINISHED ONE ################");
			System.out.println(ensemble.get(0).size());
			System.out.println(ensemble.size());
		}*/
	}


	/* Function to construct tree */
	public void buildTree(ArrayList<ArrayList<Integer>> dataLists)
	{
		if(method.equals("bagging")){entropyUtility.calculatePriorEntropy(dataLists);} // Function to calculate prior entropy for bagging
		
		else {entropyUtility.boostingCalcPrioEntropy(dataLists, weightList);}
				
		/*decideRootNode(dataLists);										// Function to decide the root feature
		Queue<Node> rootChildNodes = new LinkedList<>();
		rootChildNodes.addAll(rootNode.childNodes);
		// Builds the tree in BFS manner 
		while(!rootChildNodes.isEmpty() && rootChildNodes.peek().depth<= depth)  
		{
			Node nodeChild = rootChildNodes.poll();

			//System.out.println("Node in question "+ nodeChild.featureName + " child of " + nodeChild.parent.featureName);

			splitDataSet(nodeChild.parent.nodeDataset, nodeChild.parent.featureName, nodeChild);		// splits the dataset

			//printUtilityFunction(nodeChild.nodeDataset);
			Map<Integer, Map<Integer, double[]>> emptyFeatureAttrCountMap = getFeatureValues(featureQueue, nodeChild.nodeDataset);  // function to get all unique feature values

			//System.out.println("&&&& check here &&&&");
			//printUtilityForFeatureAttrMap(emptyFeatureAttrCountMap);


			Map<Integer, Map<Integer, double[]>> filledFeatureAttrCountMap = calculateBestFeature(nodeChild.nodeDataset, emptyFeatureAttrCountMap); // gets the count of the above values


			//printUtilityForFeatureAttrMap(filledFeatureAttrCountMap);

			entropyUtility.calculateTotalEntropy(filledFeatureAttrCountMap, nodeChild.nodeDataset.get(0).size());    // calculates entropy for all features
			if(entropyUtility.entropyMap.values().stream().mapToDouble(Double::doubleValue).sum() == 0)
			{
				System.out.println("###### leaf found at "+nodeChild.featureName +  " of featureParent "+ nodeChild.parent.featureName );
				nodeChild.leaf = true;
				nodeChild.nodeEntropy = 0;
				nodeChild.childNodes = null;
				nodeChild.setDepth(nodeChild.parent.depth);
				entropyUtility.entropyMap.clear();
				entropyUtility.infoMap.clear();
				continue;
			}
			entropyUtility.calculateInfoGain(nodeChild.parent.nodeEntropy); 					// calculates info gain for all features
			double[] array = entropyUtility.returnHighestInfoGain();							// returns feature with highest info gain and corresponding entropy
			int rootFeature = (int) array[1];

			entropyUtility.infoMap.clear();
			entropyUtility.entropyMap.clear();
			Node feature = new Node(rootFeature);  
			feature.parent = nodeChild;
			feature.setEntropy(array[0]);

			System.out.println("^^^^^^^ splitting at "+rootFeature + " of attribute " + nodeChild.featureName + " of featureParent "+ nodeChild.parent.featureName + " with entropy "+ feature.nodeEntropy );

			feature.depth = nodeChild.depth+1;
			feature.feature = true;
			feature.nodeDataset = feature.parent.nodeDataset;
			nodeChild.childNodes.add(feature);					 
			for(int childNode : filledFeatureAttrCountMap.get(feature.featureName).keySet())    // adding feature unique children (attributes)
			{
				feature.childNodes.add(new Node(childNode));   // setting children
			}
			for(Node childNode : feature.childNodes)
			{
				childNode.setParent(feature);
				childNode.setDepth(childNode.parent.depth);   // setting children depth
				rootChildNodes.add(childNode);				  // filling up the while Queue	
			}
		} */
	}
	/**
	 * Function to split the dataset according the attribute of the parent feature
	 * 
	 * Accepts previous dataset, feature to split by, child attribute node
	 * 
	 * 
	 * **/
	public void splitDataSet(List<ArrayList<Integer>> dataLists2, int featureIndexToSplitBy, Node nodeChild  )
	{
		List<ArrayList<Integer>> newdataLists = new ArrayList<ArrayList<Integer>>();    // declaring new dataset to be set for given node
		ArrayList<Integer> featureListToSplitBy = dataLists2.get(featureIndexToSplitBy); 	// takes feature list (highest info gain feature) from original dataset
		for(int i=0; i< featureQueue.size(); i++)				  						// + 1 since we are considering features + class label list				
		{
			newdataLists.add(new ArrayList<Integer>());									// adding arraylists depending on feature count (one less)
		}
		for(int i=0; i< featureListToSplitBy.size(); i++)
		{
			if(featureListToSplitBy.get(i) == nodeChild.featureName)					// comparing when node child attr value appears in the feature (list)
			{																			// used to split by	
				for(int j=0; j< dataLists2.size(); j++ )
				{
					newdataLists.get(j).add(dataLists2.get(j).get(i)); 					// when found adding other list values to new dataset
				}
			}
		}
		nodeChild.setDataset(newdataLists);
		//printUtilityFunction(newdataLists);
		//System.out.println(newdataLists.size() + " for "+ nodeChild.featureName);
	}
	/*
	 * Function that decides the rootNode
	 * */
	public void decideRootNode(ArrayList<ArrayList<Integer>> dataLists) 
	{
		Map<Integer, Map<Integer, double[]>> emptyFeatureAttrCountMap = getFeatureValues(featureQueue, dataLists);
		// printing empty feature map
		System.out.println(featureQueue.toString());
		//printUtilityForFeatureAttrMap(emptyFeatureAttrCountMap);

		Map<Integer, Map<Integer, double[]>> filledFeatureAttrCountMap = calculateBestFeature(dataLists, emptyFeatureAttrCountMap);

		//printUtilityForFeatureAttrMap(filledFeatureAttrCountMap);

		entropyUtility.calculateTotalEntropy(filledFeatureAttrCountMap, dataLists.get(0).size());
		entropyUtility.calculateInfoGain(entropyUtility.priorEntropy);
		double[] array = entropyUtility.returnHighestInfoGain();
		int rootFeature = (int) array[1];

		System.out.println( "Root feature is "+  rootFeature);

		entropyUtility.infoMap.clear();
		entropyUtility.entropyMap.clear();
		rootNode = new Node(rootFeature);       // setting root
		rootNode.setDataset(dataLists);			// setting dataset
		rootNode.setEntropy(array[0]);			// setting entropy	
		rootNode.setDepth(0);					// setting depth
		rootNode.feature = true;				// setting feature	
		rootNode.setParent(new Node(0));        // setting parent as 0 meaning null 
		//System.out.println("Unique values are "+ filledFeatureAttrCountMap.get(rootNode.featureName).keySet().toString());
		for(int childNode : filledFeatureAttrCountMap.get(rootNode.featureName).keySet())    // adding feature unique children (attributes)
		{
			System.out.println("rootchildren are " + childNode);
			rootNode.childNodes.add(new Node(childNode));   // setting children
		}
		for(Node childNode : rootNode.childNodes)
		{
			childNode.setParent(rootNode);
			childNode.setDepth(childNode.parent.depth);   // setting children depth
		}
	}
	/**
	 *  Best feature Function()
	 * @return 
	 * **/
	public Map<Integer, Map<Integer, double[]>> calculateBestFeature(List<ArrayList<Integer>> dataLists2, Map<Integer, Map<Integer, double[]>> featureAttrCountMap) 
	{
		Map<Integer, Map<Integer, double[]>> filledFeatureAttrCountMap = getImpurity(featureAttrCountMap, dataLists2);
		return filledFeatureAttrCountMap;
	}
	/**
	 *  Best feature Function()
	 * **/
	/**
	 *  impurity Function(): Calculates the no. of feature values again the class labels
	 * @return 
	 * **/
	public Map<Integer, Map<Integer, double[]>> getImpurity(Map<Integer, Map<Integer, double[]>> featureAttrCountMap, List<ArrayList<Integer>> dataLists2) 
	{
		for(Integer columnIndex : featureAttrCountMap.keySet())
		{
			ArrayList<Integer> featureColumn = dataLists2.get(columnIndex);
			for(int i=0; i< featureColumn.size(); i++)
			{
				//System.out.println(dataLists2.get(dataLists2.size() -1).get(i)+ "*****");
				if(dataLists2.get(dataLists2.size() -1).get(i)== 1)
				{
					Map<Integer, double[]> attrCountMap = featureAttrCountMap.get(columnIndex);
					double[] array = attrCountMap.get(featureColumn.get(i));
					array[1]++;
					attrCountMap.put(featureColumn.get(i), array);
					featureAttrCountMap.put(columnIndex, attrCountMap);
				}
				else
				{
					Map<Integer, double[]> attrCountMap = featureAttrCountMap.get(columnIndex);
					double[] array = attrCountMap.get(featureColumn.get(i));
					array[0]++;
					attrCountMap.put(featureColumn.get(i), array);
					featureAttrCountMap.put(columnIndex, attrCountMap);
				}
			}
		}
		return featureAttrCountMap;
	}
	/*
	 * Function for getting unique feature values in map format from the existing features in feature queue and ones present in dataset
	 * */
	public Map<Integer, Map<Integer, double[]>> getFeatureValues(List<Integer> featureQueue2, List<ArrayList<Integer>> dataLists2)
	{
		Map<Integer, Map<Integer, double[]>> featureAttrCountMap = new LinkedHashMap<>();
		//for(Integer columnIndex : featureQueue2)
		for(int i=0; i< featureQueue2.size()-1; i++)
		{
			Map<Integer, double[]> attrCountMap = new LinkedHashMap<>();
			Set<Integer> uniqueSet = dataLists2.get(featureQueue2.get(i)).stream().collect(Collectors.toSet());
			for(Integer featureAttrs : uniqueSet){attrCountMap.put(featureAttrs,new double[]{0,0});}
			featureAttrCountMap.put(featureQueue2.get(i), attrCountMap);
		}
		return featureAttrCountMap;
	}
	/**######################################################
	 * ###### Utility Functions for printing  #######
	 * ######################################################
	 * **/
	public void printUtilityFunction(List<ArrayList<Integer>> dataLists2)
	{
		for(int i=0; i< dataLists2.get(0).size(); i++)
		{
			System.out.println("");
			System.out.print(i);
			for(int j=0; j< dataLists2.size(); j++ )
			{
				System.out.print(" "+dataLists2.get(j).get(i)+ " ");
			}
		}
		System.out.println("");
	}
	public String printUtilityForArrays(double[] array)
	{
		return Arrays.toString(array);
	}
	public void printUtilityForFeatureAttrMap(Map<Integer, Map<Integer, double[]>> featureAttrCountMap)
	{
		for(Integer feature : featureAttrCountMap.keySet())
		{
			Map<Integer, double[]> attrCountMap = featureAttrCountMap.get(feature);
			System.out.println("For feature "+ feature);
			for(Integer value : attrCountMap.keySet())
			{
				System.out.print(value + " : " + printUtilityForArrays(attrCountMap.get(value)) + " ");
			}
			System.out.println("");
		}
	}
	/**######################################################
	 * ###### Utility Functions for printing  #######
	 * ######################################################
	 * **/
	/**######################################################
	 * ###### File read on training set and feature creation functions #######
	 * ######################################################
	 * @param dataLists2 
	 * @return 
	 * **/
	/*
	 * Function for reading file and storing the features and attributes
	 * */
	public List<String> readFile(String filename)
	{
		List<String> recordList = new ArrayList<>();
		try (Stream<String> stream = Files.lines(Paths.get(filename))) 
		{
			recordList = stream.collect(Collectors.toList());
		} catch (IOException e) { 
			e.printStackTrace();
		}
		//System.out.println(recordList.toString());
		return recordList;
	}

	/**
	 * Testing 
	 * **/

	public void createFeatureList1(List<String> recordList, List<ArrayList<Integer>> parsedDataList)			// made changes here since only 21 attributes required
	{
		int featureNo = recordList.get(0).trim().split(",").length;   // getting first line of dataset for features


		for(int i=0; i< featureNo; i++)				// adding all features and then will remove feature bruises?-no 	
		{
			parsedDataList.add(new ArrayList<Integer>());
		}
		for(int j=1; j< recordList.size(); j++)
		{
			String [] parseRecord = recordList.get(j).trim().split(",");
			for(int i=0; i< featureNo; i++)
			{

				parsedDataList.get(i).add(Integer.valueOf(parseRecord[i]));  
			}
		}

		parsedDataList.remove(21);
		ArrayList<Integer> classLabelColumn = parsedDataList.get(20); parsedDataList.remove(20);
		parsedDataList.add(classLabelColumn);
		System.out.println(parsedDataList.size());
		//printUtilityFunction(parsedDataList);
	}


	/*
	 * Function to decide the no. of features
	 * */
	/*public void createFeatureList(List<String> recordList, List<ArrayList<Integer>> parsedDataList)			// made changes here since only 21 attributes required
	{
		int featureNo = recordList.get(0).trim().split(",").length;   // getting first line of dataset for features

//		String[] features = recordList.get(0).trim().split(",");
//		for(String feature : features)
//		{
//			System.out.println(feature);
//		}

		System.out.println(featureNo);
		for(int i=0; i< featureNo; i++ )
		{
			featureQueue.add(i);
		}

//		for(Integer feature : featureQueue)
//		{
//			System.out.println(feature);
//		}

		for(int i=0; i< featureNo; i++)								
		{
			parsedDataList.add(new ArrayList<Integer>());
		}
		for(int j=1; j< recordList.size(); j++)
		{
			String [] parseRecord = recordList.get(j).trim().split(",");
			for(int i=0; i< featureNo; i++)
			{
				parsedDataList.get(i).add(Integer.valueOf(parseRecord[i]));  
			}
		}

		//printUtilityFunction(parsedDataList);
	}*/


	/**######################################################
	 * ###### File read on training set and feature creation functions #######
	 * ######################################################
	 * **/
	/**######################################################
	 * ###### TESTING SET #######
	 * ######################################################
	 * **/
	public void testdata_traversal(String test_data_filename)
	{
		List<String> recordList = readFile(test_data_filename); 
		List<Integer> resultList = new ArrayList<>();

		createFeatureList1(recordList, testDataLists);

		for(int i=0; i< testDataLists.get(0).size(); i++ )
		{
			int class_label_0_count =0; boolean didNotFindMatch = false;
			int class_label_1_count =0;
			Node next_feature;
			Node next_attr = null;
			Stack<Node> nodeFeature_stack = new Stack<>();
			nodeFeature_stack.push(rootNode);
			while(nodeFeature_stack.peek().depth < depth+1) 					// sets depth (depth + 1)
			{
				next_feature = nodeFeature_stack.pop();
				int comparing_root = testDataLists.get(next_feature.featureName).get(i);
				for(Node feature_child : next_feature.childNodes)
				{
					if(feature_child.featureName == comparing_root)
					{
						next_attr = feature_child;
						didNotFindMatch = false;
						break;
					}
					else{ didNotFindMatch = true; }
				}
				if(next_attr.leaf == false && didNotFindMatch== false)
				{
					nodeFeature_stack.push(next_attr.childNodes.peek());
				}
				else {break;}
			}
			Node deciding_node = next_attr;							// deciding attr node
			for(Integer j : deciding_node.nodeDataset.get(deciding_node.nodeDataset.size()-1))		// make changes in the index // getting class labels
			{
				if(j == 0 )
				{
					class_label_0_count ++;
				}
				else{ class_label_1_count++;}
			}
			if(class_label_0_count > class_label_1_count )
			{
				resultList.add(i, 0);
			}
			else{ resultList.add(i, 1);} 		// creating results list based on classification
		}
		/**
		 *  TESTING ACCURACY AND CONFUSION MATRIX
		 * **/
		Map<String, Double> confusion_matrix = new LinkedHashMap<>();
		confusion_matrix.put("actual 0 predicted 0", 0.0); confusion_matrix.put("actual 1 predicted 1", 0.0); 
		confusion_matrix.put("actual 0 predicted 1", 0.0); confusion_matrix.put("actual 1 predicted 0", 0.0);
		double right =0; double wrong =0; 
		for(int i=0; i< testDataLists.get(0).size(); i++)
		{
			if(testDataLists.get(testDataLists.size() -1).get(i) == resultList.get(i))		// make changes in the index and below as well
			{
				if(testDataLists.get(testDataLists.size() -1).get(i) ==0){ confusion_matrix.put("actual 0 predicted 0", confusion_matrix.get("actual 0 predicted 0")+1 ); }
				else if (testDataLists.get(testDataLists.size() -1).get(i) ==1) { confusion_matrix.put("actual 1 predicted 1", confusion_matrix.get("actual 1 predicted 1")+1 ); }
				right++;
			}
			else 
			{
				if(testDataLists.get(testDataLists.size() -1).get(i) ==0 && resultList.get(i)==1) { confusion_matrix.put("actual 0 predicted 1", confusion_matrix.get("actual 0 predicted 1")+1 ); }
				else if(testDataLists.get(testDataLists.size() -1).get(i) ==1 && resultList.get(i)==0) { confusion_matrix.put("actual 1 predicted 0", confusion_matrix.get("actual 1 predicted 0")+1 ); }
				wrong ++;
			}
		}
		System.out.println("Printing confusion matrix "+ confusion_matrix.toString());
		// ASSUMING 0 IS FALSE OR NO AND 1 VICE VERSA  //
		//System.out.println("FALSE POSITIVE RATE "+ confusion_matrix.get("actual 0 predicted 1") / (confusion_matrix.get("actual 0 predicted 0") + confusion_matrix.get("actual 0 predicted 1"))   );
		//System.out.println("FALSE NEGATIVE RATE "+ confusion_matrix.get("actual 1 predicted 0") / (confusion_matrix.get("actual 1 predicted 0") + confusion_matrix.get("actual 1 predicted 1"))   );
		System.out.println("ACCURACY IS "+ right/(right + wrong));
		testDataLists.clear();    // clearing test data lists


		resultLists.add((ArrayList<Integer>) resultList);			// adding in the aggregated result lists

		/**
		 *  TESTING ACCURACY
		 * **/
	}
	/**######################################################
	 * ###### TESTING SET #######
	 * ######################################################
	 * **/
	/*	Function to print tree	*/
	public void print_decision_tree(Node node)
	{
		Queue<Node> node_list = new LinkedList<>();
		node_list.add(node);
		while(node_list.isEmpty()!= true && node_list.peek().depth <= depth)  
		{
			Queue<Node> tempQueue = new LinkedList<>();
			System.out.println("####### LEVEL " + node_list.peek().depth + " #######" );
			for(Node popped_node : node_list)
			{
				System.out.print("a"+ popped_node.featureName +"("+popped_node.parent.featureName+")"+ "\t");
			}
			System.out.println();
			for(Node popped_node : node_list)
			{
				if(popped_node.childNodes!= null)
				{
					for(Node child_popped_node : popped_node.childNodes)
					{
						System.out.print(child_popped_node.featureName);  				
						if(child_popped_node.childNodes!= null)
						{
							tempQueue.add(child_popped_node.childNodes.peek());
						}
					}
					System.out.print("\t");
				}
			}
			System.out.println();
			node_list.clear();
			node_list.addAll(tempQueue);
			tempQueue.clear();
		}
	}
}
