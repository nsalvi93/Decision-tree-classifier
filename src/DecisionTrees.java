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
	List<ArrayList<Integer>> dataLists = new ArrayList<ArrayList<Integer>>(); 		
	List<ArrayList<Integer>> testDataLists = new ArrayList<ArrayList<Integer>>();
	List<Integer> featureQueue = new LinkedList<>();		// total no. features in dataset
	static Entropy entropyUtility = new Entropy();			// global object of Entropy class to calculate gain and entropy
	int depth;											// global variable for depth
	public static void main(String[] args) 
	{
		DecisionTrees tree = new DecisionTrees();
		String filename = args[0];								// getting train_data filename
		tree.depth = Integer.parseInt(args[2]);					// getting depth
		List<String> recordList = tree.readFile(filename);
		if(filename.equals("owndataset.txt"))
		{
			tree.create_featurelist_owndataset(recordList, tree.dataLists);
		}
		else tree.createFeatureList(recordList, tree.dataLists);
		tree.buildTree();											// builds decision tree on training data
		tree.testdata_traversal(args[1]);									// tests on test data
		tree.print_decision_tree(tree.rootNode);					// prints the decision tree. Comment if not needed
	}
	/* Function to construct tree */
	public void buildTree()
	{
		entropyUtility.calculatePriorEntropy(dataLists);		// Function to calculate prior entropy
		decideRootNode();										// Function to decide the root feature
		Queue<Node> rootChildNodes = new LinkedList<>();
		rootChildNodes.addAll(rootNode.childNodes);
		/* Builds the tree in BFS manner */
		while(!rootChildNodes.isEmpty() && rootChildNodes.peek().depth<= depth)  
		{
			Node nodeChild = rootChildNodes.poll();
			splitDataSet(nodeChild.parent.nodeDataset, nodeChild.parent.featureName, nodeChild);		// splits the dataset
			Map<Integer, Map<Integer, double[]>> emptyFeatureAttrCountMap = getFeatureValues(featureQueue, nodeChild.nodeDataset);  // function to get all unique feature values
			Map<Integer, Map<Integer, double[]>> filledFeatureAttrCountMap = calculateBestFeature(nodeChild.nodeDataset, emptyFeatureAttrCountMap); // gets the count of the above values
			entropyUtility.calculateTotalEntropy(filledFeatureAttrCountMap, nodeChild.nodeDataset.get(0).size());    // calculates entropy for all features
			if(entropyUtility.entropyMap.values().stream().mapToDouble(Double::doubleValue).sum() == 0)
			{
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
		} 
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
		for(int i=0; i< featureQueue.size()+1; i++)				  						// + 1 since we are considering features + class label list				
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
	}
	/*
	 * Function that decides the rootNode
	 * */
	public void decideRootNode() 
	{
		Map<Integer, Map<Integer, double[]>> emptyFeatureAttrCountMap = getFeatureValues(featureQueue, dataLists);
		Map<Integer, Map<Integer, double[]>> filledFeatureAttrCountMap = calculateBestFeature(dataLists, emptyFeatureAttrCountMap);
		entropyUtility.calculateTotalEntropy(filledFeatureAttrCountMap, dataLists.get(0).size());
		entropyUtility.calculateInfoGain(entropyUtility.priorEntropy);
		double[] array = entropyUtility.returnHighestInfoGain();
		int rootFeature = (int) array[1];
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
				if(dataLists2.get(0).get(i)== 1)
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
		for(Integer columnIndex : featureQueue2)
		{
			Map<Integer, double[]> attrCountMap = new LinkedHashMap<>();
			Set<Integer> uniqueSet = dataLists2.get(columnIndex).stream().collect(Collectors.toSet());
			for(Integer featureAttrs : uniqueSet){attrCountMap.put(featureAttrs,new double[]{0,0});}
			featureAttrCountMap.put(columnIndex, attrCountMap);
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
			for(int j=0; j< dataLists2.size(); j++ )
			{
				System.out.print(dataLists2.get(j).get(i)+ " ");
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
		return recordList;
	}
	/*
	 * Function to decide the no. of features
	 * */
	public void createFeatureList(List<String> recordList, List<ArrayList<Integer>> dataLists2)
	{
		int featureNo = recordList.get(0).trim().split("\\s+").length;   // getting first line of dataset for features
		for(int i=1; i< featureNo-1; i++ )
		{
			featureQueue.add(i);
		}
		for(int i=0; i< featureNo-1; i++)								
		{
			dataLists2.add(new ArrayList<Integer>());
		}
		for(String record : recordList)
		{
			String [] parseRecord = record.trim().split("\\s+");
			for(int i=0; i< parseRecord.length-1; i++)
			{
				dataLists2.get(i).add(Integer.valueOf(parseRecord[i]));  
			}
		}
	}
	/*For my own dataset*/
	public void create_featurelist_owndataset(List<String> own_dataset_recordList, List<ArrayList<Integer>> dataLists2)
	{
		int featureNo = own_dataset_recordList.get(0).trim().split(",").length;
		for(int i=1; i< featureNo; i++ )
		{
			featureQueue.add(i);
		}
		for(int i=0; i< featureNo; i++)								
		{
			dataLists2.add(new ArrayList<Integer>());
		}
		for(String record : own_dataset_recordList)
		{
			String [] parseRecord = record.trim().split(",");
			for(int i=0; i< parseRecord.length; i++)
			{
				dataLists2.get(i).add(Integer.valueOf(parseRecord[i]));  
			}
		}
	}
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
		if(test_data_filename.equals("owndataset_test.txt"))
		{
			create_featurelist_owndataset(recordList, testDataLists);
		}
		else
			createFeatureList(recordList, testDataLists);
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
			for(Integer j : deciding_node.nodeDataset.get(0))
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
			if(testDataLists.get(0).get(i) == resultList.get(i))
			{
				if(testDataLists.get(0).get(i) ==0){ confusion_matrix.put("actual 0 predicted 0", confusion_matrix.get("actual 0 predicted 0")+1 ); }
				else if (testDataLists.get(0).get(i) ==1) { confusion_matrix.put("actual 1 predicted 1", confusion_matrix.get("actual 1 predicted 1")+1 ); }
				right++;
			}
			else 
			{
				if(testDataLists.get(0).get(i) ==0 && resultList.get(i)==1) { confusion_matrix.put("actual 0 predicted 1", confusion_matrix.get("actual 0 predicted 1")+1 ); }
				else if(testDataLists.get(0).get(i) ==1 && resultList.get(i)==0) { confusion_matrix.put("actual 1 predicted 0", confusion_matrix.get("actual 1 predicted 0")+1 ); }
				wrong ++;
			}
		}
		System.out.println("Printing confusion matrix "+ confusion_matrix.toString());
		// ASSUMING 0 IS FALSE OR NO AND 1 VICE VERSA  //
		//System.out.println("FALSE POSITIVE RATE "+ confusion_matrix.get("actual 0 predicted 1") / (confusion_matrix.get("actual 0 predicted 0") + confusion_matrix.get("actual 0 predicted 1"))   );
		//System.out.println("FALSE NEGATIVE RATE "+ confusion_matrix.get("actual 1 predicted 0") / (confusion_matrix.get("actual 1 predicted 0") + confusion_matrix.get("actual 1 predicted 1"))   );
		System.out.println("ACCURACY IS "+ right/(right + wrong));
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
