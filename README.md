# Community Detection in Social Networks using Ant Colony Optimization and Fuzzy Clustering

## Overview

This project implements a hybrid approach for detecting communities in social networks using **Ant Colony Optimization (ACO)** and **Fuzzy Clustering**. The methodology combines nature-inspired optimization techniques with fuzzy logic to identify overlapping community structures. By leveraging **pheromone trails** in ACO and **probabilistic clustering** in FCM, this method provides a **scalable** and **adaptive** way to detect communities within complex networks.

## Authors

- **Sharon David J**
- **Ravulapally Goutham**
- **Indian Institute of Information Technology, Design and Manufacturing, Kancheepuram**
- December 14, 2024

## Problem Definition

Community detection in social networks involves identifying groups of nodes that are more densely connected to each other than to the rest of the network. Challenges include:

- **Handling overlapping communities**: Many nodes belong to multiple communities in real-world networks.
- **Scalability for large networks**: Processing massive networks efficiently remains a significant hurdle.
- **Maintaining accuracy in community assignments**: Ensuring that detected communities reflect real-world relationships accurately.

## Methodology

The approach is divided into four main phases:

1. **Exploration Phase**: 
   - Artificial ants traverse the network, laying pheromone trails to identify potential community structures.
   - Movement is influenced by pheromone intensity and heuristic information.

2. **Construction Phase**: 
   - Communities are formed based on the pheromone levels collected during exploration.
   - Stronger pheromone trails lead to more reliable community assignments.

3. **Local Optimization Phase**: 
   - The community structure is refined by reassigning vertices based on internal and external degrees.
   - Modularity is used as a measure to improve the quality of detected communities.

4. **Fuzzy Clustering**: 
   - A fine-tuning step using **Fuzzy C-Means (FCM)** allows nodes to have probabilistic memberships in multiple communities.
   - This step enhances the flexibility of community assignments and captures overlapping communities.

## Implementation Details

The implementation is done in **C++** with structured data handling. The key components include:

- **Graph Representation**: Adjacency list format with additional pheromone-based edge attributes.
- **Ant Simulation**: Movement of artificial ants based on heuristic probabilities.
- **Community Construction**: Assigning nodes to communities using pheromone levels and connectivity strength.
- **Optimization Techniques**: Modularity maximization for refining community structures and increasing accuracy.
- **Fuzzy Clustering**: Adjusting membership values to accommodate nodes that belong to multiple communities.

### Input Format

The input graph is read from a text file (`graph.txt`) with the following format:

```
<V>  # Number of vertices
<E>  # Number of edges
V1 V2  # Edge between V1 and V2
...
```

### Output

The final community structure is stored in `fcm_results.txt`, which includes:
- Community assignments for each node.
- Membership probabilities for fuzzy clustering.
- Modularity score of the detected communities.

A Python script is provided for visualizing the detected communities.

## Running the Program

### Compilation

To compile the C++ program:

```
g++ -o community_detection main.cpp
```

### Execution

```
./community_detection graph.txt
```

### Visualization

Python scripts are provided to generate visualizations of the detected communities. To visualize the output:

```
python visualize.py fcm_results.txt
```

## Results

The algorithm effectively detects community structures, including overlapping groups, and refines them using fuzzy clustering. The visualization step provides graphical representations of the detected communities. 

Key Observations:
- ACO efficiently identifies **strongly connected** community structures.
- Fuzzy clustering enables **overlapping membership**, making the model adaptable.
- The use of modularity maximization enhances community detection accuracy.

## Limitations and Future Work

- **Graph Type Limitation**: The algorithm currently supports **only undirected graphs**.
- **Scalability Enhancements**: Further optimization is needed for very large networks.
- **Directed Graphs**: Future work will focus on adapting the approach to **directed graphs**.
- **Real-World Applications**: The model can be extended to **biological networks**, **social influence analysis**, and **fraud detection**.

## References

- E. Noveiri, M. Naderan, and S. E. Alavi, "Community detection in social networks using ant colony algorithm and fuzzy clustering," ICCKE, 2015.
- C. Honghao, F. Zuren, R. Zhigang, "Community Detection Using Ant Colony Optimization," IEEE Congress on Evolutionary Computation, 2013.
- Thomas Stutzle and Holger H. Hoos, "Max-min ant system," Future Generation Computer Systems, 2000.

