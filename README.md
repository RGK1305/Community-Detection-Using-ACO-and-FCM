# Community Detection in Social Networks using Ant Colony Optimization and Fuzzy Clustering

## Overview

This project implements a hybrid approach for detecting communities in social networks using **Ant Colony Optimization (ACO)** and **Fuzzy Clustering**. The methodology combines nature-inspired optimization techniques with fuzzy logic to identify overlapping community structures.

## Authors

- **Sharon David J**
- **Ravulapally Goutham**
- Indian Institute of Information Technology, Design and Manufacturing, Kancheepuram
- December 14, 2024

## Problem Definition

Community detection in social networks involves identifying groups of nodes that are more densely connected to each other than to the rest of the network. Challenges include:

- Handling overlapping communities
- Scalability for large networks
- Maintaining accuracy in community assignments

## Methodology

The approach is divided into four main phases:

1. **Exploration Phase**: Artificial ants traverse the network, laying pheromone trails to identify potential community structures.
2. **Construction Phase**: Communities are formed based on the pheromone levels collected during exploration.
3. **Local Optimization Phase**: The community structure is optimized by reassigning vertices based on internal and external degrees.
4. **Fuzzy Clustering**: A fine-tuning step using **Fuzzy C-Means (FCM)** allows nodes to have probabilistic memberships in multiple communities.

## Implementation Details

The implementation is done in **C++** with structured data handling. The key components include:

- **Graph Representation**: Adjacency list format with additional pheromone-based edge attributes.
- **Ant Simulation**: Movement of artificial ants based on heuristic probabilities.
- **Community Construction**: Assigning nodes to communities using pheromone levels.
- **Optimization Techniques**: Modularity maximization for refining community structures.
- **Fuzzy Clustering**: Adjusting membership values for overlapping community assignments.

### Input Format

The input graph is read from a text file (`graph.txt`) with the following format:

```
<V>  # Number of vertices
<E>  # Number of edges
V1 V2  # Edge between V1 and V2
...
```

### Output

The final community structure is stored in `fcm_results.txt` and visualized using Python.

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

## Limitations and Future Work

- Currently, the algorithm supports **only undirected graphs**.
- Future work will focus on adapting the approach to **directed graphs**.
- Enhancements to improve scalability and execution efficiency for very large networks.

## References

- E. Noveiri, M. Naderan, and S. E. Alavi, "Community detection in social networks using ant colony algorithm and fuzzy clustering," ICCKE, 2015.
- C. Honghao, F. Zuren, R. Zhigang, "Community Detection Using Ant Colony Optimization," IEEE Congress on Evolutionary Computation, 2013.
- Thomas Stutzle and Holger H. Hoos, "Max-min ant system," Future Generation Computer Systems, 2000.

