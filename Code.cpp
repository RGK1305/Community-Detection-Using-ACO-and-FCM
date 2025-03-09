#include <iostream>
#include <vector>
#include <list>
#include <set>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <fstream>
#include <string>

#define ALPHA 1.0  // Importance of pheromone
#define BETA 2.0   // Importance of heuristic information
#define EVAPORATION 0.93  // Pheromone evaporation rate (7% evaporation)
#define UPDATE_INTERVAL 3 // Interval for pheromone updates based on traversal
using namespace std;
struct Edge {
    int src, dest;
    double pheromone;
    int traversals;
};

struct CommunityEdge {
    int src, dest;
    int edge_count;
    double pheromone;
};

struct Graph {
    int V, E;
     vector< list<int>> adjList; // Adjacency list
     vector<Edge> edges; // List of edges
     vector< vector<int>> adjMatrix; // Adjacency matrix
     vector< vector<double>> pheromoneLevels; // Pheromone levels on edges
     vector<int> degrees; // Degree of each vertex
    int totalEdges; // Total number of edges
};

struct Ant {
    int current_vertex;
     vector<int> tabu_list;
};

// Function prototypes
void initialize_graph(Graph &graph, int V, int E);
void initialize_ants( vector<Ant> &ants, int V);
void print_graph(const Graph &graph);
void explore_graph(Graph &graph,  vector<Ant> &ants);
int select_next_vertex(const Graph &graph, Ant &ant);
void update_pheromones(Graph &graph);
double calculate_modularity_gain(const Graph &graph, int vertex, int neighbor);
void construct_communities(Graph &graph,  vector<int> &communities,  vector<CommunityEdge> &community_edges, int &num_communities);
void local_optimization(Graph &graph,  vector<int> &communities,  vector<CommunityEdge> &community_edges);
void cleanup_empty_communities( vector<int> &communities,  vector<CommunityEdge> &community_edges);
void fuzzy_c_means(Graph &graph,  vector<int> &communities, int c, double m = 2.0, double epsilon = 1e-5,int max_iters=1000,const  string &filename="fcm_results.txt");

int main() {
      string inp;
      ifstream file;
    file.open("graph.txt");
    file >>inp;
    int V = stoi(inp); // Number of vertices
    file >>inp;
    int E = stoi(inp); // Number of edges
	file.close();
    srand(time(0)); // Seed for random number generation

    Graph graph;
    initialize_graph(graph, V, E);

     vector<Ant> ants(V);
    initialize_ants(ants, V);

    // Exploration phase
    explore_graph(graph, ants);
  

    // Construction phase
     vector<int> communities(graph.V, -1);
     vector<CommunityEdge> community_edges;
    int num_communities = 0;
    construct_communities(graph, communities, community_edges, num_communities); // Pass num_communities by reference

     cout << "\nConstruction Phase Output:\n";
    for (int i = 0; i < communities.size(); ++i) {
         cout << "Vertex " << i << " is in community " << communities[i] <<  endl;
    }

    // Local optimization phase (called before fuzzy clustering)
    local_optimization(graph, communities, community_edges);

     cout << "\nLocal Optimization Phase Output:\n";
    for (int i = 0; i < communities.size(); ++i) {
         cout << "Vertex " << i << " Optimized Community: " << communities[i] <<  endl;
    }

    // Update the number of communities after local optimization
     set<int> unique_communities(communities.begin(), communities.end());
    num_communities = unique_communities.size(); // Recalculate the number of distinct communities
     cout << "\nUpdated Number of Communities After Local Optimization: " << num_communities <<  endl;

    // Optionally, cleanup empty communities and edges
    cleanup_empty_communities(communities, community_edges);

     cout << "\nAfter Cleanup (Empty Communities and Edges Removed):\n";
    for (int i = 0; i < communities.size(); ++i) {
         cout << "Vertex " << i << " is in community " << communities[i] <<  endl;
    }

     cout << "\nCommunity Edges After Cleanup:\n";
    for (const auto &ce : community_edges) {
         cout << "Community Edge (" << ce.src << ", " << ce.dest << ") - Count: " << ce.edge_count << ", Pheromone: " << ce.pheromone <<  endl;
    }

    // Call Fuzzy C-Means to refine community assignments
    fuzzy_c_means(graph, communities, num_communities); // Pass the updated number of communities

     cout << "\nFuzzy C-Means Clustering Output:\n";
    for (int i = 0; i < communities.size(); ++i) {
         cout << "Vertex " << i << " Fuzzy Community: " << communities[i] <<  endl;
    }

    return 0;
}



// Function implementations
void initialize_graph(Graph &graph, int V, int E) {
    graph.V = V;
    graph.E = E;
    graph.adjList.resize(V);
    graph.adjMatrix.resize(V,  vector<int>(V, 0));
    graph.pheromoneLevels.resize(V,  vector<double>(V, 1.0));
    graph.degrees.resize(V, 0);
    graph.totalEdges = E;
     set< pair<int, int>> edgeSet;
     string inp;
       ifstream file;
    file.open("graph.txt");
    int src=0;
    int dest=0;
    // Add edges and initialize pheromones
    while (graph.edges.size() < E) {
        file>>inp;
        src=stoi(inp);
        file>>inp;
        dest=stoi(inp);
        
        if (src != dest && edgeSet.find({src, dest}) == edgeSet.end() && edgeSet.find({dest, src}) == edgeSet.end()) {
            double pheromone = 1.0;
            graph.edges.push_back({src, dest, pheromone, 0});
            graph.adjList[src].push_back(dest);
            graph.adjList[dest].push_back(src);
            graph.adjMatrix[src][dest] = 1;
            graph.adjMatrix[dest][src] = 1;
            graph.degrees[src]++;
            graph.degrees[dest]++;
            edgeSet.insert({src, dest});
        }
    }
    file.close();
}

void initialize_ants( vector<Ant> &ants, int V) {
    for (int i = 0; i < V; ++i) {
        ants[i].current_vertex = i;
    }
}

void print_graph(const Graph &graph) {
     cout << "Graph representation (Adjacency List):\n";
    for (int v = 0; v < graph.V; ++v) {
         cout << v << ": ";
        for (auto &adj : graph.adjList[v]) {
             cout << adj << " ";
        }
         cout <<  endl;
    }

     cout << "\nEdges with Pheromone Levels:\n";
    for (auto &edge : graph.edges) {
         cout << "Edge (" << edge.src << ", " << edge.dest << ") - Pheromone: " << graph.pheromoneLevels[edge.src][edge.dest] << ", Traversals: " << edge.traversals <<  endl;
    }
}

double calculate_modularity_gain(const Graph &graph, int vertex, int neighbor) {
    double m = graph.totalEdges;
    double sum_degrees = 0.0;
    for (int i = 0; i < graph.V; ++i) {
        sum_degrees += graph.degrees[i];
    }
    double expected_edges = (graph.degrees[vertex] * graph.degrees[neighbor]) / (2 * m);
    double actual_edges = graph.adjMatrix[vertex][neighbor];
    double modularity_gain = (actual_edges - expected_edges) / (2 * m);
    return modularity_gain;
}

void explore_graph(Graph &graph,  vector<Ant> &ants) {
    int max_steps = 15; // Maximum number of steps for exploration
    int max_degree=INT_MIN;
    for (int degree : graph.degrees) {
            if(max_degree<degree){
            		max_degree=degree;
            }
    }
    //  cout<<"Maximum degree = "<<max_degree;
    for (int step = 0; step < max_steps; ++step) {
          cout << "\nExploration Phase Output "<<step<< ":\n";
        for (auto &ant : ants) {
            int next_vertex = select_next_vertex(graph, ant);
            
            for (auto &edge : graph.edges) {
                if ((edge.src == ant.current_vertex && edge.dest == next_vertex) || (edge.src == next_vertex && edge.dest == ant.current_vertex)) {
                    edge.traversals++;
                    break;
                }
            }
              cout<< endl<<"Ant in vertex "<<ant.current_vertex;
            ant.current_vertex = next_vertex;
            ant.tabu_list.push_back(ant.current_vertex);
              cout<<" is moved to next vertex "<<ant.current_vertex;
            
            if (ant.tabu_list.size() > max_degree) { // Limit the size of the tabu list
                ant.tabu_list.erase(ant.tabu_list.begin());
            }
        }
        // Evaporate pheromones each iteration
        for (int i = 0; i < graph.V; ++i) {
            for (int j = 0; j < graph.V; ++j) {
                graph.pheromoneLevels[i][j] *= EVAPORATION;
            }
        }
        // Update pheromones based on traversals at specified intervals
        if ((step + 1) % UPDATE_INTERVAL == 0) {
            update_pheromones(graph);
        }
        //  update_pheromones(graph);

     cout<< endl;
    print_graph(graph);
         initialize_ants(ants, graph.V);
    }
    
}

int select_next_vertex(const Graph &graph, Ant &ant) {
    int current = ant.current_vertex;
    double max_prob = -1.0;
    int best_neighbor = current;
    
    // Calculate the probabilities and select the best neighbor
    for (int neighbor : graph.adjList[current]) {
        if ( find(ant.tabu_list.begin(), ant.tabu_list.end(), neighbor) == ant.tabu_list.end()) {
            double deltaQ = calculate_modularity_gain(graph, current, neighbor);
            double eta = (deltaQ > graph.pheromoneLevels[current][neighbor]) ? deltaQ + graph.pheromoneLevels[current][neighbor] : 0;
            double prob =  pow(graph.pheromoneLevels[current][neighbor], ALPHA) *  pow(eta, BETA);
            if (prob > max_prob) {
                max_prob = prob;
                best_neighbor = neighbor;
            }
        }
    }
    return best_neighbor;
}

void update_pheromones(Graph &graph) {
    // Update pheromones based on edge traversals
    for (auto &edge : graph.edges) {
        if (edge.traversals > 0) {
            graph.pheromoneLevels[edge.src][edge.dest] += edge.traversals;
            graph.pheromoneLevels[edge.dest][edge.src] = graph.pheromoneLevels[edge.src][edge.dest]; // Ensure symmetry
            edge.pheromone = graph.pheromoneLevels[edge.src][edge.dest]; // Update the edge's pheromone level
            edge.traversals = 0; // Reset the traversal count after updating pheromones
        }
    }
}

void construct_communities(Graph &graph,  vector<int> &communities,  vector<CommunityEdge> &community_edges, int &num_communities) {
    // Sort edges in descending order of pheromone levels
     sort(graph.edges.begin(), graph.edges.end(), [](const Edge &a, const Edge &b) {
        return a.pheromone > b.pheromone;
    });

    // Print sorted edges to verify order
     cout << "\nSorted Edges by Pheromone Levels:\n";
    for (const auto &edge : graph.edges) {
         cout << "Edge (" << edge.src << ", " << edge.dest << ") - Pheromone: " << edge.pheromone <<  endl;
    }

    int community_id = 0;
    for (const auto &edge : graph.edges) {
        int src = edge.src;
        int dest = edge.dest;

        // Case 1: Both vertices are not assigned to any community, create a new community
        if (communities[src] == -1 && communities[dest] == -1) {
            communities[src] = community_id;
            communities[dest] = community_id;
             cout << "Assigning vertices " << src << " and " << dest << " to new community " << community_id <<  endl;
            community_id++;
        }
        // Case 2: Source vertex is assigned, but destination vertex is not
        else if (communities[src] != -1 && communities[dest] == -1) {
            communities[dest] = communities[src];
             cout << "Assigning vertex " << dest << " to community " << communities[src] <<  endl;
        }
        // Case 3: Destination vertex is assigned, but source vertex is not
        else if (communities[src] == -1 && communities[dest] != -1) {
            communities[src] = communities[dest];
             cout << "Assigning vertex " << src << " to community " << communities[dest] <<  endl;
        }
        // Case 4: Both vertices are assigned to different communities
        else if (communities[src] != communities[dest]) {
            int src_community = communities[src];
            int dest_community = communities[dest];
            bool edge_found = false;
            for (auto &ce : community_edges) {
                if ((ce.src == src_community && ce.dest == dest_community) || (ce.src == dest_community && ce.dest == src_community)) {
                    ce.edge_count++;
                    ce.pheromone += edge.pheromone;
                     cout << "Incrementing edge count between community " << src_community << " and community " << dest_community <<  endl;
                    edge_found = true;
                    break;
                }
            }
            if (!edge_found) {
                community_edges.push_back({src_community, dest_community, 1, edge.pheromone});
                 cout << "Adding edge between community " << src_community << " and community " << dest_community <<  endl;
            }
        }
    }

    // Handle disconnected (isolated) vertices
    for (int v = 0; v < graph.V; ++v) {
        // If the vertex is not assigned to any community and has no neighbors (isolated)
        if (communities[v] == -1 && graph.adjList[v].empty()) {
            communities[v] = community_id;
             cout << "Assigning isolated vertex " << v << " to new community " << community_id <<  endl;
            community_id++; // Create a new community for isolated vertex
        }
    }

    // Update the number of communities
     set<int> unique_communities(communities.begin(), communities.end());
    num_communities = unique_communities.size(); // Number of distinct communities

     cout << "Number of communities detected: " << num_communities <<  endl;
}



void cleanup_empty_communities( vector<int> &communities,  vector<CommunityEdge> &community_edges) {
    // Remove empty communities and their edges
     set<int> non_empty_communities;
    for (int community : communities) {
        non_empty_communities.insert(community);
    }

    // Remove edges that involve empty communities
    community_edges.erase(
         remove_if(community_edges.begin(), community_edges.end(), [&](const CommunityEdge &ce) {
            return non_empty_communities.find(ce.src) == non_empty_communities.end() ||
                   non_empty_communities.find(ce.dest) == non_empty_communities.end();
        }),
        community_edges.end()
    );

    // Update the communities by removing empty ones
    for (int i = 0; i < communities.size(); ++i) {
        if (non_empty_communities.find(communities[i]) == non_empty_communities.end()) {
            communities[i] = -1; // Assign an invalid community ID to removed communities
        }
    }
}

void local_optimization(Graph &graph,  vector<int> &communities,  vector<CommunityEdge> &community_edges) {
    // Step 1: Calculate internal and external degrees
     vector< pair<int, int>> max_degrees(graph.V); // (vertex, max_external_degree)
    for (int v = 0; v < graph.V; ++v) {
        int current_community = communities[v];
        int internal_degree = 0;
         vector<int> external_degrees(graph.V, 0); // Degrees towards each community

        // Step 1a: Calculate internal and external degrees for each vertex
        for (int neighbor : graph.adjList[v]) {
            if (communities[neighbor] == current_community) {
                internal_degree++;
            } else {
                external_degrees[communities[neighbor]]++;
            }
        }

        int max_external_degree = * max_element(external_degrees.begin(), external_degrees.end());
        max_degrees[v] = {v,  max(internal_degree, max_external_degree)};
    }

    // Step 2: Sort vertices by their maximum external degree
     sort(max_degrees.begin(), max_degrees.end(), [](const  pair<int, int> &a, const  pair<int, int> &b) {
        return a.second > b.second;  // Sort by max external degree
    });

    // Step 3: Reassign communities based on the calculated degrees
    for (const auto &vertex_degree : max_degrees) {
        int v = vertex_degree.first;
        int current_community = communities[v];
        int internal_degree = 0;
         vector<int> external_degrees(graph.V, 0);

        // Recalculate internal and external degrees for the vertex
        for (int neighbor : graph.adjList[v]) {
            if (communities[neighbor] == current_community) {
                internal_degree++;
            } else {
                external_degrees[communities[neighbor]]++;
            }
        }

        int max_external_degree = * max_element(external_degrees.begin(), external_degrees.end());
        if (max_external_degree > internal_degree) {
            // Step 4: Move the vertex to the community with the maximum external degree
            int new_community =  distance(external_degrees.begin(),  max_element(external_degrees.begin(), external_degrees.end()));
            communities[v] = new_community;
             cout << "Moving vertex " << v << " from community " << current_community << " to community " << new_community <<  endl;

            // Step 5: Update the edge counts and pheromone levels
            for (auto &edge : graph.edges) {
                if ((edge.src == v && communities[edge.dest] != current_community) || (edge.dest == v && communities[edge.src] != current_community)) {
                    // Check if edge between the communities exists
                    if (communities[edge.src] != communities[edge.dest]) {
                        // Update the pheromone levels of the edge
                        edge.pheromone += 0.1; // Increment pheromone on edge
                        // Update the edge count between the communities
                        bool edge_found = false;
                        for (auto &ce : community_edges) {
                            if ((ce.src == communities[edge.src] && ce.dest == communities[edge.dest]) || 
                                (ce.src == communities[edge.dest] && ce.dest == communities[edge.src])) {
                                ce.edge_count++;
                                edge_found = true;
                                break;
                            }
                        }
                        if (!edge_found) {
                            community_edges.push_back({communities[edge.src], communities[edge.dest], 1, edge.pheromone});
                        }
                    }
                }
            }
        }
    }
}

void save_results(const  string &filename, const  vector<int> &initial_communities, const  vector<int> &final_communities, const  vector< vector<double>> &U, const  vector< vector<double>> &V, const  vector< vector<int>> &adjMatrix) {
     ofstream file(filename);

    if (file.is_open()) {
        // Save adjacency matrix
        file << "Adjacency Matrix:\n";
        for (const auto& row : adjMatrix) {
            for (double value : row) {
                file << value << " ";
            }
            file << "\n";
        }
        file << "\n";

        // Save initial communities (before clustering)
        file << "Initial Communities (Before Fuzzy Clustering):\n";
        for (int community : initial_communities) {
            file << community << " ";
        }
        file << "\n\n";

        // Save final communities (after fuzzy clustering)
        file << "Final Communities (After Fuzzy Clustering):\n";
        for (int community : final_communities) {
            file << community << " ";
        }
        file << "\n\n";

        // Save membership matrix
        file << "Membership Matrix (U):\n";
        for (const auto& row : U) {
            for (double value : row) {
                file << value << " ";
            }
            file << "\n";
        }
        file << "\n";

        // Save cluster centers
        file << "Cluster Centers (V):\n";
        for (const auto& row : V) {
            for (double value : row) {
                file << value << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
         cerr << "Unable to open file: " << filename <<  endl;
    }
}



// Helper function to calculate squared Euclidean distance
double squared_euclidean_distance(const  vector<int>& a, const  vector<double>& b) {
    double dist = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dist +=  pow(a[i] - b[i], 2);
    }
    return dist;
}


// Fuzzy C-Means Algorithm
void fuzzy_c_means(Graph &graph,  vector<int> &communities, int c, double m, double epsilon, int max_iters, const  string &filename) {
    int n = graph.V; // Number of vertices
     vector< vector<double>> U(n,  vector<double>(c, 0.0)); // Membership matrix
     vector< vector<double>> V(c,  vector<double>(n, 0.0)); // Cluster centers
     vector<int> initial_communities = communities;

    // Step 1: Randomly initialize cluster centers (V)
     set<int> chosen_centers;
    srand(time(NULL)); // Ensure randomness is seeded
    for (int i = 0; i < c; ++i) {
        int center;
        do {
            center = rand() % n; // Random vertex
        } while (chosen_centers.find(center) != chosen_centers.end()); // Ensure not already selected

        chosen_centers.insert(center);

        // Initialize cluster centers using the selected vertex's row from the adjacency matrix
        for (int j = 0; j < n; ++j) {
            V[i][j] = graph.adjMatrix[center][j];
        }
    }

    int iteration = 0;
    double prev_J =  numeric_limits<double>::infinity();
    while (iteration < max_iters) {
        // Step 2: Update membership matrix U
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < c; ++j) {
                double dist_ij = squared_euclidean_distance(graph.adjMatrix[i], V[j]);
                if (dist_ij == 0) dist_ij = 1e-6; // Avoiding division by zero

                double sum = 0.0;
                for (int k = 0; k < c; ++k) {
                    double dist_ik = squared_euclidean_distance(graph.adjMatrix[i], V[k]);
                    if (dist_ik == 0) dist_ik = 1e-6;
                    sum +=  pow(dist_ij / dist_ik, 1.0 / (m - 1));
                }
                U[i][j] = 1.0 / sum;
            }
        }

        // Step 3: Update cluster centers V
        for (int j = 0; j < c; ++j) {
            for (int k = 0; k < n; ++k) {
                double numerator = 0.0, denominator = 0.0;
                for (int i = 0; i < n; ++i) {
                    double weight =  pow(U[i][j], m);
                    numerator += weight * graph.adjMatrix[i][k];
                    denominator += weight;
                }
                V[j][k] = numerator / denominator;  // Update cluster center
            }
        }

        // Step 4: Compute objective function J(U, V)
        double J = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < c; ++j) {
                double dist = squared_euclidean_distance(graph.adjMatrix[i], V[j]);
                J +=  pow(U[i][j], m) * dist;
            }
        }

        // Check for convergence
        if ( abs(J - prev_J) < epsilon) break;
        prev_J = J;
        iteration++;
    }

    // Assign community based on the highest membership value
    for (int i = 0; i < n; ++i) {
        communities[i] =  distance(U[i].begin(),  max_element(U[i].begin(), U[i].end()));
    }

    // Save results to file, including initial and final communities
    save_results(filename, initial_communities, communities, U, V, graph.adjMatrix);
}






