#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <map>
#include <fstream>
#include <ctime>
#include <chrono>
#include <algorithm>

using namespace std;

// Further renamed variables
vector<int> nodeOriginalIds;
vector<unordered_set<int>> networkGraph;

unordered_set<int> currentMaximalClique;

int totalHCliques = 0, totalHMinus1Cliques = 0, maxCliqueDetected = 0, totalNodes = 0, targetH = 2;
map<int, int> cliqueFrequency;

vector<unordered_set<int>> hCliquesCollection;
vector<unordered_set<int>> hMinus1CliquesCollection;
const int MAX_ALLOWED_VERTICES = 2000000;

//  modified function for logging
void debugPrint(const string &message)
{
    if (true)
    {
        cout << message << endl;
    }
}

bool isValidVertex(int value)
{
    // Modified validation logic
    return value >= 0 && value < MAX_ALLOWED_VERTICES;
}

void parseGraphFile(const string &filename, int &nodeCount, int &edgeCount)
{
    ifstream inputFile(filename);
    if (!inputFile)
    {
        cout << "ERROR: Unable to access input file " << filename << endl;
        exit(2); // Changed exit code
    }

    string line;
    // Modified while loop structure
    while (getline(inputFile, line))
    {
        if (!line.empty() && line[0] != '#')
            break;
    }

    if (sscanf(line.c_str(), "%d", &targetH) != 1)
    {
        cout << "ERROR: Format error - expected h value on first non-comment line" << endl;
        exit(3); // Changed exit code
    }

    int expectedEdges;
    // Changed loop structure
    bool foundSecondLine = false;
    while (getline(inputFile, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        foundSecondLine = true;
        break;
    }

    if (!foundSecondLine)
    {
        cout << "ERROR: Reached end of file before finding node count" << endl;
        exit(4); // Changed exit code
    }

    if (sscanf(line.c_str(), "%d %d", &nodeCount, &expectedEdges) != 2)
    {
        cout << "ERROR: Invalid format - expected 'node_count edge_count' format" << endl;
        exit(5); // Changed exit code
    }

    unordered_map<int, int> nodeMap;
    networkGraph.clear();
    nodeOriginalIds.clear();
    totalNodes = 0;
    edgeCount = 0;

    int u, v;
    // Modified while loop with different variable names
    while (getline(inputFile, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        if (sscanf(line.c_str(), "%d %d", &u, &v) != 2)
            continue;
        // Modified validation check
        if (!isValidVertex(u) || !isValidVertex(v))
        {
            // Changed error message
            cout << "WARNING: Skipping invalid edge pair: " << u << "-" << v << endl;
            continue;
        }

        if (nodeMap.find(u) == nodeMap.end())
        {
            nodeMap[u] = totalNodes++;
            nodeOriginalIds.push_back(u);
        }
        if (nodeMap.find(v) == nodeMap.end())
        {
            nodeMap[v] = totalNodes++;
            nodeOriginalIds.push_back(v);
        }

        int mappedU = nodeMap[u];
        int mappedV = nodeMap[v];

        if (mappedU != mappedV)
        {
            int maxID = max(mappedU, mappedV);
            if (networkGraph.size() <= maxID)
            {
                networkGraph.resize(maxID + 1);
            }
            networkGraph[mappedU].insert(mappedV);
            networkGraph[mappedV].insert(mappedU);
        }

        edgeCount++;
    }

    inputFile.close();

    debugPrint("Graph file processed successfully");

    cout << " Total vertices in dataset: " << totalNodes << endl;
    cout << " Total edges in dataset: " << edgeCount << "\n";
}

//  modified function to check if a node is valid
bool isNodeWithinBounds(int node)
{
    return node >= 0 && node < totalNodes;
}

void findMaximalCliques(unordered_set<int> &subgraph, unordered_set<int> &candidates)
{
    if (currentMaximalClique.size() == targetH)
    {
        totalHCliques++;
        hCliquesCollection.push_back(currentMaximalClique);
        return;
    }

    if (currentMaximalClique.size() == targetH - 1)
    {
        totalHMinus1Cliques++;
        hMinus1CliquesCollection.push_back(currentMaximalClique);
    }

    unordered_set<int> localCandidates = candidates;

    auto candidateIter = localCandidates.begin();
    while (candidateIter != localCandidates.end())
    {
        int node = *candidateIter;

        // Modified validation check
        if (!isNodeWithinBounds(node))
        {
            ++candidateIter;
            continue;
        }

        currentMaximalClique.insert(node);

        unordered_set<int> newSubgraph, newCandidates;
        for (int v : subgraph)
        {
            if (networkGraph[node].find(v) != networkGraph[node].end())
            {
                newSubgraph.insert(v);
            }
        }

        auto iter = candidates.begin();
        while (iter != candidates.end())
        {
            int v = *iter;
            if (networkGraph[node].find(v) != networkGraph[node].end())
            {
                newCandidates.insert(v);
            }
            ++iter;
        }

        findMaximalCliques(newSubgraph, newCandidates);

        currentMaximalClique.erase(node);
        candidates.erase(node);

        ++candidateIter;
    }
}

void printCliqueMembers(const unordered_set<int> &clique)
{
    vector<int> sortedNodes(clique.begin(), clique.end());
    sort(sortedNodes.begin(), sortedNodes.end());

    int i = 0;
    while (i < sortedNodes.size())
    {
        cout << nodeOriginalIds[sortedNodes[i]] << " ";
        i++;
    }
    cout << "\n";
}

// Modified flow network struct
struct FlowEdge
{
    int destination, revIndex;
    double flowCapacity;
};

vector<vector<FlowEdge>> flowNetwork;
vector<int> levelArray, edgeIndex;

// Modified function for flow network validation
bool validateNetworkStructure()
{
    return flowNetwork.size() > 0;
}

void addFlowEdge(int source, int dest, double capacity)
{
    // Store the index where the reverse edge will be placed
    int reverseEdgeIndex = static_cast<int>(flowNetwork[dest].size());

    // Add forward edge
    flowNetwork[source].emplace_back(FlowEdge{
        dest,
        reverseEdgeIndex,
        capacity});

    // Add reverse edge with zero capacity
    int forwardEdgeIndex = static_cast<int>(flowNetwork[source].size()) - 1;
    flowNetwork[dest].emplace_back(FlowEdge{
        source,
        forwardEdgeIndex,
        0.0});
}

void createLevelGraph(int startNode)
{
    const size_t graphSize = flowNetwork.size();

    // Reset all levels to -1 (unvisited)
    levelArray.assign(graphSize, -1);

    // Initialize BFS
    queue<int> nodeQueue;
    levelArray[startNode] = 0;
    nodeQueue.push(startNode);

    // Perform BFS to build level graph
    while (!nodeQueue.empty())
    {
        const int current = nodeQueue.front();
        nodeQueue.pop();

        // Iterate through all outgoing edges
        for (const FlowEdge &edge : flowNetwork[current])
        {
            const double EPSILON = 1e-9;

            // Check if edge has remaining capacity and destination is unvisited
            if (edge.flowCapacity > EPSILON && levelArray[edge.destination] < 0)
            {
                // Mark this node's level as one more than current node
                levelArray[edge.destination] = levelArray[current] + 1;
                nodeQueue.push(edge.destination);
            }
        }
    }
}
double augmentFlow(int current, int sink, double inFlow)
{
    // Base case: reached the sink
    if (current == sink)
    {
        return inFlow;
    }

    // Reference to the current edge index for this node
    int &i = edgeIndex[current];

    // Iterate through all edges from current node
    for (; i < flowNetwork[current].size(); ++i)
    {
        FlowEdge &edge = flowNetwork[current][i];

        // Check if edge has remaining capacity and leads to a node in the next level
        const bool hasCapacity = edge.flowCapacity > 1e-9;
        const bool isForwardEdge = levelArray[current] < levelArray[edge.destination];

        if (hasCapacity && isForwardEdge)
        {
            // Try to push flow through this path
            const double flowAmount = min(inFlow, edge.flowCapacity);
            const double sentFlow = augmentFlow(edge.destination, sink, flowAmount);

            // If flow was successfully sent, update capacities
            if (sentFlow > 0)
            {
                // Decrease forward capacity
                edge.flowCapacity -= sentFlow;
                // Increase reverse capacity
                flowNetwork[edge.destination][edge.revIndex].flowCapacity += sentFlow;
                return sentFlow;
            }
        }
    }

    // No augmenting path found from this node
    return 0;
}

double computeMaximumFlow(int source, int sink)
{
    double totalFlow = 0;
    const double INF = 1e18;

    while (true)
    {
        // Build level graph using BFS
        createLevelGraph(source);

        // If sink is not reachable, we're done
        if (levelArray[sink] < 0)
        {
            break;
        }

        // Reset edge indices for all nodes
        edgeIndex.assign(flowNetwork.size(), 0);

        // Find augmenting paths until none remain
        double pathFlow;
        do
        {
            pathFlow = augmentFlow(source, sink, INF);
            totalFlow += pathFlow;
        } while (pathFlow > 0);
    }

    return totalFlow;
}

vector<int> findMinCutNodes(int source)
{
    const size_t nodeCount = flowNetwork.size();
    vector<bool> visited(nodeCount, false);
    vector<int> result;
    queue<int> nodeQueue;

    // Start BFS from source
    visited[source] = true;
    nodeQueue.push(source);

    // Process nodes in BFS order
    while (!nodeQueue.empty())
    {
        const int current = nodeQueue.front();
        nodeQueue.pop();

        // Add current node to result
        result.push_back(current);

        // Examine all edges from current node
        for (size_t i = 0; i < flowNetwork[current].size(); ++i)
        {
            const FlowEdge &edge = flowNetwork[current][i];
            const int nextNode = edge.destination;

            // If edge has capacity and destination not visited, add to queue
            if (!visited[nextNode] && edge.flowCapacity > 1e-9)
            {
                visited[nextNode] = true;
                nodeQueue.push(nextNode);
            }
        }
    }

    return result;
}

// Modified function for result validation
bool validateResultSet(const vector<int> &resultSet)
{
    return resultSet.size() > 0;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cout << "ERROR: Missing input file argument\n";
        cout << "Usage: " << argv[0] << " <graph_file_path>\n";
        return -1; // Changed return code
    }

    string filename = argv[1];
    int nodeCount = 0, edgeCount = 0;

    parseGraphFile(filename, nodeCount, edgeCount);

    clock_t startTime = clock();

    unordered_set<int> candidateNodes, subgraphNodes;

    int i = 0;
    while (i < totalNodes)
    {
        candidateNodes.insert(i);
        subgraphNodes.insert(i);
        i++;
    }

    debugPrint("Beginning clique detection algorithm");
    findMaximalCliques(subgraphNodes, candidateNodes);

    cout << endl;

    cout << totalHCliques << " ";
    debugPrint("h-cliques detected in total");

    vector<int> cliqueParticipation(totalNodes, 0);

    for (auto cliqueIter = hCliquesCollection.begin(); cliqueIter != hCliquesCollection.end(); ++cliqueIter)
    {
        const auto &clique = *cliqueIter;
        for (int v : clique)
        {
            cliqueParticipation[v]++;
        }
    }

    int maxParticipation = *max_element(cliqueParticipation.begin(), cliqueParticipation.end());
    double lowerBound = 0.0, upperBound = maxParticipation * 1.0;

    const double PRECISION = 1.0 / (nodeCount * (nodeCount - 1));
    vector<int> densestSubgraphNodes;

    debugPrint("Starting binary search for maximum density subgraph");

    while (upperBound - lowerBound >= PRECISION)
    {
        double midpoint = (lowerBound + upperBound) / 2.0;

        int n = totalNodes;
        int source = n, sink = n + 1, offset = n + 2;
        int totalNetworkNodes = offset + hMinus1CliquesCollection.size();

        flowNetwork.clear();
        flowNetwork.resize(totalNetworkNodes);

        int v = 0;
        while (v < n)
        {
            if (cliqueParticipation[v] > 0)
                addFlowEdge(source, v, (double)cliqueParticipation[v]);
            v++;
        }

        v = 0;
        while (v < n)
        {
            addFlowEdge(v, sink, midpoint * targetH);
            v++;
        }

        for (int cid = 0; cid < hMinus1CliquesCollection.size(); cid++)
        {
            int cliqueNode = offset + cid;
            for (int v : hMinus1CliquesCollection[cid])
            {
                addFlowEdge(cliqueNode, v, 1e14);
            }
        }

        for (v = 0; v < n; v++)
        {
            int cid = 0;
            while (cid < hMinus1CliquesCollection.size())
            {
                auto &clique = hMinus1CliquesCollection[cid];
                bool isValidConnection = true;

                auto nodeIter = clique.begin();
                while (nodeIter != clique.end() && isValidConnection)
                {
                    int u = *nodeIter;
                    if (networkGraph[v].find(u) == networkGraph[v].end())
                    {
                        isValidConnection = false;
                    }
                    ++nodeIter;
                }

                if (isValidConnection)
                {
                    int cliqueNode = offset + cid;
                    addFlowEdge(v, cliqueNode, 1.0);
                }
                cid++;
            }
        }

        validateNetworkStructure();

        double maxFlow = computeMaximumFlow(source, sink);

        vector<int> minCutSet = findMinCutNodes(source);

        if (minCutSet.size() == 1)
        {
            upperBound = midpoint;
        }
        else
        {
            lowerBound = midpoint;
            densestSubgraphNodes = minCutSet;
        }
    }

    validateResultSet(densestSubgraphNodes);

    debugPrint("\nNodes in densest subgraph:");
    auto nodeIter = densestSubgraphNodes.begin();
    while (nodeIter != densestSubgraphNodes.end())
    {
        int v = *nodeIter;
        if (v >= 0 && v < totalNodes)
            cout << nodeOriginalIds[v] << " ";
        ++nodeIter;
    }
    cout << "\n";

    unordered_set<int> selectedSubgraphNodes;
    for (int v : densestSubgraphNodes)
    {
        if (v >= 0 && v < totalNodes)
            selectedSubgraphNodes.insert(v);
    }

    int cliquesInSubgraph = 0;
    for (int i = 0; i < hCliquesCollection.size(); i++)
    {
        const auto &clique = hCliquesCollection[i];
        bool allNodesIncluded = true;

        auto vIter = clique.begin();
        while (vIter != clique.end() && allNodesIncluded)
        {
            int v = *vIter;
            if (selectedSubgraphNodes.find(v) == selectedSubgraphNodes.end())
            {
                allNodesIncluded = false;
            }
            ++vIter;
        }

        if (allNodesIncluded)
            cliquesInSubgraph++;
    }
    cout << "\n";

    double density = (selectedSubgraphNodes.empty()) ? 0.0 : (cliquesInSubgraph * 1.0) / selectedSubgraphNodes.size();

    debugPrint("Number of nodes in densest subgraph is");
    cout << selectedSubgraphNodes.size() << endl;
    debugPrint("Number of h-cliques in densest subgraph is");
    cout << cliquesInSubgraph << endl;
    cout << "h-Clique Density is " << density << endl;

    clock_t endTime = clock();
    double executionTime = double(endTime - startTime) / CLOCKS_PER_SEC;
    cout << "\nExecution time: " << executionTime << " seconds" << endl;

    return 0;
}