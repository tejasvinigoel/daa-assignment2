# Implementing Efficient Algorithms for Densest Subgraph Discovery
This repository contains implementations of two algorithms for clique densest subgraph in graphs, along with datasets and their analysis.

| Name | ID | Individual Contributions |
|------|-------|-------|
| Tanisha Agarwal | 2022A7PS0078H | Implementation of Algorithm 4 |
| Thrisha Arrabolu | 2022A7PS0127H | Implementation of Algorithm 4 |
| Snigdha Barui | 2022A7PS0215H | Implementation of Algorithm 1 & Documentation and analysis |
| Tejasvini Goel | 2022A7PS1672H | Implementation of Algorithm 1 & Website Development |
| B.Vaishnavi | 2022A7PS1357H | Implementation of Algorithm 1 & Documentation and analysis |

## Report link:
https://docs.google.com/document/d/1wVR-fJXoiuKwuarOJ8yLDRcS12AT7vNz_sp4gcVc0ss/edit?tab=t.0

## Website link:
https://tejasvinigoel.github.io/daa-assignment2/

## Edited Datasets link:
https://drive.google.com/drive/folders/12K8Z_Hlq1qxXqcx4fFfJFPL4AxPLvz-2

## Code Execution:
- Ensure the datasets are in the same directory as the source code.
- Once all files are downloaded in the local system, on terminal:

## ⚙️ Compilation Instructions(on windows)

Use the following commands to compile the algorithms:

```bash
# Compile algo_1 algorithm
g++ -O2  algo_1.cpp -o algo_1

# Compile algo_4 algorithm
g++ -O2  algo_4.cpp -o algo_4

for datasets, first line is h (clique) value, second line is number of vertices and edges and next lines are the edges.
save the dataset in a text file named input.txt

# Run algorithm 1
./algo_1 input.txt

# Run algorithm 4
./algo_4 input.txt
 
```

## Algorithm Suitability

| Graph Type | Recommended Algorithm |
|------------|------------------------|
| Small graphs (few thousand vertices) | **Algorithm 1 (Exact)** |
| Medium to large dense graphs | **Algorithm 4 (CoreExact)** |
| Very large graphs (millions of vertices, social networks, biological networks) | **Algorithm 4 (CoreExact)** |
| High hhh-clique density subgraph search (e.g., triangle-dense, 4-clique-dense) | **Algorithm 4 (CoreExact)** |

---
## Observations:
We compared Algorithm 1 (Exact) and Algorithm 4 (CoreExact) for Densest Subgraph Discovery (DSD).

Algorithm 1 uses binary search with flow network constructions to find exact solutions but is computationally expensive for large or dense graphs. CoreExact improves on this by applying (k, Ψ)-core decomposition to reduce the search space, significantly speeding up computations while maintaining exactness. It achieves up to four orders of magnitude faster performance than Algorithm 1.

Thus, CoreExact is recommended for large-scale, real-world networks, while Algorithm 1 remains suitable for small or toy datasets. Overall, CoreExact bridges the gap between theoretical accuracy and practical scalability.


  
  
 
