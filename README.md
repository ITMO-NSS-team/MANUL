<!-- 
---
mathjax: true
---
-->


# Manifold Und Learning - MANUL

**MANUL** - tool for extracting topology from data as a graph structure and associated model regularization. Application employs novel approach to build the neighborhood graph in the initial feature space.

Evolutionary algorithm extracts geometry and topology from the data, using the specific machine learning model. It is used as alternative to Euclidean metric for graph building with further graph distillation to avoid unnecessarily complex structures.

![Logo of MANUL tools](media/logo2.png)

## Background

In that tool data represents as the graph $G_n = (X, W)$, where vertices $X = (x_1, ..., x_n)$ are the data records (points) and $W_{ij}$ is the distance between two data points.

![The scheme with transition from data points with features to topologies structure data](media/img/ds_to_graph_scheme.png)

As background for our method, we will use manifold regularization. It allows one to train a smooth machine-learning model on a found manifold. To formulate the neighborhood graph learning problem, the manifold regularization formulation could be extended to:




$$L^* = \min \limits_{G_n \in \mathcal{G}} \left[\min \limits_{f \in H_k} \mathcal{L}(f)+\lambda(f^T l(G_n) f) \right]$$

