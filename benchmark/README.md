[//]: # (# Preparation)

# GC4NC: A Benchmark Framework for Graph Condensation on Node Classification with New Insights
 
[[Paper Link]](https://arxiv.org/pdf/2406.16715) 

Graph condensation (GC) is an emerging technique designed to learn a significantly smaller graph that retains the essential information of the original graph.  Despite the rapid development of GC methods, a systematic evaluation framework remains absent, which is necessary to clarify the critical designs for particular evaluative aspects. Furthermore, several meaningful questions have not been investigated, such as whether GC inherently preserves certain graph properties and offers robustness even without targeted design efforts. Here, we introduce GC-Bench, a comprehensive framework to evaluate recent GC methods across multiple dimensions and generate new insights. Our experimental findings provide deeper insights into the GC process and the characteristics of condensed graphs, guiding future efforts in enhancing performance and exploring new applications.




## Requirements

Please see `requirements.txt`.

<!--## Download Datasets

For cora, citeseer, flickr and reddit, the pyg code will directly download them.
For arxiv, we use the datasets provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT). Our code will also automatically download it.-->


[//]: # (# Abstract)

[//]: # ()

[//]: # (Graph reduction for all graph algorithms especially for graph neural networks &#40;GNNs&#41;.)

[//]: # (This package aims to reduce the large, original graph into a small, synthetic and highly-informative graph.)

[//]: # ()

[//]: # (# Features)

[//]: # (* Covering 3 mainstream reduction strategies: Sparsificaiton, Coarsening and Condensation)

[//]: # (* Unified test tools for easily producing benchmarks)

## Benchmark Reproduction

All the scripts are in `benchmark` folder.

For Table 1 7 8, use `sh performacne.sh`.

For Figure 3, use `sh scalability.sh`.

For Figure 4, use `sh data_initialization.sh`.

For Figure 5 9, use `sh transferability.sh`.

For Table 2, use `sh nas.sh`.

For Table 3 10 11, use `sh graph_property_preservation.sh`.

For Table 4 12, use `sh robustness.sh`.
