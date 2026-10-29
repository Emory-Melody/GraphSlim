# ogbn-arxiv tradeoff: author reports and GraphSlim

![Author paper points moving to GraphSlim package results on ogbn-arxiv](../figures/arxiv_author_vs_graphslim.svg)

The horizontal axis is graph reduction time; the vertical axis is downstream test
accuracy. Open circles are the authors' reported results and filled circles are
GraphSlim package measurements. Arrows match the number of reduced nodes. The
gray points place 17 methods from the CGC paper on the same axes. The plot has
no error bars.

| Method | Reduced nodes | Author paper: time, accuracy | GraphSlim: time, accuracy |
| --- | ---: | ---: | ---: |
| CGC | 454 | 8.8 s, 66.4% | 27.63 s, 64.27% |
| ClustGDD | 90 | approximately 79 s, 63.7% | 14.92 s, 59.87% |
| GCGP | 909 | 298.4082 s, 65.7% | 4.42 s, 10.63% |

Author values: [CGC Table 3 and Table 4](https://arxiv.org/html/2405.13707),
[ClustGDD Table 3 and Appendix D.1](https://arxiv.org/html/2505.20807), and
[GCGP Table III and Table IV](https://arxiv.org/html/2501.02565). ClustGDD's
paper describes the 90-node arXiv run as taking “around 79s”; its time in the
plot is approximate. CGC's author code maps the paper's 0.25% Arxiv label to a
0.005 fraction of labeled training nodes, matching 454 reduced nodes.
The author points are measurements reported by the original implementations'
papers, rather than new runs of their code on the GraphSlim GPU.

GraphSlim values come from the [frozen ogbn-arxiv result snapshot](../benchmark/results/tradeoff_20261028.json):
three reduction seeds, ten GCN evaluation seeds per reduction, 300 GCN updates,
and the median of three reduction times. The measured GPU was an NVIDIA RTX PRO
6000 Blackwell Server Edition. The frozen scientific source has SHA-256
`66e175c052a27271d5dbda4adb1d587e02fa9ae69c3171d7c78f3f315481dda0`;
these are the package results from that source, not a fresh run of the current
checkout. [Machine-readable points and raw-result hashes](../benchmark/results/arxiv_author_vs_graphslim_20261029.json).

The author papers used their own hardware and GCN settings. The arrows locate
the reported points relative to the package runs; they do not isolate the effect
of GraphSlim code changes. The frozen GCGP package run's 10.63% accuracy is
retained as measured.

Regenerate the figure from the checked-in data with:

```bash
python -m benchmark.plot_arxiv_author_shift
```

With the original local run directory available, add `--collect` to also verify
the raw result files and source archive.
