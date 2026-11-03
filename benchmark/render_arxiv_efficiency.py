"""Compose one matched GECC/CGC/ClustGDD ogbn-arxiv comparison block."""

import json
from pathlib import Path

from benchmark.plot_clustering_arxiv import METHODS, NAMES, REVIEWED, ROOT, STEM, reviewed_data, sha


def speed_figure(language):
    data = json.loads(REVIEWED.read_text())
    hashes = data.pop("assets_sha256")
    assert data == reviewed_data()
    assert all(sha(Path(str(STEM) + ext)) == digest for ext, digest in hashes.items())
    ratios = {
        method: [p["relative_time"] for p in data["pairs"] if p["method"] == method]
        for method in METHODS
    }
    ranges = {method: f"{min(values):.2f}–{max(values):.2f}×" for method, values in ratios.items()}
    summary = ", ".join(f"**{NAMES[method]} {ranges[method]}**" for method in METHODS)
    passes = sum(p["quality_relative_gap_percent"] < 5 for p in data["pairs"])
    if language == "en":
        intro = (
            "Measured complete reductions on **full ogbn-arxiv: 169,343 nodes and 2,315,598 stored edges after symmetrization**. "
            "All three methods share identical saved input tensors, official splits, the same RTX PRO 6000 Blackwell GPU and four CPU threads. "
            f"GraphSlim/author reduction-time ratios: {summary}; **lower is better**."
        )
        alt = "Matched GECC, CGC and ClustGDD ogbn-arxiv reductions: time above, peak allocated GPU memory below"
        links = "[Vector PDF](figures/clustering_arxiv_comparison.pdf) · [High-resolution PNG](figures/clustering_arxiv_comparison.png) · [Reviewed chart data](benchmark/results/clustering_arxiv_figure.json)"
        caption = (
            "Columns show GECC, CGC and ClustGDD. **Top: time / matched author median (1), with measured seconds. Bottom: peak allocated GPU memory (GiB). Both metrics: lower is better.** "
            "Each bar is the median of three fresh-process reductions; whiskers show observed min–max. All **54 reductions / nine method–budget pairs** are included. "
            "Budgets of 0.1%, 0.5% and 1% of the **90,941 labeled training nodes** retain 90, 454 and 909 nodes. "
            "GECC adds a separately frozen measurement cohort. Slower results are retained."
        )
        details = [
            "<details>",
            "<summary>Measurement protocol and frozen sources</summary>",
            "",
            "- Timing includes reducer construction, internal preprocessing and transfers from the common standardized CPU graph, with CUDA synchronization; imports, shared input preparation, saving and GCN evaluation are excluded. GPU peaks reset after warmup, before input transfers.",
            "- Seeds 1–3 alternate implementation order. GECC is measured on the same input tensors and GPU UUID as the existing CGC/ClustGDD cohort, with four CPU threads and TF32 disabled.",
            "- GECC: full transductive propagation, depth two, weights (0.6, 0.5, 0), hard clustering, one restart, maximum 300 iterations. Both implementations use the author's uniform sampled centroid initialization; GraphSlim's native initializer is not measured. Reduced features, adjacency and labels match exactly in all nine rate–seed pairs. The author's dense diagnostic assignments and GraphSlim's sparse diagnostic assignments are included in reduction memory.",
            "- CGC: matched ten initializations / 300 iterations with native FAISS/sklearn convergence rules and Float64 reconstruction. ClustGDD: matched 600 pretraining / 100 refinement updates and three MiniBatchKMeans initializations.",
            f"- Outside timing, each reduced graph receives one 300-epoch GCN fit with validation selection. **{passes}/9** method–budget pairs have two-sided relative differences between implementation mean accuracies below 5%. These comparisons use pinned author implementations and have a separate scope from paper-reference reproduction coverage.",
            "- [CGC/ClustGDD records](benchmark/results/arxiv_clustering_comparison_20261101.json) · [measured source archive](benchmark/results/arxiv_clustering_source_20261101.zip).",
            "- [GECC records and pinned author commit](benchmark/results/gecc_arxiv_comparison_20261102.json) · [measured source archive](benchmark/results/gecc_arxiv_source_20261102.zip) · [runner](benchmark/compare_gecc_arxiv.py). GECC requires the pinned author checkout under runs/reference/GECC; the common runtime uses faiss-cpu==1.8.0.post1 with NumPy 1.x.",
            "- Regenerate with `python -m benchmark.plot_clustering_arxiv`; append `--check` to verify all source records and figure hashes.",
            "- Historical synthetic-tensor operator reports remain available separately: [CPU](benchmark/results/gecc_operator_cpu_20261029.json), [CUDA](benchmark/results/gecc_operator_gpu_20261029.json). Their operator timings are excluded from this figure.",
            "</details>",
        ]
    else:
        intro = (
            "在**完整 ogbn-arxiv（169,343 个节点，对称化后存储 2,315,598 条边）**上测量完整缩减流程。"
            "三种方法使用相同的冻结输入、官方划分、同一张 RTX PRO 6000 Blackwell GPU 和 4 个 CPU 线程。"
            f"GraphSlim / 作者实现缩减耗时比：{summary}，**越低越好**。"
        )
        alt = "GECC、CGC、ClustGDD 在完整 ogbn-arxiv 上的配对缩减比较：上排时间，下排峰值已分配 GPU 显存"
        links = "[矢量 PDF](figures/clustering_arxiv_comparison.pdf) · [高清 PNG](figures/clustering_arxiv_comparison.png) · [经核对的绘图数据](benchmark/results/clustering_arxiv_figure.json)"
        caption = (
            "三列依次为 GECC、CGC、ClustGDD。**上排：耗时 / 同组作者实现中位数（作者为 1），标注实测秒数；下排：峰值已分配 GPU 显存（GiB）。两项均越低越好。**"
            "每根柱为 3 个独立进程缩减的中位数，误差线为实测最小值至最大值，包含全部 **54 次缩减、9 个方法×预算组合**。"
            "0.1%、0.5%、1% 均以 **90,941 个有标签训练节点**为分母，分别保留 90、454、909 个节点。"
            "GECC 为另行冻结的补测批次，图中保留耗时更长的结果。"
        )
        details = [
            "<details>",
            "<summary>实测口径与冻结源码</summary>",
            "",
            "- 从共同的标准化 CPU 图开始计时，包含缩减器构造、内部预处理和设备传输，起止同步 CUDA；不含导入、共同输入准备、保存和 GCN 评估。GPU 峰值在预热后、输入传入 GPU 前重置。",
            "- 种子 1–3 交替执行两种实现。GECC 与既有 CGC/ClustGDD 批次的输入张量及 GPU UUID 一致，使用 4 个 CPU 线程并关闭 TF32。",
            "- GECC：完整直推式传播，深度 2，权重 (0.6, 0.5, 0)，硬聚类，单次初始化，最多 300 轮。双方均采用作者的均匀随机中心初始化；此处不测量 GraphSlim 原生初始化。全部 9 个预算×种子组合的缩减特征、邻接和标签逐元素一致。作者的稠密诊断分配与 GraphSlim 的稀疏诊断分配均计入缩减显存。",
            "- CGC：双方均为 10 次初始化、最多 300 轮，保留 FAISS/sklearn 各自收敛规则，重构采用 Float64。ClustGDD：双方均为 600 轮预训练、100 轮优化、3 次 MiniBatchKMeans 初始化。",
            f"- 计时之外，每个缩减图训练一次 300 轮 GCN，按验证集选择检查点。**{passes}/9** 个方法×预算组合的实现间准确率均值双边相对差距低于 5%。参照为固定作者版本，与论文数值参照的复现覆盖率各有独立统计范围。",
            "- [CGC/ClustGDD 原始记录](benchmark/results/arxiv_clustering_comparison_20261101.json) · [实测源码归档](benchmark/results/arxiv_clustering_source_20261101.zip)。",
            "- [GECC 原始记录及作者固定版本](benchmark/results/gecc_arxiv_comparison_20261102.json) · [实测源码归档](benchmark/results/gecc_arxiv_source_20261102.zip) · [运行器](benchmark/compare_gecc_arxiv.py)。在 runs/reference/GECC 准备固定作者版本；共同运行环境使用 faiss-cpu==1.8.0.post1 与 NumPy 1.x。",
            "- 执行 `python -m benchmark.plot_clustering_arxiv` 重新生成，加 `--check` 核验全部原始记录和图文件哈希。",
            "- 历史合成张量算子记录单独保留：[CPU](benchmark/results/gecc_operator_cpu_20261029.json)、[CUDA](benchmark/results/gecc_operator_gpu_20261029.json)，其算子耗时不进入此图。",
            "</details>",
        ]
    image = f"[![{alt}](figures/clustering_arxiv_comparison.svg)](figures/clustering_arxiv_comparison.pdf)"
    return "\n\n".join([intro, image, links, caption, "\n".join(details)])
