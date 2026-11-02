"""Compose the README's measured full-ogbn-arxiv time and GPU-memory block."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark/results/arxiv_clustering_comparison_20261101.json"
FIGURE = ROOT / "benchmark/results/arxiv_clustering_figure.json"


def speed_figure(language):
    source = json.loads(SOURCE.read_text())
    data = json.loads(FIGURE.read_text())
    assert data["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert data["displayed_metrics"] == ["reduction_seconds", "peak_cuda_allocated_bytes"]
    for ext, digest in data["assets_sha256"].items():
        assert (
            hashlib.sha256(
                (ROOT / f"figures/arxiv_clustering_comparison{ext}").read_bytes()
            ).hexdigest()
            == digest
        )
    archive = source["source_archive"]["path"]
    assert (
        hashlib.sha256((ROOT / archive).read_bytes()).hexdigest()
        == source["source_archive"]["sha256"]
    )
    ratios = {
        m: [p["time_ratio"] for p in data["pairs"] if p["method"] == m] for m in ["cgc", "clustgdd"]
    }
    ranges = {m: f"{min(v):.2f}–{max(v):.2f}×" for m, v in ratios.items()}
    passes = sum(p["quality_shortfall_percent"] < 5 for p in data["pairs"])
    exceptions = [p for p in data["pairs"] if p["quality_shortfall_percent"] >= 5]
    if language == "en":
        intro = (
            "Measured on **full ogbn-arxiv: 169,343 nodes and 2,315,598 stored edges after symmetrization**. "
            "Both implementations use the same saved inputs, one RTX PRO 6000 Blackwell GPU and four CPU threads. "
            f"Observed author/GraphSlim reduction-time ratios: **CGC {ranges['cgc']}**, **ClustGDD {ranges['clustgdd']}**."
        )
        alt = "Measured ogbn-arxiv reduction time and peak GPU memory for CGC and ClustGDD at three reduction rates"
        links = "[Vector PDF](figures/arxiv_clustering_comparison.pdf) · [High-resolution PNG](figures/arxiv_clustering_comparison.png) · [Raw measurements](benchmark/results/arxiv_clustering_comparison_20261101.json)"
        caption = (
            "Each bar summarizes **three fresh-process reductions**; whiskers show their observed minimum–maximum. "
            "Left: speed relative to the matched author's median (1×), with measured seconds above the bars. Right: peak GPU memory in GiB. "
            "The 0.1%, 0.5% and 1% budgets refer to the **90,941 labeled training nodes**, retaining 90, 454 and 909 nodes. "
            "GPU memory is PyTorch peak allocated memory during reduction."
        )
        details = [
            "<details>",
            "<summary>Measurement protocol and frozen source</summary>",
            "",
            "- Timing starts from the common CPU graph after symmetrization and training-fitted standardization. It includes reducer-internal preprocessing and device transfers, with CUDA synchronization; imports, shared input preparation, saving and downstream evaluation are excluded.",
            "- GPU peaks are reset after context warmup, before input transfers. Process RSS and allocator-reserved peaks are also retained in the raw records.",
            "- Inputs use the official ogbn-arxiv splits and feature standardization fitted on training nodes. Seeds 1–3 alternate implementation order. All three declared rates and all 36 reduction records are included.",
            "- CGC retains its rate-specific settings and matched caps of 10 initializations / 300 iterations (author FAISS versus GraphSlim sklearn, retaining their convergence rules). ClustGDD uses matched 600 pretraining / 100 refinement updates and MiniBatchKMeans with three initializations. Both CGC reconstruction solves use float64; TF32 is disabled.",
            f"- A separate shared-GCN check uses one 300-epoch fit per reduced graph (three graphs per implementation); validation selects the checkpoint. **{passes}/6** pairs have mean one-sided accuracy shortfall below 5%. The reference is the matched pinned author implementation; these checks have their own scope from the paper-reference coverage below.",
            f"- [Measured GraphSlim source and harness]({archive}) · [Pinned author versions, adapters and per-seed results](benchmark/results/arxiv_clustering_comparison_20261101.json).",
            "- Runner prerequisites: `faiss-cpu==1.8.0.post1` with NumPy 1.x, and the pinned CGC / ClustGDD checkouts under `runs/reference`; versions and commits are recorded in the snapshot.",
            "- Extract the archive with `unzip benchmark/results/arxiv_clustering_source_20261101.zip -d runs/arxiv_measured_source`. Plan using the measured development source: `python -m benchmark.compare_arxiv plan --package-source runs/arxiv_measured_source/source/graphslim --run-dir runs/arxiv_NEW`; then run with `python -m benchmark.compare_arxiv run --run-dir runs/arxiv_NEW`. GPU selection is explicit.",
            "- Regenerate with `python -m benchmark.plot_arxiv_efficiency`; append `--check` to validate source records and figure hashes.",
        ]
        for p in exceptions:
            details.append(
                f"- {p['method'].upper()} / rate {p['rate']}: observed shared-GCN shortfall **{p['quality_shortfall_percent']:.2f}%**."
            )
    else:
        intro = (
            "在**完整 ogbn-arxiv（169,343 个节点，对称化后存储 2,315,598 条边）**上实测。"
            "作者实现与 GraphSlim 共用同一份冻结输入、一张 RTX PRO 6000 Blackwell GPU 和 4 个 CPU 线程。"
            f"实测作者实现 / GraphSlim 缩减耗时比：**CGC {ranges['cgc']}**，**ClustGDD {ranges['clustgdd']}**。"
        )
        alt = "完整 ogbn-arxiv 上 CGC 和 ClustGDD 在三档缩减预算下的耗时与峰值 GPU 显存实测"
        links = "[矢量 PDF](figures/arxiv_clustering_comparison.pdf) · [高清 PNG](figures/arxiv_clustering_comparison.png) · [原始实测记录](benchmark/results/arxiv_clustering_comparison_20261101.json)"
        caption = (
            "每根柱汇总 **3 个独立进程缩减**，误差线为实测最小值至最大值。"
            "左图以同组作者实现的耗时中位数为 1×，柱顶标注实测秒数；右图为峰值 GPU 显存（GiB）。"
            "0.1%、0.5%、1% 均以 **90,941 个有标签训练节点**为分母，分别保留 90、454、909 个节点。"
            "显存为缩减阶段的 PyTorch 峰值已分配显存。"
        )
        details = [
            "<details>",
            "<summary>实测口径与冻结源码</summary>",
            "",
            "- 从已完成对称化、训练集拟合标准化的共同 CPU 图开始计时，覆盖缩减内部的预处理和设备间传输，起止均同步 CUDA；不含导入、共同输入准备、保存与下游评估。",
            "- GPU 峰值在上下文预热后、输入传入 GPU 前重置。原始记录同时保留进程 RSS 和缓存保留显存峰值。",
            "- 使用 ogbn-arxiv 官方划分，特征标准化仅在训练节点上拟合。种子 1–3 交替执行两种实现，保留全部 3 档预算和 36 次缩减记录。",
            "- CGC 沿用分档参数，双方聚类上限均为 10 次初始化、300 轮（作者 FAISS 与 GraphSlim sklearn 各自保留收敛规则）。ClustGDD 双方均使用 600 轮预训练、100 轮优化，以及初始化 3 次的 MiniBatchKMeans。CGC 双方重构求解均使用 float64，关闭 TF32。",
            f"- 另行使用共享 GCN 检查效果：每个缩减图训练 1 次、每次 300 轮（每种实现 3 个图），按验证集选检查点。**{passes}/6** 组的准确率均值单边相对短缺低于 5%。此处参照为固定版本的作者实现，与下方采用论文数值参照的覆盖率各有独立统计范围。",
            f"- [实测 GraphSlim 源码与运行器]({archive}) · [作者固定版本、适配说明及逐种子结果](benchmark/results/arxiv_clustering_comparison_20261101.json)。",
            "- 运行前需安装 `faiss-cpu==1.8.0.post1`（NumPy 1.x），并在 `runs/reference` 准备固定版本的 CGC / ClustGDD；版本与提交均记录于快照中。",
            "- 先执行 `unzip benchmark/results/arxiv_clustering_source_20261101.zip -d runs/arxiv_measured_source`，再用实测开发版本建任务：`python -m benchmark.compare_arxiv plan --package-source runs/arxiv_measured_source/source/graphslim --run-dir runs/arxiv_NEW`，随后执行 `python -m benchmark.compare_arxiv run --run-dir runs/arxiv_NEW`；GPU 可显式指定。",
            "- 执行 `python -m benchmark.plot_arxiv_efficiency` 重新生成图，加 `--check` 核验原始记录与图文件哈希。",
        ]
        for p in exceptions:
            details.append(
                f"- {p['method'].upper()} / 比例 {p['rate']}：共享 GCN 下单边短缺实测为 **{p['quality_shortfall_percent']:.2f}%**。"
            )
    image = f"[![{alt}](figures/arxiv_clustering_comparison.svg)](figures/arxiv_clustering_comparison.pdf)"
    return "\n\n".join([intro, image, links, caption, "\n".join(details + ["", "</details>"])])
