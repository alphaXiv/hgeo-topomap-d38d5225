## Reproduction: do geometric priors improve centerline detection?

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/hgeo-topomap-d38d5225/blob/main/notebooks/hgeo_topomap_reproduction.py)

**Assessment: partially reproduced.** We tested the paper's main claim on the actual public OpenLane-V2 sample with a matched compact TopoLogic-style detector. The paper reports centerline DET_l increasing from **29.7 to 31.3 (+1.6)**; our smaller Fréchet-AP scale increased from **0.1054 to 0.1330 (+0.0276, +26.2%)** across eight seeds after retuning the GCL loss weight. GAL and retuned GCL each helped separately, their combination helped most, and the combined model improved the missing-camera macro score by **20.9%**.

This is directional evidence, not a full-score replication. The experiment used 48 training and 16 held-out frames, public ResNet-18 features, SegFormer Cityscapes road masks in place of the paper's Mask2Former/Mapillary masks, a compact decoder, and an orientation-grouped contrastive GCL loss rather than the complete geometry-aware decoder. The disclosed GCL weight `0.1` hurt this compact model; `0.001` was required. The GAL road content helped, but the prior mask itself was not independently superior in GAL-only controls.

- [Illustrated report](reports/hgeo-topomap/report.md)
- [Self-contained marimo tutorial](notebooks/hgeo_topomap_reproduction.py)
- [Exact aggregate results](reports/hgeo-topomap/results.json)
- [Open the notebook directly in Molab](https://molab.marimo.io/github/alphaXiv/hgeo-topomap-d38d5225/blob/main/notebooks/hgeo_topomap_reproduction.py)

All evidence runs used Kubernetes on NVIDIA RTX PRO 6000 Blackwell GPUs. Each run allocated four GPUs and executed four seeds in parallel; four concurrent runs reached a peak of **16 GPUs**. The recovery campaign took **49m 28s (0.83 wall hours)** from inspection through the last terminal result.

### Experiment log

The command is copied verbatim from `orx exp status`; variants differ only through committed code/config.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public report and tutorial | Not run as an experiment (publication surface) | Presentation only | — |
| [baseline 0–3](https://github.com/alphaXiv/hgeo-topomap-d38d5225/tree/orx/recovery-baseline-fresh-0-3) + [4–7](https://github.com/alphaXiv/hgeo-topomap-d38d5225/tree/orx/recovery-baseline-seeds-4-7) | Matched compact control | `bash scripts/run_k8s.sh` | DET_l 0.1054 | Kubernetes, 4 GPUs/run |
| [GAL 0–3](https://github.com/alphaXiv/hgeo-topomap-d38d5225/tree/orx/recovery-gal-fresh-0-3) + [4–7](https://github.com/alphaXiv/hgeo-topomap-d38d5225/tree/orx/recovery-gal-seeds-4-7) | Road prior and masked attention, `k=0.1` | `bash scripts/run_k8s.sh` | Aligned: 0.1148 | Kubernetes, 4 GPUs/run |
| [GCL, disclosed weight](https://github.com/alphaXiv/hgeo-topomap-d38d5225/tree/orx/recovery-gcl-fresh-0-3) | Orientation consistency, `β=0.1` | `bash scripts/run_k8s.sh` | Divergent: 0.0634 | Kubernetes, 4 GPUs/run |
| [GCL, retuned 0–3](https://github.com/alphaXiv/hgeo-topomap-d38d5225/tree/orx/recovery-gcl-beta-0-001) + [4–7](https://github.com/alphaXiv/hgeo-topomap-d38d5225/tree/orx/recovery-gcl-beta-0-001-seeds-4-7) | Orientation consistency, `β=0.001` | `bash scripts/run_k8s.sh` | Aligned after scaling: 0.1247 | Kubernetes, 4 GPUs/run |
| [combined 0–3](https://github.com/alphaXiv/hgeo-topomap-d38d5225/tree/orx/recovery-combined-beta-0-001) + [4–7](https://github.com/alphaXiv/hgeo-topomap-d38d5225/tree/orx/recovery-combined-beta-0-001-seeds-4-7) | GAL + retuned GCL | `bash scripts/run_k8s.sh` | Strongest: 0.1330; missing-view 0.0674 | Kubernetes, 4 GPUs/run |
| [GCL index control](https://github.com/alphaXiv/hgeo-topomap-d38d5225/tree/orx/recovery-gcl-index-group-control) | Replace geometry groups with arbitrary index groups | `bash scripts/run_k8s.sh` | 0.1069, near baseline | Kubernetes, 4 GPUs/run |
| [GAL global-attention control](https://github.com/alphaXiv/hgeo-topomap-d38d5225/tree/orx/recovery-gal-global-attention-control) | Remove spatial mask selectivity | `bash scripts/run_k8s.sh` | 0.1266; mask not isolated for GAL alone | Kubernetes, 4 GPUs/run |
| [combined zero-road control](https://github.com/alphaXiv/hgeo-topomap-d38d5225/tree/orx/recovery-combined-zero-road-prior-control) | Remove semantic road occupancy | `bash scripts/run_k8s.sh` | 0.1183; semantic prior contributes | Kubernetes, 4 GPUs/run |

---

## Upstream project README

<p align="center">

  <h1 align="center">HGeo-TopoMap: Boosting Topological Mapping with Hierarchical Geometric Priors</h1>
  <p align="center">
    <a href="https://www.researchgate.net/profile/Siyu-Li-45"><strong>Siyu Li</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=pA9c0YsAAAAJ"><strong>Kunyu Peng†</strong></a>
    ·
   <a href="https://scholar.google.com/citations?user=aqGMqEcAAAAJ"><strong>Di Wen</strong></a>
    ·
    <a href="https://aee.zust.edu.cn/info/1101/3809.htm"><strong>Beiping Hou</strong></a>
    ·
    <a href="http://robotics.hnu.edu.cn/info/1176/2960.htm"><strong>Zhiyong Li</strong></a>
    ·
    <a href="https://yangkailun.com"><strong>Kailun Yang†</strong></a>
    
</p>

<p align="center">
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <br>
    <a href="https://arxiv.org/pdf/2607.21281">
      <img src='https://img.shields.io/badge/Paper-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
</p>

## Motivation
<div align=center>
<img src="https://github.com/lynn-yu/HGeo-Topomap/blob/main/img2.png" width="300" height="500" >
</div>

## Framework
<div align=center>
<img src="https://github.com/lynn-yu/HGeo-Topomap/blob/main/img1.png" >
</div>

### Abstract
Topological maps are key outputs of autonomous driving perception systems, delivering essential road information for path planning. They identify instances such as centerlines and traffic signs, along with their connectivity relationships. Due to the lack of explicit markings for centerlines in real-world environments, the detection of centerline instances remains a significant challenge. To tackle this problem, we propose HGeoTopoMap, which leverages an explicit prior map and implicit spatial relations to hierarchically boost topological mapping. First, a geometric adaptive learning module is designed for the road structure map obtained via inverse perspective mapping. This module discretely encodes semantic and spatial features from the map, followed by a prior-mask attention mechanism that selectively focuses on informative regions. Then, a geometric consistency learning module is devised, which leverages the geometric properties and spatial relationships of centerlines. Built on the geometry-aware decoder, it enforces spatial consistency by aligning features of centerline instances with identical geometric orientations. The proposed method is evaluated on the OpenLaneV2 dataset across the centerline, lane segment, and robustness benchmarks. Beyond substantial improvements in topological mapping accuracy, the proposed method offers the added benefit of strong robustness, consistently outperforming baselines under both standard and challenging conditions.

### Update
2026.7 Init repository.

### Acknowledgement
The code framework of this project is based on ![TopoLogic](https://github.com/Franpin/TopoLogic) and ![LaneSegNet]([https://github.com/Junyu-Z/Semi-BEVseg](https://github.com/OpenDriveLab/LaneSegNet)), thanks to this excellent work.
The pespective road map are generated by ![Mask2Former](https://github.com/facebookresearch/Mask2Former).

## 🤝 Publication:
Please consider referencing this paper if you use the ```code``` from our work.
Thanks a lot :)

```
@article{li2026hgeo_topomap,
  title={HGeo-TopoMap: Boosting Topological Mapping with Hierarchical Geometric Priors},
  author={Li, Siyu and Peng, Kunyu and Wen, Di and Hou, Beiping and Li, Zhiyong and Yang, Kailun},
  journal={arXiv preprint arXiv:2607.21281},
  year={2026}
}
```
