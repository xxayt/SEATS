<div align="center">

<img src="assets/logo.png" alt="SEATS" width="160">

<h1>Stage-adaptive Token Selection for Efficient Omni-modal LLMs</h1>

<div align="center">
  <a href="https://arxiv.org/abs/2605.20035"><img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red&logo=arxiv"></a> &ensp;
  <a href="https://xxayt.github.io/SEATS/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=green"></a> &ensp;
  <a href="https://github.com/xxayt/SEATS"><img src="https://img.shields.io/badge/GitHub-Code-black?logo=github"></a> &ensp;
  <a href="https://github.com/xxayt/SEATS">
  <img src="https://img.shields.io/github/stars/xxayt/SEATS?style=social"></a> &ensp;
</div>

<div>
    <a href='https://xxayt.github.io/' target='_blank'>Zijie Xin</a><sup>1</sup>&emsp;
    <a href='https://yangjie-cv.github.io/' target='_blank'>Jie Yang</a><sup>2,📧</sup>&emsp;
    <a href='https://ruixiangzhao.github.io/' target='_blank'>Ruixiang Zhao</a><sup>1</sup>&emsp;
    <a href='' target='_blank'>Tianyi Wang</a><sup>2</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=38dACd4AAAAJ&hl=zh-CN&oi=ao' target='_blank'>Fengyun Rao</a><sup>2</sup>&emsp;
    <a href='' target='_blank'>Jing Lyu</a><sup>2</sup>&emsp;
    <a href='http://lixirong.net/' target='_blank'>Xirong Li</a><sup>1,📧</sup>&emsp;
</div>
<div>
    📧 Corresponding authors
</div>
<div>
    <sup>1</sup> Renmin University of China&emsp; 
    <sup>2</sup> WeChat Vision, Tencent Inc.&emsp;
</div>

---

## 👀 Overview
**SEATS** is a training-free, <u>s</u>tag<u>e</u>-<u>a</u>daptive <u>t</u>oken <u>s</u>election method for efficient omni-modal LLM inference. By analyzing layer-wise token dependency, it reveals that visual and audio dependencies follow a block-wise pattern and weaken with depth. SEATS removes spatiotemporal redundancy before the LLM, progressively prunes tokens inside the LLM, and fully removes non-textual tokens in late layers.

<img src="assets/teasor_overall.png" width="900px"/>
</div>

<hr>

## ✨ Key Highlights

- 💡 **New Insight:** Reveals a block-wise dependence pattern in omni-modal LLMs, where reliance on visual and audio tokens weakens with layer depth.
- ⚡ **Strong Efficiency:** **9.3x FLOPs reduction** and **4.8x prefill speedup** at 10% token retention while preserving **96.3%** performance.
- 🎯 **Stage-adaptive Design:** Diversity-based pre-LLM selection + query-guided inner-LLM progressive pruning + late-layer full removal.
- 🔌 **Broad Compatibility:** Plug-and-play and training-free for direct application to Qwen2.5-Omni-7B and Qwen3-Omni-30B.


## 📅 TODO
Code will be released by June 2026.
- [ ] Support Qwen2.5-Omni-7B
- [ ] Support Qwen3-Omni-30B
- [ ] Release benchmark adaptation code for LMMs-Eval (WorldSense, Daily-Omni, OmniVideoBench, Video-MME, LVOmniBench)
- [ ] Evaluation scripts and reproduction guide (adapted for LMMs-Eval)
- [ ] Release more baseline implementations (FastV, VisionZip, DivPrune, DyCoke, and OmniZip)
- [ ] *future work*: Support more models (OmniVinci-7B)

## 🏗️ Method

![Method](assets/method.png)

SEATS is a three-stage method:

1. **Pre-LLM Token Selection:** Removes spatiotemporal redundancy within each temporal window via attention-weighted diversity selection.
2. **Inner-LLM Token Selection:** Progressively prunes tokens with a block-wise TRR decay schedule and top-down budget allocation (inter-window then intra-window) guided by query relevance.
3. **Late-block Removal:** Removes all remaining non-textual tokens in late layers where cross-modal fusion is complete.


## 📈 Main Results

Results on **Qwen2.5-Omni-7B** (5 audio-visual benchmarks):

| Method | R | TFLOPs | WorldSense | Daily-Omni | OmniVideoBench | Video-MME | LVOmniVideo | Mean |
|--------|---|--------|------------|------------|----------------|-----------|-------------|------|
| Full tokens | 100% | 111.0 (1.0x) | 46.7 | 64.0 | 34.1 | 65.3 | 33.3 | 48.7 (100.0%) |
| SEATS | 35% | 36.7 (3.0x) | 46.2 | 62.1 | 35.0 | 66.8 | 36.2 | **49.3** (101.1%) |
| SEATS | 25% | 26.5 (4.2x) | 45.3 | 60.9 | 34.7 | 66.5 | 35.7 | **48.6** (99.8%) |
| SEATS | 10% | 12.0 (9.3x) | 43.5 | 57.8 | 33.6 | 64.6 | 35.1 | **46.9** (96.3%) |

Efficiency analysis (A800 GPU, WorldSense):

| Method | R | Prefill Speedup | TTFT Reduction | GPU Mem. (GB) |
|--------|---|-----------------|----------------|---------------|
| SEATS | 35% | 2.1x | 1.4x | 18.68 |
| SEATS | 25% | 2.7x | 1.6x | 18.29 |
| SEATS | 10% | 4.8x | 1.9x | 17.65 |


## 🚀 How to Run
coming soon...


## 🤝 Acknowledgement
This implementation relies on resources from [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni), [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni), [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and [DivPrune](https://github.com/vbdi/divprune). We thank the original authors for their excellent contributions and for making their work publicly available.


## ✏️ Citation
If you find this work useful, please consider citing:

```bibtex
@article{xin2026seats,
  title={Stage-adaptive Token Selection for Efficient Omni-modal LLMs},
  author={Xin, Zijie and Yang, Jie and Zhao, Ruixiang and Wang, Tianyi and Rao, Fengyun and Lyu, Jing and Li, Xirong},
  journal={arXiv preprint arXiv:2605.20035},
  year={2026}
}
```


## 📜 License
This project is licensed under the [MIT License](./LICENSE). For commercial licensing or any use beyond research, please contact the authors.

#### 📬 Contact for Issues
For any questions about this project (e.g., corrupted files or loading errors), please reach out at: [xinzijie@ruc.edu.cn](mailto:xinzijie@ruc.edu.cn)