# 当前基线的小型真实语料

固定取 [HotpotQA](https://hotpotqa.github.io/) distractor validation 原始顺序的前两题：一题比较、一题桥接，每题保留全部 10 个候选 Wikipedia 段落，合并为一个 20 文档小图。

- `documents.json`：标题加原始段落，实际建图输入。
- `questions.json`：问题及事后核对用的答案、支持事实。
- `manifest.json`：来源、选取方式、问题 ID 和输入文件 SHA-256。

数据由 Yang、Qi、Zhang 等人在 EMNLP 2018 的 HotpotQA 中发布，许可为 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。没有改写原段落，也没有依据参考支持事实筛选文档。参考答案不参与建图或模型提示。

运行：`python -m hgrag pilot --run-name pilot-002`。当前配置在 `configs/pilot.json`，最新实验指针在 `reports/hotpot-mini/latest-run.json`，已完成的小图在 `artifacts/runs/pilot-001/`。

这是开发用流程样本，不是标准 HotpotQA 正式评测。后续应从正式留出测试中排除这两个问题 ID。
