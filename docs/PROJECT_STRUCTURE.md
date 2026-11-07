# 当前基线工程

```text
hgrag/
  runtime.py       唯一基线工厂与查询参数入口
  configuration.py 配置与固定输入校验
  source.py        上游源码校验与导入
  models/          OpenCode、BGE-M3、分词编码
  common/          路径、脱敏、哈希与源码快照
  evaluation/      环境检查、嵌入检查、端到端试运行、复核与重放
  tools/           环境复建、模型下载与实验快照
  methods/         基线阶段的空占位包
configs/           唯一基线配置与输入清单
environment/       Python 依赖声明、精确锁与安装清单
upstream/          固定 HyperGraphRAG 源码，使用 Git 子模块
models/bge-m3/     模型版本、文件清单与下载说明
pilots/hotpot-mini/ 小规模验收语料与问题
reports/           基线、环境、嵌入与端到端验收记录
artifacts/models/  本地 BGE-M3 权重
artifacts/toolchain/ 虚拟环境依赖的 Python 解释器
artifacts/tokenizer-cache/ 固定分词编码缓存
artifacts/runs/pilot-001/ 原始端到端验收的 93 个产物
artifacts/runs/9a11a7bc9b53/offline/ 基线嵌入与向量库存储验收
artifacts/rollback/ 后续研究内容的本地恢复包，不纳入 Git
```

活动源码已恢复为基线检查点的 24 个 Python 文件；没有预算访问、缺口评分、证据选择、两步探索或 MuSiQue 训练入口。

基线配置始终读取 `configs/baseline.json`。原始 HyperGraphRAG 源码未改动，模型替换通过 `hgrag.runtime.make_rag` 注入回调完成。

当前验收指针为 `reports/hotpot-mini/latest-run.json`。检查命令会核对原始语料、模型/检索参数、真实本地向量检索、生成请求及保存的回答。

回退记录在 [reports/baseline/rollback.json](../reports/baseline/rollback.json)，本地恢复包路径为 `artifacts/rollback/research-before-baseline.zip`。恢复包不会被运行入口读取，也不上传至 GitHub。

Git 仓库包含活动源码、固定输入、依赖锁、小型语料、验收报告和保存的基线索引。模型权重、解释器、虚拟环境与缓存由固定版本的下载/复建命令恢复。克隆时使用 `--recurse-submodules`，完整步骤见根目录 [README](../README.md)。
