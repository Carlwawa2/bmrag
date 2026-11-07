# 模型替换后的 HyperGraphRAG 基线

当前工程保留：固定的 HyperGraphRAG 原始源码、`gpt-6-luna` 模型适配、本地 BGE-M3，以及已完成的小规模端到端验收。后续预算策略、缺口估计、证据选择、两步探索和 MuSiQue 训练已从活动工程撤回。

| 内容 | 位置 |
| --- | --- |
| 唯一基线配置 | [configs/baseline.json](configs/baseline.json) |
| 基线工厂 | [hgrag/runtime.py](hgrag/runtime.py) |
| 当前基线入口与记录 | [baseline-current.json](baseline-current.json) |
| 基线验证 | [reports/baseline/validation.json](reports/baseline/validation.json) |
| 小规模端到端验收 | [reports/hotpot-mini/REPORT.md](reports/hotpot-mini/REPORT.md) |
| 离线检索与回答重放 | [reports/baseline/replay.json](reports/baseline/replay.json) |
| 环境与依赖 | [environment/README.md](environment/README.md) |
| BGE-M3 固定版本 | [models/bge-m3/README.md](models/bge-m3/README.md) |
| 工程结构 | [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) |

## 已固定的内容

- HyperGraphRAG commit：`e01d2f2a5f3d1e80415b933c10fa7d14587cf811`，上游源码未改动。
- 抽取、摘要、查询分析和生成模型：`gpt-6-luna`，通过原有回调接口注入。
- 嵌入模型：本地 `BAAI/bge-m3`，revision `5617a9f61b028005a4858fdac845db406aefb181`，CPU、FP32、CLS、L2 归一化，1024 维。
- 分块：1200 token，重叠 100；检索：原始 hybrid，top_k=60，各分区上限 4000 token。
- 配置 SHA-256：`9a11a7bc9b537758eecc188aa5fa3e4aca6894f030c557df4cc7e0cf66ca00bc`。

`tiktoken_model_name` 只指定分词编码映射。上游分区限制不等于总上下文硬预算，当前基线没有加入论文方法。

## 使用

在项目根目录运行，无需激活虚拟环境。本地原验收目录为 `D:\step_1`。

### 从 GitHub 克隆并复现

上游源码以固定提交的 Git 子模块保存，需要同时克隆子模块：

```powershell
git -c core.autocrlf=false clone --recurse-submodules https://github.com/Carlwawa2/bmrag.git
Set-Location bmrag
$env:PYTHONHASHSEED = '42'
```

Windows x86-64 环境需要已安装系统 Python、Git 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。首次安装固定环境并下载固定版本模型与分词编码：

```powershell
python -m hgrag environment --restore
.\.venv\Scripts\python.exe -B -m hgrag download-model
.\.venv\Scripts\python.exe -B -m hgrag prepare-tokenizer
```

`.venv`、约 2.30 GB 的 BGE-M3 文件、Python 下载缓存和本地回退恢复包不纳入 Git。模型文件通过已提交的 revision 与校验值重新下载；原始小规模验收产物和基线源码快照已纳入仓库。`.gitattributes` 保持文件原始字节，避免换行转换破坏固定输入的哈希。

如果普通克隆时漏掉子模块，执行 `git -c core.autocrlf=false submodule update --init --recursive`。

### 检查已有基线

```powershell
$env:PYTHONHASHSEED = '42'
.\.venv\Scripts\python.exe -B -m hgrag verify
.\.venv\Scripts\python.exe -B -m hgrag environment
.\.venv\Scripts\python.exe -B -m hgrag replay
.\.venv\Scripts\python.exe -B -m hgrag review
```

上述命令无需模型 API 调用。`replay` 使用真实 BGE-M3 重新检索保存的超图，核对生成请求，再重放已有模型回答。

重新执行本地嵌入与向量库检查：

```powershell
.\.venv\Scripts\python.exe -B -m hgrag smoke
```

在新运行目录执行在线抽取、建图、检索和回答：

```powershell
.\.venv\Scripts\python.exe -B -m hgrag pilot --run-name baseline-new-run
```

在线运行通过 `OPENCODE_API_KEY` 或隐藏输入读取密钥，不将密钥写入文件。模型设置继续使用唯一的 `configs/baseline.json`。

## 当前验收

`artifacts/runs/pilot-001/` 保存了 2 道 HotpotQA 开发题、20 个候选段落的端到端验收：377 个实体、268 条超边、600 条关联。93 个原始产物保留不变。两题回答与参考答案含义一致，这属于工程验收，尚未形成正式评测集上的 EM/F1 成绩。

后续研究文件保存在本地的 `artifacts/rollback/research-before-baseline.zip`，不参与基线运行，也不上传至 GitHub；回退记录见 [rollback.json](reports/baseline/rollback.json)。
