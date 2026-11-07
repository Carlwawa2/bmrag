# 当前 BGE-M3 模型

- 官方仓库：<https://huggingface.co/BAAI/bge-m3>
- revision：`5617a9f61b028005a4858fdac845db406aefb181`
- 权重位置：`artifacts/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/`
- 输出：1024 维稠密向量，CLS pooling 和 L2 归一化
- 运行：CPU、FP32、4 线程、单条编码
- 输入上限：8192 个本模型 tokenizer tokens（含特殊 tokens），超出报错
- 模型卡声明许可证：MIT

`revision.json` 保存官方 revision、长度、Git blob OID 和 LFS SHA-256；`manifest.json` 保存本地文件 SHA-256。共 13 个 PyTorch、tokenizer、配置和模型卡文件，包含原模型的可选头；当前只使用稠密向量。

下载固定模型（需要网络）和离线复验：

```powershell
.\.venv\Scripts\python.exe -m hgrag download-model
.\.venv\Scripts\python.exe -m hgrag download-model --verify-only
```

验证结果在 `reports/models/bge-m3-validation.json`。本地嵌入不需要 API Key。首次准备分块 tokenizer 使用 `python -m hgrag prepare-tokenizer`；数据缓存仍应保留以支持离线运行。
