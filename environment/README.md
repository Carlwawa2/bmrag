# 当前基线的 Python 环境

使用 `.venv`：Python 3.11.16、CPU PyTorch 2.3.0、NumPy 1.26.4、Transformers 4.44.2，共 65 个固定依赖。

- [直接依赖](requirements.in)
- [精确版本和 SHA-256 锁](requirements-win-py311.lock.txt)
- [安装包清单](installed-packages.json)

默认检查环境，不安装软件：

```powershell
.\.venv\Scripts\python.exe -m hgrag environment
```

结果保存在 `reports/environment/validation.json`。从锁文件复建：

```powershell
.\.venv\Scripts\python.exe -m hgrag environment --restore
```

首次复建需要 uv 和下载网络。若 `.venv` 不存在，可以先用系统 Python 执行 `python -m hgrag environment --restore`。复建器使用项目内的 Python 和缓存，不自动删除版本不符的已有环境。

直接指定解释器即可，无需 PowerShell 激活脚本。实验前设置 `$env:PYTHONHASHSEED = '42'`。`artifacts/toolchain/python/` 是此虚拟环境依赖的基础解释器，应保留。
