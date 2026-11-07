# 真实语料端到端试运行结果

`gpt-6-luna` + 固定本地 BGE-M3 已完成抽取、建图、磁盘重载、检索与回答。当前具备经过小样本端到端验证的模型替换基线；正式基线成绩仍需在预先固定的独立评测集上测量。

## 输入与配置

使用 [HotpotQA](https://hotpotqa.github.io/) distractor validation 原始顺序的前两题：一题 comparison、一题 bridge。保留每题全部 10 个候选 Wikipedia 段落，合并为一个 20 文档小图。参考答案和支持事实只用于事后核对。数据许可和来源见 [语料记录](../../pilots/hotpot-mini/manifest.json)，这两题后续应排除在正式留出评测之外。

模型、原始提示、1200/100 分块、hybrid 检索、top_k=60 和各分区 4000-token 限制均由[当前配置](../../configs/baseline.json)固定。执行器使用异常安全信号量、瞬时错误最多一次重试和 120 次 API 尝试上限。当前配置 SHA-256：`9a11a7bc9b537758eecc188aa5fa3e4aca6894f030c557df4cc7e0cf66ca00bc`。

## 建图与持久化

| 项目 | 结果 |
|---|---:|
| 原始文档 / 分块 | 20 / 20 |
| 实体节点 | 377 |
| 超边节点 | 268 |
| 实体与超边间的关联 | 600 |
| 至少关联 3 个实体的超边 | 80 |
| 无关联实体的超边 | 4 |
| 建图耗时 | 122.15 s |

二部图关联类型和正文来源链接检查通过。GraphML 与三个向量库保存后重新加载，节点数、关联数、向量数和来源检查保持一致；回答使用的是重新加载的索引。

## 回答与实际证据

| 题型 | 原始问题 | 模型实际回答 | 上下文包含的来源段落 | 序列化上下文 tokens | 查询耗时 |
|---|---|---|---:|---:|---:|
| comparison | Were Scott Derrickson and Ed Wood of the same nationality? | Yes. Both Scott Derrickson and Ed Wood were American. | 19/20 | 10,500 | 4.85 s |
| bridge | What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell? | Shirley Temple portrayed Corliss Archer in *Kiss and Tell*. As an adult, she served as **Chief of Protocol of the United States**. | 19/20 | 11,502 | 6.73 s |

两题回答均与参考答案含义一致，全部 5 条标注支持句均出现在实际回答上下文的 Sources 分区。这里确认的是模型获得了相关证据，并未通过消融实验验证其因果依赖；没有计算正式 EM/F1，也不报告“100% 准确率”。

实际检索上下文均保留了 19/20 个来源段落，说明本轮尚未施加紧张的总上下文预算。10,500 和 11,502 tokens 也再次说明：上游的 4000-token 分区限制不等于总上下文硬上限。当前尚未实现研究中的探索预算与总上下文预算。

## 抽取格式与质量观察

20 次成功的初始抽取都返回了可识别的实体和超边记录，原始解析器完成建图。其中 3 次缺少或拼错了 `<|COMPLETE|>`；服务均以 `stop` 结束，不能把这些现象归因为输出长度截断。图中另有 4 个零成员超边，涉及 Tyler Bates、Village accountant 和 Meet Corliss Archer。详细条目见 [质量复核](review.json)。

这些记录作为当前基线的实际输出保留。正式评测前需要检查抽取格式遵循率、实体粒度、超边成员质量和摘要忠实性；本轮结构验证不代表全部 268 条超边均已完成语义核验。

## 调用与复验

本轮建图与问答共约 133.93 s（不含初始模型加载与后续复核），81 次 API 尝试：80 次成功，一次读取超时后重试成功。成功调用为初始抽取 20 次、补充抽取及循环判断 56 次、查询分析 2 次、回答 2 次。所有成功响应返回 `gpt-6-luna`。

80 个成功响应报告输入 390,957、输出 34,528，合计 425,485 tokens；其中包含服务报告的缓存输入，超时请求没有返回 usage。逐次 usage 和 Go 账目字段保存在脱敏调用审计中。密钥未写入文件，已扫描输出中的凭据模式。

- [完整运行摘要](../../artifacts/runs/pilot-001/pilot-summary.json)
- [图结构](../../artifacts/runs/pilot-001/graph_chunk_entity_relation.graphml)
- [逐次调用与失败记录](../../artifacts/runs/pilot-001/pilot-calls.json)
- [问题 1 实际上下文](../../artifacts/runs/pilot-001/context-5a8b57f25542995d1e6f1371.txt)
- [问题 2 实际上下文](../../artifacts/runs/pilot-001/context-5a8c7595554299585d9e36b6.txt)

93 个运行产物的校验值已核对，运行中的模型和检索参数、实际文档均与当前基线匹配。离线复核命令：

```powershell
.\.venv\Scripts\python.exe -m hgrag review
```

下一步固定正式评测数据划分、指标与预算口径，并在本轮开发样本之外检查抽取质量和预算敏感性，再记录后续改进方法要比较的基线成绩。
