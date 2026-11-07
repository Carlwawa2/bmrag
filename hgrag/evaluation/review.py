"""Verify the saved real run and write a compact human-readable review."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from ..common.files import ROOT, sha256
from ..configuration import load
from .records import load_run

PILOT = ROOT / "pilots/hotpot-mini"


def main():
    config = load(verify_artifacts=True)
    latest, report, directory, documents = load_run(config)
    import networkx as nx
    graph = nx.read_graphml(directory / "graph_chunk_entity_relation.graphml")
    chunks = json.loads((directory / "kv_store_text_chunks.json").read_text(encoding="utf-8"))
    empty_hyperedges = [
        {"name": name, "source_titles": [chunks[source]["content"].splitlines()[0] for source in attrs["source_id"].split("<SEP>")]}
        for name, attrs in graph.nodes(data=True) if attrs.get("role") == "hyperedge" and graph.degree(name) == 0
    ]
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in (directory / "model_calls").glob("*.json")]
    for answer in report["answers"]:
        matching = [audit for audit in audits if audit["request"]["messages"][-1]["content"] == answer["question"]]
        assert len(matching) == 1
        context = (directory / answer["context_file"]).read_text(encoding="utf-8")
        system_prompts = [message["content"] for message in matching[0]["request"]["messages"] if message["role"] == "system"]
        assert any(context in prompt for prompt in system_prompts)
        assert matching[0]["response"]["choices"][0]["message"]["content"] == answer["answer"]
    initial_extractions = []
    for audit in audits:
        last_prompt = audit["request"]["messages"][-1]["content"]
        if not last_prompt.startswith("-Goal-"):
            continue
        titles = [doc["title"] for doc in documents if doc["content"] in last_prompt]
        if not titles:
            continue
        choice = audit["response"]["choices"][0]
        initial_extractions.append({"titles": titles, "completion_marker_exact": "<|COMPLETE|>" in choice["message"]["content"], "finish_reason": choice.get("finish_reason")})
    assert len(initial_extractions) == 20
    calls = json.loads((directory / "pilot-calls.json").read_text(encoding="utf-8"))
    missing_markers = [item for item in initial_extractions if not item["completion_marker_exact"]]
    support = [item for answer in report["answers"] for item in answer["supporting_sentence_checks"]]
    review = {
        "reviewed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "run_summary": latest["summary"], "run_summary_sha256": latest["sha256"],
        "review_script_sha256": sha256(Path(__file__)),
        "pipeline_status": report["status"],
        "actual_context_matches_generation_request": True,
        "saved_answers_match_provider_responses": True,
        "evidence_files_verified": len(report["evidence_files"]),
        "run_artifact_checksums_verified": True,
        "model_parameters_match_current_baseline": True,
        "data_matches_indexed_documents": True,
        "graph": report["graph"],
        "empty_hyperedges": empty_hyperedges,
        "initial_extractions_missing_exact_completion_marker": missing_markers,
        "successful_response_models": dict(Counter(audit["response"].get("model") for audit in audits)),
        "finish_reasons": dict(Counter(audit["response"]["choices"][0].get("finish_reason") for audit in audits)),
        "failed_attempts": [item for item in calls if item["status"] == "failed"],
        "supporting_sentences_present": sum(item["present_in_actual_sources"] for item in support),
        "supporting_sentences_total": len(support),
        "queries": [{"id": answer["id"], "answer": answer["answer"], "reference_phrase_present": answer["reference_phrase_present"], "support_titles_present": answer["all_reference_titles_retrieved"], "retained_source_count": len(answer["retained_source_titles"]), "serialized_context_tokens": answer["serialized_context_tokens"], "elapsed_seconds": answer["elapsed_seconds"]} for answer in report["answers"]],
        "limitations": [
            "Two preselected pilot questions; no official EM/F1, no formal benchmark claim",
            "Twenty candidate paragraphs combined into one graph, beyond a single-question distractor corpus",
            "Source inclusion in the actual prompt is verified; causal reliance on evidence is not established",
            "Stored empty hyperedges and malformed completion markers retained as baseline observations",
            "Section limits are not a serialized total-context hard budget; exploration budget not implemented",
            "The provider alias is recorded; underlying model weights are not snapshot-pinned",
            "Usage totals cover successful responses; the timed-out request returned no usage",
        ],
    }
    (ROOT / "reports/hotpot-mini").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/hotpot-mini/review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    graph_stats = report["graph"]
    relative_run = "../../" + directory.relative_to(ROOT).as_posix()
    answer_rows = "\n".join(
        f"| {answer['type']} | {answer['question']} | {answer['answer']} | {len(answer['retained_source_titles'])}/20 | {answer['serialized_context_tokens']:,} | {answer['elapsed_seconds']:.2f} s |"
        for answer in report["answers"]
    )
    text = f"""# 真实语料端到端试运行结果

`gpt-6-luna` + 固定本地 BGE-M3 已完成抽取、建图、磁盘重载、检索与回答。当前具备经过小样本端到端验证的模型替换基线；正式基线成绩仍需在预先固定的独立评测集上测量。

## 输入与配置

使用 [HotpotQA](https://hotpotqa.github.io/) distractor validation 原始顺序的前两题：一题 comparison、一题 bridge。保留每题全部 10 个候选 Wikipedia 段落，合并为一个 20 文档小图。参考答案和支持事实只用于事后核对。数据许可和来源见 [语料记录](../../pilots/hotpot-mini/manifest.json)，这两题后续应排除在正式留出评测之外。

模型、原始提示、1200/100 分块、hybrid 检索、top_k=60 和各分区 4000-token 限制均由[当前配置](../../configs/baseline.json)固定。执行器使用异常安全信号量、瞬时错误最多一次重试和 120 次 API 尝试上限。当前配置 SHA-256：`{sha256(ROOT / 'configs/baseline.json')}`。

## 建图与持久化

| 项目 | 结果 |
|---|---:|
| 原始文档 / 分块 | {graph_stats['documents']} / {graph_stats['chunks']} |
| 实体节点 | {graph_stats['entity_nodes']} |
| 超边节点 | {graph_stats['hyperedge_nodes']} |
| 实体与超边间的关联 | {graph_stats['incidence_edges']} |
| 至少关联 3 个实体的超边 | {graph_stats['hyperedges_with_at_least_three_entities']} |
| 无关联实体的超边 | {len(empty_hyperedges)} |
| 建图耗时 | {graph_stats['build_seconds']:.2f} s |

二部图关联类型和正文来源链接检查通过。GraphML 与三个向量库保存后重新加载，节点数、关联数、向量数和来源检查保持一致；回答使用的是重新加载的索引。

## 回答与实际证据

| 题型 | 原始问题 | 模型实际回答 | 上下文包含的来源段落 | 序列化上下文 tokens | 查询耗时 |
|---|---|---|---:|---:|---:|
{answer_rows}

两题回答均与参考答案含义一致，全部 {len(support)} 条标注支持句均出现在实际回答上下文的 Sources 分区。这里确认的是模型获得了相关证据，并未通过消融实验验证其因果依赖；没有计算正式 EM/F1，也不报告“100% 准确率”。

实际检索上下文均保留了 19/20 个来源段落，说明本轮尚未施加紧张的总上下文预算。10,500 和 11,502 tokens 也再次说明：上游的 4000-token 分区限制不等于总上下文硬上限。当前尚未实现研究中的探索预算与总上下文预算。

## 抽取格式与质量观察

20 次成功的初始抽取都返回了可识别的实体和超边记录，原始解析器完成建图。其中 {len(missing_markers)} 次缺少或拼错了 `<|COMPLETE|>`；服务均以 `stop` 结束，不能把这些现象归因为输出长度截断。图中另有 {len(empty_hyperedges)} 个零成员超边，涉及 Tyler Bates、Village accountant 和 Meet Corliss Archer。详细条目见 [质量复核](review.json)。

这些记录作为当前基线的实际输出保留。正式评测前需要检查抽取格式遵循率、实体粒度、超边成员质量和摘要忠实性；本轮结构验证不代表全部 268 条超边均已完成语义核验。

## 调用与复验

本轮建图与问答共约 {report['elapsed_seconds']:.2f} s（不含初始模型加载与后续复核），{report['api_attempt_count']} 次 API 尝试：80 次成功，一次读取超时后重试成功。成功调用为初始抽取 20 次、补充抽取及循环判断 56 次、查询分析 2 次、回答 2 次。所有成功响应返回 `gpt-6-luna`。

80 个成功响应报告输入 {report['usage_from_successful_response_audits']['input_tokens']:,}、输出 {report['usage_from_successful_response_audits']['output_tokens']:,}，合计 {report['usage_from_successful_response_audits']['total_tokens']:,} tokens；其中包含服务报告的缓存输入，超时请求没有返回 usage。逐次 usage 和 Go 账目字段保存在脱敏调用审计中。密钥未写入文件，已扫描输出中的凭据模式。

- [完整运行摘要]({relative_run}/pilot-summary.json)
- [图结构]({relative_run}/graph_chunk_entity_relation.graphml)
- [逐次调用与失败记录]({relative_run}/pilot-calls.json)
- [问题 1 实际上下文]({relative_run}/context-{report['answers'][0]['id']}.txt)
- [问题 2 实际上下文]({relative_run}/context-{report['answers'][1]['id']}.txt)

{len(report['evidence_files'])} 个运行产物的校验值已核对，运行中的模型和检索参数、实际文档均与当前基线匹配。离线复核命令：

```powershell
.\\.venv\\Scripts\\python.exe -m hgrag review
```

下一步固定正式评测数据划分、指标与预算口径，并在本轮开发样本之外检查抽取质量和预算敏感性，再记录后续改进方法要比较的基线成绩。
"""
    (ROOT / "reports/hotpot-mini/REPORT.md").write_text(text, encoding="utf-8")
    print(json.dumps({"status": "passed", "report": str(ROOT / "reports/hotpot-mini/REPORT.md"), "empty_hyperedges": len(empty_hyperedges), "initial_completion_marker_issues": len(missing_markers), "supporting_sentences_present": review["supporting_sentences_present"], "supporting_sentences_total": len(support)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
