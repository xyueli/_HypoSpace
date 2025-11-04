import json
from pathlib import Path

def convert_to_chat_jsonl(input_path: Path, output_path: Path):
    """将训练集JSON转换为Qwen LoRA微调所需JSONL格式"""
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取所有数据集
    if "datasets" in data:
        datasets = data["datasets"]
    elif "datasets_by_n_observations" in data:
        datasets = []
        for group in data["datasets_by_n_observations"].values():
            datasets.extend(group)
    else:
        raise ValueError(f"Invalid dataset format in {input_path}")

    records = []
    for ds in datasets:
        observations = ds.get("observations", [])
        ground_truths = ds.get("ground_truth_graphs", [])
        nodes = ds.get("nodes", [])
        
        # 数据验证
        if not nodes or not observations or not ground_truths:
            continue
            
        node_str = ", ".join(sorted(nodes))
        obs_lines = "\n".join([obs["string"] for obs in observations])

        user_content = (
            f"Nodes: {node_str}\n"
            f"Observations:\n{obs_lines}\n"
            f"Output a causal graph consistent with all observations:"
        )

        system_prompt = (
            "You are a causal reasoning expert. "
            "Infer causal graphs from perturbation experiments. "
            "Output format: Graph: A->B, B->C or Graph: No edges"
        )

        # 🚨 关键修正：为每个ground truth生成样本，但确保训练/测试分离
        for gt in ground_truths:
            edges = gt.get("edges", [])
            
            # 验证并格式化边
            valid_edges = []
            for u, v in edges:
                if u in nodes and v in nodes:
                    valid_edges.append(f"{u}->{v}")
            
            # 排序确保一致性
            gt_edges = sorted(valid_edges)
            gt_text = ", ".join(gt_edges) if gt_edges else "No edges"

            record = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": f"Graph: {gt_text}"}
                ]
            }
            records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ {input_path.name} → {output_path.name} ({len(records)} samples)")

if __name__ == "__main__":
    main()