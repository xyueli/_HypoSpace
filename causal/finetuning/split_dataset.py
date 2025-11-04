import json
import random
from pathlib import Path

def split_dataset(input_path, train_ratio=0.6, seed=42):
    """将因果数据集划分为训练集和测试集"""
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # 获取所有观测集
    if 'datasets_by_n_observations' in data:
        all_observation_sets = []
        for n_obs, datasets in data['datasets_by_n_observations'].items():
            all_observation_sets.extend(datasets)
    else:
        all_observation_sets = data.get('datasets', [])
    
    print(f"📊 原始数据: {len(all_observation_sets)} 个观测集")
    
    # 随机划分
    random.seed(seed)
    random.shuffle(all_observation_sets)
    split_idx = int(len(all_observation_sets) * train_ratio)
    
    train_sets = all_observation_sets[:split_idx]
    test_sets = all_observation_sets[split_idx:]
    
    # 构建训练集数据
    train_data = data.copy()
    if 'datasets_by_n_observations' in train_data:
        train_data['datasets_by_n_observations'] = {}
        for obs_set in train_sets:
            n_obs = obs_set['n_observations']
            if n_obs not in train_data['datasets_by_n_observations']:
                train_data['datasets_by_n_observations'][n_obs] = []
            train_data['datasets_by_n_observations'][n_obs].append(obs_set)
    else:
        train_data['datasets'] = train_sets
    
    # 构建测试集数据
    test_data = data.copy()
    if 'datasets_by_n_observations' in test_data:
        test_data['datasets_by_n_observations'] = {}
        for obs_set in test_sets:
            n_obs = obs_set['n_observations']
            if n_obs not in test_data['datasets_by_n_observations']:
                test_data['datasets_by_n_observations'][n_obs] = []
            test_data['datasets_by_n_observations'][n_obs].append(obs_set)
    else:
        test_data['datasets'] = test_sets
    
    # 保存文件
    input_path = Path(input_path)
    train_output = input_path.parent / f"{input_path.stem}_train.json"
    test_output = input_path.parent / f"{input_path.stem}_test.json"
    
    with open(train_output, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(test_output, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✅ 划分完成:")
    print(f"   - 训练集: {len(train_sets)} 个观测集 -> {train_output}")
    print(f"   - 测试集: {len(test_sets)} 个观测集 -> {test_output}")
    
    return train_output, test_output

def main():
    """划分所有节点的数据集"""
    # 从 finetuning 目录向上到 causal，然后进入 datasets
    base_dir = Path(__file__).parent.parent / "datasets"
    
    nodes = ["node03", "node04", "node05"]
    
    for node in nodes:
        node_dir = base_dir / node
        json_files = list(node_dir.glob("*.json"))
        
        if not json_files:
            print(f"⚠️  在 {node_dir} 中未找到JSON文件")
            continue
            
        input_file = json_files[0]  # 使用第一个JSON文件
        print(f"\n🔹 处理 {node}: {input_file.name}")
        split_dataset(input_file)

if __name__ == "__main__":
    main()