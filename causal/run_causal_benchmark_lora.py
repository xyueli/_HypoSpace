import sys
import json
import argparse
import os
import re
import yaml
import random
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from datetime import datetime
from textwrap import dedent
from collections import Counter
from scipy import stats
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from modules.models import CausalGraph
from modules.llm_interface import LLMInterface, OpenRouterLLM, OpenAILLM, AnthropicLLM
from generate_causal_dataset import PerturbationObservation, CausalDatasetGenerator

class LoRALLM(LLMInterface):
    """专门用于 LoRA 微调模型的 LLM 接口"""
    
    def __init__(self, base_model_path: str, lora_path: str, temperature: float = 0.7):
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.temperature = temperature
        
        print(f"🔹 加载基础模型: {base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        print(f"🔹 加载 LoRA 适配器: {lora_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        print("✅ LoRA 模型加载成功!")
    
    def query(self, prompt: str) -> str:
        """查询 LoRA 模型"""
        try:
            # 直接使用用户提示，不添加系统消息（因为微调时已经包含）
            formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # 减少生成长度
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    num_return_sequences=1
                )
            
            # 提取助理回复部分
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # 更精确地提取助理回复
            if "<|im_start|>assistant" in response:
                assistant_part = response.split("<|im_start|>assistant")[-1]
                if "<|im_end|>" in assistant_part:
                    assistant_part = assistant_part.split("<|im_end|>")[0].strip()
                return assistant_part.strip()
            else:
                # 如果没有找到标记，返回最后一部分
                return response.split("<|im_start|>assistant")[-1].strip() if "<|im_start|>assistant" in response else response.strip()
                
        except Exception as e:
            return f"Error querying LoRA model: {str(e)}"
    
    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        """查询并返回使用情况（简化版本）"""
        response = self.query(prompt)
        return {
            'response': response,
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'cost': 0.0
        }
    
    def get_name(self) -> str:
        return f"LoRA-Qwen3-4B({os.path.basename(self.lora_path)})"

class CausalBenchmarkEnhanced:
    """增强的因果基准测试类（与原始相同）"""
    
    def __init__(self, complete_dataset_path: Optional[str] = None, 
                 n_observations_filter: Optional[List[int]] = None,
                 gt_filter: Optional[Tuple] = None):
        self.n_observations_filter = n_observations_filter
        self.gt_filter = gt_filter
        self.filtered_observation_sets = []
        self.excluded_observation_sets = []
        
        if complete_dataset_path:
            with open(complete_dataset_path, 'r') as f:
                self.complete_dataset = json.load(f)
            
            self.metadata = self.complete_dataset.get('metadata', {})
            self.nodes = self.metadata.get('nodes', [])
            self.max_edges = self.metadata.get('max_edges', None)
            
            self.all_observation_sets = []
            if 'datasets_by_n_observations' in self.complete_dataset:
                for n_obs, datasets in self.complete_dataset['datasets_by_n_observations'].items():
                    self.all_observation_sets.extend(datasets)
            elif 'datasets' in self.complete_dataset:
                self.all_observation_sets = self.complete_dataset['datasets']
            elif 'sampled_datasets' in self.complete_dataset:
                self.all_observation_sets = self.complete_dataset['sampled_datasets']
            
            stage1_filtered = self.all_observation_sets
            
            if n_observations_filter:
                stage1_filtered = [
                    obs_set for obs_set in self.all_observation_sets
                    if obs_set.get('n_observations') in n_observations_filter
                ]
                print(f"Stage 1: Filtered to {len(stage1_filtered)} observation sets with n_observations in {n_observations_filter}")
            
            if gt_filter:
                if gt_filter[1] is not None:
                    min_gt, max_gt = gt_filter
                    self.filtered_observation_sets = [
                        obs_set for obs_set in stage1_filtered
                        if min_gt <= obs_set.get('n_compatible_graphs', 0) <= max_gt
                    ]
                    print(f"Stage 2: Filtered to {len(self.filtered_observation_sets)} observation sets with n_compatible_graphs in [{min_gt}, {max_gt}]")
                else:
                    allowed_values = gt_filter[0] if isinstance(gt_filter[0], list) else []
                    self.filtered_observation_sets = [
                        obs_set for obs_set in stage1_filtered
                        if obs_set.get('n_compatible_graphs', 0) in allowed_values
                    ]
                    print(f"Stage 2: Filtered to {len(self.filtered_observation_sets)} observation sets with n_compatible_graphs in {allowed_values}")
                
                self.excluded_observation_sets = [
                    obs_set for obs_set in self.all_observation_sets
                    if obs_set not in self.filtered_observation_sets
                ]
            else:
                self.filtered_observation_sets = stage1_filtered
                self.excluded_observation_sets = []
            
            if self.max_edges is None and self.all_observation_sets:
                max_edges_in_gts = 0
                for obs_set in self.all_observation_sets:
                    for gt in obs_set.get('ground_truth_graphs', []):
                        num_edges = len(gt.get('edges', []))
                        max_edges_in_gts = max(max_edges_in_gts, num_edges)
                self.max_edges = max_edges_in_gts
                print(f"Inferred max_edges={self.max_edges} from ground truth graphs")
            
            print(f"Loaded complete dataset with {len(self.all_observation_sets)} observation sets")
            if n_observations_filter or gt_filter:
                print(f"After filtering: {len(self.filtered_observation_sets)} observation sets meet criteria")
            print(f"Nodes: {', '.join(self.nodes)}")
            print(f"Max edges in hypothesis space: {self.max_edges if self.max_edges is not None else 'unlimited'}")
        else:
            self.complete_dataset = None
            self.all_observation_sets = []
            self.filtered_observation_sets = []
            print("Initialized empty benchmark - will generate datasets on demand")
    
    def sample_observation_sets(self, n_samples: int, seed: Optional[int] = None) -> List[Dict]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if self.n_observations_filter or self.gt_filter:
            primary_pool = self.filtered_observation_sets
            backup_pool = self.excluded_observation_sets
        else:
            primary_pool = self.all_observation_sets
            backup_pool = []
        
        sampled = []
        
        n_primary = len(primary_pool)
        if n_primary > 0:
            n_from_primary = min(n_samples, n_primary)
            sampled_primary = random.sample(primary_pool, n_from_primary)
            sampled.extend(sampled_primary)
            
            for obs_set in sampled_primary:
                obs_set['meets_filter_criteria'] = True
        
        n_still_needed = n_samples - len(sampled)
        if n_still_needed > 0 and backup_pool:
            n_backup = len(backup_pool)
            n_from_backup = min(n_still_needed, n_backup)
            
            if n_from_backup > 0:
                print(f"\nBackfilling: Only {n_primary} datasets met filter criteria.")
                print(f"  Adding {n_from_backup} randomly selected datasets from outside the filter range.")
                
                sampled_backup = random.sample(backup_pool, n_from_backup)
                
                for obs_set in sampled_backup:
                    obs_set['meets_filter_criteria'] = False
                    obs_set['backfilled'] = True
                
                sampled.extend(sampled_backup)
        
        if len(sampled) < n_samples:
            total_available = len(primary_pool) + len(backup_pool)
            print(f"\nWarning: Requested {n_samples} samples but only {total_available} total datasets available.")
            print(f"  Returning {len(sampled)} datasets.")
        
        return sampled
    
    def create_prompt(self, observations: List[Dict], prior_hypotheses: List[CausalGraph]) -> str:
        nodes_str = ", ".join(self.nodes)
        obs_block = "\n".join(obs["string"] for obs in observations)
        
        if prior_hypotheses:
            prior_lines = []
            for h in prior_hypotheses:
                edges = [f"{s}->{d}" for s, d in h.edges]
                if edges:
                    prior_lines.append("Graph: " + ", ".join(edges))
                else:
                    prior_lines.append("Graph: No edges")
            prior_block = "\n".join(prior_lines)
        else:
            prior_block = "None"
        
        constraint_info = ""
        if self.max_edges is not None:
            constraint_info = f"\nConstraint: The graph should have at most {self.max_edges} edges."
        
        prompt = f"""
        You are given observations from perturbation experiments on a causal system.
        
        Semantics:
        - When a node is perturbed, the perturbed node is 0.
        - A node is 1 if it is a downstream descendant of the perturbed node in the causal graph.
        - All other nodes are 0.
        
        Nodes: {nodes_str}{constraint_info}
        
        Observations:
        {obs_block}
        
        Prior predictions (do not repeat if avoidable):
        {prior_block}
        
        Task:
        Output a single directed acyclic graph (DAG) over the nodes above that explains all observations.
        
        Diversity rule:
        - A "diverse" graph is any valid graph whose edge set is NOT identical to any prior prediction.
        - Generate diverse graphs when possible to explore the solution space.
        
        Formatting rules:
        1) Use only the listed nodes. No self-loops. No cycles.
        2) Respond with exactly one line:
        - If there are edges: Graph: A->B, B->C
        - If there are no edges: Graph: No edges
        """
        return dedent(prompt).strip()
    
    def parse_llm_response(self, response: str) -> Optional[CausalGraph]:
        if not isinstance(response, str):
            return None
        
        s = response.replace("```", "").strip()
        
        m = re.search(r'(?i)\bgraph\s*:\s*(.+)$', s, flags=re.MULTILINE)
        line = m.group(1).strip() if m else ""
        
        if not line:
            edge_pattern = r'([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)'
            edges = re.findall(edge_pattern, s)
            if edges:
                edge_str = ", ".join([f"{src}->{dst}" for src, dst in edges])
                line = edge_str
            elif re.search(r'\b(no\s+edges?|empty|none|null)\b', s, re.I):
                line = "No edges"
            else:
                lines = s.splitlines()
                for l in lines:
                    if "->" in l or re.search(r'\b(no\s+edges?)\b', l, re.I):
                        line = l.strip()
                        break
        
        if not line and s:
            line = s.splitlines()[0].strip() if s.splitlines() else ""
        
        if not line:
            return CausalGraph(nodes=self.nodes, edges=[])
        
        if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
            line = line[1:-1].strip()
        line = (line
                .replace("→", "->")
                .replace("-->", "->")
                .replace("=>", "->")
                .rstrip(" .;"))
        
        if re.search(r'\b(no\s+edges?|empty|none|null)\b', line, re.I):
            return CausalGraph(nodes=self.nodes, edges=[])
        
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            edge_pattern = r'([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)'
            edges = re.findall(edge_pattern, s)
            if edges:
                edges = list(dict.fromkeys(edges))
                parts = [f"{src}->{dst}" for src, dst in edges]
            else:
                return CausalGraph(nodes=self.nodes, edges=[])
        
        edges = []
        for part in parts:
            m = re.fullmatch(r'([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)', part)
            if not m:
                edge_match = re.search(r'([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)', part)
                if edge_match:
                    u, v = edge_match.group(1), edge_match.group(2)
                else:
                    continue
            else:
                u, v = m.group(1), m.group(2)
                
            if u not in self.nodes or v not in self.nodes or u == v:
                continue
            edges.append((u, v))
        
        edges = list(dict.fromkeys(edges))
        
        if self.max_edges is not None and len(edges) > self.max_edges:
            return CausalGraph(nodes=self.nodes, edges=[])
        
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(edges)
        if not nx.is_directed_acyclic_graph(G):
            return CausalGraph(nodes=self.nodes, edges=[])
        
        return CausalGraph(nodes=self.nodes, edges=edges)
    
    def validate_hypothesis(self, hypothesis: CausalGraph, observations: List[Dict]) -> bool:
        for obs_dict in observations:
            perturbed_node = obs_dict['perturbed_node']
            expected_effects = obs_dict['effects']
            
            hypothesis_obs = CausalDatasetGenerator.get_perturbation_effects(hypothesis, perturbed_node)
            
            if hypothesis_obs.effects != expected_effects:
                return False
        
        return True
    
    def _classify_error(self, error_message: str) -> str:
        if "Expecting value" in error_message:
            match = re.search(r'line (\d+) column (\d+)', error_message)
            if match:
                return f"json_parse_error (line {match.group(1)}, col {match.group(2)})"
            return "json_parse_error"
        elif "Rate limit" in error_message.lower() or "rate_limit" in error_message.lower():
            return "rate_limit"
        elif "timeout" in error_message.lower():
            return "timeout"
        elif "401" in error_message or "unauthorized" in error_message.lower():
            return "auth_error"
        elif "403" in error_message or "forbidden" in error_message.lower():
            return "forbidden_error"
        elif "404" in error_message:
            return "not_found_error"
        elif "429" in error_message:
            return "rate_limit_429"
        elif "500" in error_message or "internal server error" in error_message.lower():
            return "server_error_500"
        elif "502" in error_message or "bad gateway" in error_message.lower():
            return "bad_gateway_502"
        elif "503" in error_message or "service unavailable" in error_message.lower():
            return "service_unavailable_503"
        elif "connection" in error_message.lower():
            return "connection_error"
        elif "JSONDecodeError" in error_message:
            return "json_decode_error"
        else:
            match = re.search(r'\b(\d{3})\b', error_message)
            if match:
                return f"http_error_{match.group(1)}"
            return "unknown_error"
    
    def evaluate_single_observation_set(
        self,
        llm: LLMInterface,
        observation_set: Dict,
        n_queries: int = 10,
        verbose: bool = True,
        max_retries: int = 5
    ) -> Dict:
        observations = observation_set['observations']
        ground_truth_graphs = [
            CausalGraph.from_dict(g) for g in observation_set['ground_truth_graphs']
        ]
        
        gt_hashes = {g.get_hash() for g in ground_truth_graphs}
        
        all_hypotheses = []
        valid_hypotheses = []
        unique_hashes = set()
        unique_valid_graphs = []
        all_unique_hashes = set()
        unique_all_graphs = []
        parse_success_count = 0
        
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        errors = []
        error_counts = {}

        # Debug information storage
        debug_info = {
            'first_prompt': None,
            'first_response': None,
            'first_parsed_hypothesis': None,
            'first_validation_result': None,
            'sample_responses': []
        }
        
        for i in range(n_queries):
            prompt = self.create_prompt(observations, all_hypotheses)
            
            # Store first prompt for debugging
            if i == 0:
                debug_info['first_prompt'] = prompt
            
            hypothesis = None
            query_error = None
            
            for attempt in range(max_retries):
                try:
                    if hasattr(llm, 'query_with_usage'):
                        result = llm.query_with_usage(prompt)
                        response = result['response']
                        
                        usage = result.get('usage', {})
                        total_prompt_tokens += usage.get('prompt_tokens', 0)
                        total_completion_tokens += usage.get('completion_tokens', 0)
                        total_tokens += usage.get('total_tokens', 0)
                        total_cost += result.get('cost', 0.0)
                    else:
                        response = llm.query(prompt)

                    # Store first response for debugging
                    if i == 0 and attempt == 0:
                        debug_info['first_response'] = response
                    
                    if response and response.startswith("Error querying"):
                        query_error = {
                            'query_index': i,
                            'attempt': attempt + 1,
                            'error_message': response,
                            'error_type': self._classify_error(response)
                        }
                        error_type = query_error['error_type']
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
                        continue
                    
                    hypothesis = self.parse_llm_response(response)

                    # Store first parsed hypothesis for debugging
                    if i == 0 and attempt == 0 and hypothesis:
                        debug_info['first_parsed_hypothesis'] = {
                            'nodes': hypothesis.nodes,
                            'edges': hypothesis.edges,
                            'hash': hypothesis.get_hash()
                        }

                    if hypothesis:
                        parse_success_count += 1

                        # Store sample responses for analysis
                        if len(debug_info['sample_responses']) < 5:  # Keep first 5 responses
                            debug_info['sample_responses'].append({
                                'query_index': i,
                                'response': response,
                                'parsed_nodes': hypothesis.nodes,
                                'parsed_edges': hypothesis.edges
                            })

                        break
                        
                except Exception as e:
                    query_error = {
                        'query_index': i,
                        'attempt': attempt + 1,
                        'error_message': str(e),
                        'error_type': self._classify_error(str(e))
                    }
                    if verbose:
                        print(f"  ⚠ Exception on query {i + 1}: {str(e)[:100]}")
            
            if not hypothesis and query_error:
                errors.append(query_error)
            
            if hypothesis:
                all_hypotheses.append(hypothesis)
                
                all_h_hash = hypothesis.get_hash()
                if all_h_hash not in all_unique_hashes:
                    all_unique_hashes.add(all_h_hash)
                    unique_all_graphs.append(hypothesis)
                
                is_valid = self.validate_hypothesis(hypothesis, observations)
                
                # Store first validation result for debugging
                if i == 0 and debug_info['first_validation_result'] is None:
                    debug_info['first_validation_result'] = is_valid
                    if verbose:
                        print(f"  🔍 第一个假设验证:")
                        print(f"    节点: {hypothesis.nodes}")
                        print(f"    边: {hypothesis.edges}")
                        print(f"    验证结果: {'有效' if is_valid else '无效'}")
                        
                        # 如果无效，显示为什么无效
                        if not is_valid:
                            print(f"    ❌ 无效原因分析:")
                            for obs_dict in observations:
                                perturbed_node = obs_dict['perturbed_node']
                                expected_effects = obs_dict['effects']
                                hypothesis_obs = CausalDatasetGenerator.get_perturbation_effects(hypothesis, perturbed_node)
                                if hypothesis_obs.effects != expected_effects:
                                    print(f"      观察 {perturbed_node}: 期望 {expected_effects}, 得到 {hypothesis_obs.effects}")

                if is_valid:
                    valid_hypotheses.append(hypothesis)
                    
                    h_hash = hypothesis.get_hash()
                    if h_hash not in unique_hashes:
                        unique_hashes.add(h_hash)
                        unique_valid_graphs.append(hypothesis)
        
        valid_rate = len(valid_hypotheses) / n_queries if n_queries > 0 else 0
        novelty_rate = len(unique_all_graphs) / n_queries if n_queries > 0 else 0
        parse_success_rate = parse_success_count / n_queries if n_queries > 0 else 0
        
        recovered_gts = set()
        for graph in unique_valid_graphs:
            if graph.get_hash() in gt_hashes:
                recovered_gts.add(graph.get_hash())
        
        recovery_rate = len(recovered_gts) / len(gt_hashes) if gt_hashes else 0
        
        # Print detailed debug information for the first sample
        if verbose and debug_info['first_prompt'] is not None:
            print(f"\n  🐛 详细调试信息 (第一个查询):")
            print(f"    📝 Prompt (前200字符):")
            prompt_preview = debug_info['first_prompt'][:200] + "..." if len(debug_info['first_prompt']) > 200 else debug_info['first_prompt']
            print(f"      {prompt_preview}")
            
            if debug_info['first_response']:
                print(f"    📤 模型响应:")
                response_preview = debug_info['first_response'][:200] + "..." if len(debug_info['first_response']) > 200 else debug_info['first_response']
                print(f"      {response_preview}")
            
            if debug_info['first_parsed_hypothesis']:
                print(f"    🔧 解析后的假设:")
                print(f"      节点: {debug_info['first_parsed_hypothesis']['nodes']}")
                print(f"      边: {debug_info['first_parsed_hypothesis']['edges']}")
        
            print(f"    ✅ 解析成功: {parse_success_count}/{n_queries}")
            print(f"    ✅ 有效假设: {len(valid_hypotheses)}/{n_queries}")
            print(f"    ✅ 恢复的真实图: {len(recovered_gts)}/{len(gt_hashes)}")
        
            # Show a few sample responses for analysis
            if debug_info['sample_responses']:
                print(f"    📊 样本响应分析 (前{len(debug_info['sample_responses'])}个):")
                for sample in debug_info['sample_responses'][:3]:  # Show first 3 samples
                    print(f"      查询 {sample['query_index']}:")
                    response_preview = sample['response'][:100] + "..." if len(sample['response']) > 100 else sample['response']
                    print(f"        响应: {response_preview}")
                    print(f"        解析边: {sample['parsed_edges']}")

        return {
            'observation_set_id': observation_set.get('observation_set_id', 'unknown'),
            'n_observations': len(observations),
            'n_ground_truths': len(ground_truth_graphs),
            'n_queries': n_queries,
            'n_valid': len(valid_hypotheses),
            'n_unique_valid': len(unique_valid_graphs),
            'n_unique_all': len(unique_all_graphs),
            'n_recovered_gts': len(recovered_gts),
            'parse_success_count': parse_success_count,
            'parse_success_rate': parse_success_rate,
            'valid_rate': valid_rate,
            'novelty_rate': novelty_rate,
            'recovery_rate': recovery_rate,
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens
            },
            'cost': total_cost,
            'errors': errors,
            'error_summary': {
                'total_errors': len(errors),
                'error_types': error_counts
            },
            'all_hypotheses': [h.to_dict() for h in all_hypotheses],
            'valid_hypotheses': [h.to_dict() for h in valid_hypotheses],
            'unique_graphs': [g.to_dict() for g in unique_valid_graphs],
            'debug_info': debug_info  # Include debug info in results
        }
    
    def run_benchmark(
        self,
        llm: LLMInterface,
        n_samples: int = 10,
        n_queries_per_sample: Optional[int] = None,
        query_multiplier: float = 1.0,
        seed: Optional[int] = None,
        verbose: bool = True,
        checkpoint_dir: str = "checkpoints",
        max_retries: int = 3
    ) -> Dict:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_llm_name = llm.get_name().replace('/', '_').replace('(', '_').replace(')', '_').replace(' ', '_')
        checkpoint_file = checkpoint_path / f"checkpoint_causal_enhanced_{safe_llm_name}_{run_id}.json"
        
        print(f"\nRunning Enhanced Causal Benchmark")
        print(f"LLM: {llm.get_name()}")
        print(f"Sampling {n_samples} observation sets")
        if n_queries_per_sample is not None:
            print(f"Queries per sample: {n_queries_per_sample} (fixed)")
        else:
            print(f"Queries per sample: {query_multiplier}x number of ground truths (adaptive)")
        print(f"Max retries: {max_retries}")
        print(f"Checkpoint file: {checkpoint_file}")
        print("-" * 50)
        
        sampled_sets = self.sample_observation_sets(n_samples, seed)
        
        all_results = []
        valid_rates = []
        novelty_rates = []
        recovery_rates = []
        parse_success_rates = []
        
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        all_errors = []
        total_error_counts = {}
        
        start_idx = 0
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    all_results = checkpoint_data.get('results', [])
                    start_idx = len(all_results)
                    
                    total_prompt_tokens = checkpoint_data.get('total_prompt_tokens', 0)
                    total_completion_tokens = checkpoint_data.get('total_completion_tokens', 0)
                    total_tokens = checkpoint_data.get('total_tokens', 0)
                    total_cost = checkpoint_data.get('total_cost', 0.0)
                    
                    all_errors = checkpoint_data.get('all_errors', [])
                    total_error_counts = checkpoint_data.get('total_error_counts', {})
                    
                    print(f"Resuming from checkpoint: {start_idx}/{n_samples} completed")
                    
                    for result in all_results:
                        valid_rates.append(result['valid_rate'])
                        novelty_rates.append(result['novelty_rate'])
                        recovery_rates.append(result['recovery_rate'])
                        parse_success_rates.append(result.get('parse_success_rate', 1.0))
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")
                print("Starting from beginning...")
        
        for idx in range(start_idx, len(sampled_sets)):
            obs_set = sampled_sets[idx]
            
            if verbose:
                print(f"\nSample {idx + 1}/{n_samples}")
                print(f"  Observation set ID: {obs_set.get('observation_set_id', 'unknown')}")
                print(f"  Number of observations: {len(obs_set['observations'])}")
                print(f"  Number of ground truths: {obs_set['n_compatible_graphs']}")
            
            try:
                if n_queries_per_sample is not None:
                    n_queries = n_queries_per_sample
                else:
                    n_gt = obs_set['n_compatible_graphs']
                    n_queries = max(1, int(n_gt * query_multiplier))
                    if verbose:
                        print(f"  Using {n_queries} queries ({query_multiplier}x {n_gt} ground truths)")
                
                result = self.evaluate_single_observation_set(
                    llm, obs_set, n_queries, verbose=verbose, max_retries=max_retries
                )
                
                all_results.append(result)
                valid_rates.append(result['valid_rate'])
                novelty_rates.append(result['novelty_rate'])
                recovery_rates.append(result['recovery_rate'])
                parse_success_rates.append(result['parse_success_rate'])
                
                if 'token_usage' in result:
                    total_prompt_tokens += result['token_usage']['prompt_tokens']
                    total_completion_tokens += result['token_usage']['completion_tokens']
                    total_tokens += result['token_usage']['total_tokens']
                if 'cost' in result:
                    total_cost += result['cost']
                
                if 'errors' in result and result['errors']:
                    all_errors.extend(result['errors'])
                    if 'error_summary' in result:
                        for error_type, count in result['error_summary']['error_types'].items():
                            total_error_counts[error_type] = total_error_counts.get(error_type, 0) + count
                
                if verbose:
                    print(f"  Parse success rate: {result['parse_success_rate']:.2%}")
                    print(f"  Valid rate: {result['valid_rate']:.2%}")
                    print(f"  Novelty rate: {result['novelty_rate']:.2%}")
                    print(f"  Recovery rate: {result['recovery_rate']:.2%}")
                    if result['cost'] > 0:
                        print(f"  Cost: ${result['cost']:.6f}")
                
                checkpoint_data = {
                    'run_id': run_id,
                    'llm_name': llm.get_name(),
                    'n_samples': n_samples,
                    'n_queries_per_sample': n_queries_per_sample,
                    'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
                    'seed': seed,
                    'timestamp': datetime.now().isoformat(),
                    'results': all_results,
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'total_tokens': total_tokens,
                    'total_cost': total_cost,
                    'all_errors': all_errors,
                    'total_error_counts': total_error_counts
                }
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                    
            except Exception as e:
                print(f"  Error processing sample {idx + 1}: {str(e)}")
                traceback.print_exc()
                continue
        
        def calculate_stats(rates):
            if not rates:
                return {'mean': 0, 'std': 0, 'var': 0, 'min': 0, 'max': 0}
            return {
                'mean': np.mean(rates),
                'std': np.std(rates),
                'var': np.var(rates),
                'min': np.min(rates),
                'max': np.max(rates)
            }
        
        def calculate_p_value(rates):
            if not rates or len(rates) < 2:
                return None
            t_stat, p_val = stats.ttest_1samp(rates, 0)
            return p_val
        
        final_results = {
            'run_id': run_id,
            'llm_name': llm.get_name(),
            'n_samples': len(all_results),
            'n_queries_per_sample': n_queries_per_sample,
            'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
            'query_mode': 'fixed' if n_queries_per_sample is not None else f'adaptive_{query_multiplier}x',
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'max_edges_constraint': self.max_edges,
            'statistics': {
                'parse_success_rate': {
                    **calculate_stats(parse_success_rates),
                    'p_value': calculate_p_value(parse_success_rates)
                },
                'valid_rate': {
                    **calculate_stats(valid_rates),
                    'p_value': calculate_p_value(valid_rates)
                },
                'novelty_rate': {
                    **calculate_stats(novelty_rates),
                    'p_value': calculate_p_value(novelty_rates)
                },
                'recovery_rate': {
                    **calculate_stats(recovery_rates),
                    'p_value': calculate_p_value(recovery_rates)
                }
            },
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens,
                'avg_tokens_per_sample': total_tokens / len(all_results) if all_results else 0,
                'avg_tokens_per_query': total_tokens / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'cost': {
                'total_cost': total_cost,
                'avg_cost_per_sample': total_cost / len(all_results) if all_results else 0,
                'avg_cost_per_query': total_cost / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'error_summary': {
                'total_errors': len(all_errors),
                'error_types': total_error_counts,
                'error_rate': len(all_errors) / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'per_sample_results': all_results
        }
        
        print("\n" + "=" * 60)
        print("ENHANCED BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        print(f"Samples evaluated: {len(all_results)}/{n_samples}")
        print(f"Max edges constraint: {self.max_edges if self.max_edges is not None else 'unlimited'}")
        
        for metric_name, metric_key in [('Parse Success Rate', 'parse_success_rate'),
                                        ('Valid Rate', 'valid_rate'), 
                                        ('Novelty Rate', 'novelty_rate'), 
                                        ('Recovery Rate', 'recovery_rate')]:
            stats_dict = final_results['statistics'][metric_key]
            print(f"\n{metric_name}:")
            print(f"  Mean ± Std: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}")
            print(f"  Variance: {stats_dict['var']:.3f}")
            print(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
            if stats_dict['p_value'] is not None:
                print(f"  p-value: {stats_dict['p_value']:.4f}")
        
        print(f"\nToken Usage:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Prompt tokens: {total_prompt_tokens:,}")
        print(f"  Completion tokens: {total_completion_tokens:,}")
        print(f"  Avg tokens/sample: {final_results['token_usage']['avg_tokens_per_sample']:.1f}")
        print(f"  Avg tokens/query: {final_results['token_usage']['avg_tokens_per_query']:.1f}")
        
        print(f"\nCost:")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Avg cost/sample: ${final_results['cost']['avg_cost_per_sample']:.4f}")
        print(f"  Avg cost/query: ${final_results['cost']['avg_cost_per_query']:.6f}")
        
        if all_errors:
            print(f"\nErrors:")
            print(f"  Total errors: {len(all_errors)}")
            print(f"  Error rate: {final_results['error_summary']['error_rate']:.2%}")
            if total_error_counts:
                print(f"  Error types:")
                for error_type, count in sorted(total_error_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {error_type}: {count}")
        
        print("=" * 60)
        
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                print(f"\nCleaned up checkpoint: {checkpoint_file}")
            except Exception:
                pass
        
        return final_results

def load_config(config_path: str = "config_gpt4o.yaml") -> Dict:
    """加载配置文件"""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def parse_n_observations_filter(filter_str: str) -> List[int]:
    """解析 n_observations 过滤器字符串"""
    if not filter_str:
        return []
    
    result = []
    parts = filter_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part and not part.startswith('-'):
            start, end = part.split('-')
            start, end = int(start.strip()), int(end.strip())
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    
    return sorted(list(set(result)))

def parse_gt_filter(filter_str: str) -> Tuple[Union[List[int], int, None], Optional[int]]:
    """解析 ground truth 过滤器字符串"""
    if not filter_str:
        return None, None
    
    if '-' in filter_str and not filter_str.startswith('-'):
        parts = filter_str.split('-')
        if len(parts) == 2:
            try:
                min_gt = int(parts[0].strip())
                max_gt = int(parts[1].strip())
                return min_gt, max_gt
            except ValueError:
                pass
    
    try:
        values = []
        for part in filter_str.split(','):
            values.append(int(part.strip()))
        return sorted(values), None
    except ValueError:
        print(f"Warning: Invalid GT filter format: {filter_str}")
        return None, None

def main():
    parser = argparse.ArgumentParser(
        description="Run enhanced causal discovery benchmark with LoRA model support",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("--dataset", required=True, help="Path to complete causal dataset JSON file")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--n-samples", type=int, default=30, help="Number of observation sets to sample")
    parser.add_argument("--n-observations-filter", type=str, default=None, help="Filter datasets by n_observations (e.g., '2,3,5' or '2-5')")
    parser.add_argument("--gt-filter", type=str, default=None, help="Filter datasets by number of ground truth graphs (e.g., '10-16' or '1,2,4')")
    parser.add_argument("--n-queries", type=int, default=None, help="Fixed number of queries per observation set")
    parser.add_argument("--query-multiplier", type=float, default=1.0, help="Multiplier for adaptive queries")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries per query")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    
    args = parser.parse_args()
    
    # ============================================
    # 🔍 详细调试信息开始
    # ============================================
    print("=" * 60)
    print("🔍 run_causal_benchmark_lora.py 详细调试信息")
    print("=" * 60)
    
    # 打印命令行参数
    print("📋 命令行参数:")
    print(f"  --dataset: {args.dataset}")
    print(f"  --config: {args.config}")
    print(f"  --n-samples: {args.n_samples}")
    print(f"  --n-queries: {args.n_queries}")
    print(f"  --query-multiplier: {args.query_multiplier}")
    print(f"  --seed: {args.seed}")
    print(f"  --output: {args.output}")
    
    # 打印当前工作目录
    import os
    print(f"📁 当前工作目录: {os.getcwd()}")
    print(f"📁 脚本所在目录: {os.path.dirname(os.path.abspath(__file__))}")
    
    if args.quiet:
        args.verbose = False
    
    # 加载配置文件
    print("\n🔧 加载配置文件...")
    config = load_config(args.config)
    print(f"✅ 配置文件加载成功: {args.config}")
    
    # 详细打印配置内容
    print("📄 配置文件内容:")
    print(yaml.dump(config, default_flow_style=False))
    
    llm_config = config.get('llm', {})
    print("🔍 LLM配置详情:")
    print(f"  type: {llm_config.get('type')}")
    print(f"  base_model: {llm_config.get('base_model')}")
    print(f"  model_path: {llm_config.get('model_path')}")
    print(f"  temperature: {llm_config.get('temperature')}")
    
    # 获取模型路径
    base_model_path = llm_config.get('base_model', "/opt/data/private/Qwen3-4B-Instruct-2507")
    lora_path = llm_config.get('model_path', "/opt/data/private/_HypoSpace-main/causal/finetuning/lora_output_qwen3_balanced")
    temperature = llm_config.get('temperature', 0.7)
    
    print("\n🎯 模型路径验证:")
    print(f"  基础模型路径: {base_model_path}")
    print(f"    存在: {Path(base_model_path).exists()}")
    if Path(base_model_path).exists():
        base_files = list(Path(base_model_path).glob("*"))
        print(f"    包含文件数: {len(base_files)}")
    
    print(f"  LoRA路径: {lora_path}")
    print(f"    存在: {Path(lora_path).exists()}")
    if Path(lora_path).exists():
        lora_files = list(Path(lora_path).glob("*"))
        print(f"    包含文件数: {len(lora_files)}")
        print(f"    关键文件检查:")
        key_files = ['adapter_config.json', 'adapter_model.safetensors']
        for key_file in key_files:
            key_path = Path(lora_path) / key_file
            print(f"      {key_file}: {key_path.exists()} (大小: {key_path.stat().st_size if key_path.exists() else 0} bytes)")
    
    print(f"  温度: {temperature}")
    
    # 验证路径
    if not Path(base_model_path).exists():
        print(f"❌ 错误: 基础模型路径不存在: {base_model_path}")
        sys.exit(1)
    
    if not Path(lora_path).exists():
        print(f"❌ 错误: LoRA 适配器路径不存在: {lora_path}")
        sys.exit(1)
    
    # 检查关键LoRA文件
    adapter_config = Path(lora_path) / 'adapter_config.json'
    adapter_model = Path(lora_path) / 'adapter_model.safetensors'
    
    if not adapter_config.exists():
        print(f"❌ 错误: LoRA适配器配置文件不存在: {adapter_config}")
        sys.exit(1)
    
    if not adapter_model.exists():
        print(f"❌ 错误: LoRA适配器模型文件不存在: {adapter_model}")
        sys.exit(1)
    
    # 初始化 LoRA LLM
    print("\n🚀 初始化 LoRA 模型...")
    try:
        llm = LoRALLM(base_model_path, lora_path, temperature)
        print("✅ LoRA 模型初始化成功!")
        print(f"📝 模型名称: {llm.get_name()}")
    except Exception as e:
        print(f"❌ LoRA 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 设置输出文件路径
    if args.output is None:
        dataset_name = Path(args.dataset).stem
        output_pattern = config.get('benchmark', {}).get("output_pattern", "results/{dataset_name}_lora.json")
        output = output_pattern.format(dataset_name=dataset_name)
    else:
        output = args.output
    
    print(f"\n📊 输出文件: {output}")
    
    # 解析过滤器参数
    n_observations_filter = None
    if args.n_observations_filter:
        n_observations_filter = parse_n_observations_filter(args.n_observations_filter)
        print(f"🔍 过滤 n_observations: {n_observations_filter}")
    
    gt_filter = None
    if args.gt_filter:
        gt_filter = parse_gt_filter(args.gt_filter)
        if gt_filter[0] is not None:
            if gt_filter[1] is not None:
                print(f"🔍 过滤 n_compatible_graphs: [{gt_filter[0]}, {gt_filter[1]}]")
            else:
                print(f"🔍 过滤 n_compatible_graphs: {gt_filter[0]}")
    
    # 初始化基准测试
    print("\n🎯 初始化基准测试...")
    benchmark = CausalBenchmarkEnhanced(
        args.dataset, 
        n_observations_filter=n_observations_filter,
        gt_filter=gt_filter
    )
    
    print("=" * 60)
    print("🚀 开始基准测试运行")
    print("=" * 60)
    
    # 运行基准测试
    results = benchmark.run_benchmark(
        llm=llm,
        n_samples=args.n_samples,
        n_queries_per_sample=args.n_queries,
        query_multiplier=args.query_multiplier,
        seed=args.seed,
        verbose=args.verbose,
        checkpoint_dir=args.checkpoint_dir,
        max_retries=args.max_retries
    )
    
    # 保存结果
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 最终结果保存至: {output}")
    
    # 打印最终统计摘要
    print("\n" + "=" * 60)
    print("📈 最终统计摘要")
    print("=" * 60)
    stats = results.get('statistics', {})
    for metric, values in stats.items():
        mean_val = values.get('mean', 0)
        print(f"{metric}: {mean_val:.3f}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()