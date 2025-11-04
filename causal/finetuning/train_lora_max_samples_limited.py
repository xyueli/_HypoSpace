import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from pathlib import Path

# ==============================
# ⚙️ 基础配置
# ==============================
model_name = "/opt/data/private/Qwen3-4B-Instruct-2507"
output_dir = Path(__file__).parent / "lora_output_qwen3_balanced"  # 输出到当前目录

# 分层采样配置
max_samples_per_node = 100000

# ==============================
# 🚀 1. 加载模型与分词器
# ==============================
print("🔹 Loading model and tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# ==============================
# 💡 2. 配置 LoRA
# ==============================
print("🔹 Setting up LoRA configuration ...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==============================
# 📚 3. 加载并平衡数据集
# ==============================
print("🔹 Loading and balancing dataset ...")

# 从当前目录加载JSONL文件
base_dir = Path(__file__).parent
dataset_03 = load_dataset("json", data_files=str(base_dir / "causal_training_node03.jsonl"), split="train")
dataset_04 = load_dataset("json", data_files=str(base_dir / "causal_training_node04.jsonl"), split="train") 
dataset_05 = load_dataset("json", data_files=str(base_dir / "causal_training_node05.jsonl"), split="train")

print(f"📊 原始数据量统计:")
print(f"   - Node 03: {len(dataset_03):,} 样本")
print(f"   - Node 04: {len(dataset_04):,} 样本") 
print(f"   - Node 05: {len(dataset_05):,} 样本")

# 平衡采样
dataset_03_balanced = dataset_03
dataset_04_balanced = dataset_04
dataset_05_balanced = dataset_05.shuffle(seed=42).select(
    range(min(len(dataset_05), max_samples_per_node))
)

print(f"📊 处理后数据量统计:")
print(f"   - Node 03: {len(dataset_03_balanced):,} 样本")
print(f"   - Node 04: {len(dataset_04_balanced):,} 样本")
print(f"   - Node 05: {len(dataset_05_balanced):,} 样本")

# 合并数据集
dataset = concatenate_datasets([dataset_03_balanced, dataset_04_balanced, dataset_05_balanced])
dataset = dataset.shuffle(seed=42)

print(f"🎯 最终训练数据集: {len(dataset):,} 个样本")

# ==============================
# 💬 4. 格式化聊天数据
# ==============================
def format_chat(example):
    messages = example["messages"]
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return {"text": formatted}

print("🔹 Formatting chat data ...")
dataset = dataset.map(format_chat)

# ==============================
# ✂️ 5. 分词
# ==============================
def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=1024)

print("🔹 Tokenizing data ...")
tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# ==============================
# 🧠 6. 训练配置
# ==============================
print("🔹 Preparing training arguments ...")

total_samples = len(tokenized)
total_steps = total_samples // (2 * 2) * 2
warmup_steps = max(100, int(0.05 * total_steps))

args = TrainingArguments(
    output_dir=str(output_dir),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=1e-4,
    fp16=False,
    bf16=True,
    logging_steps=10, 
    save_steps=500,
    save_total_limit=2,
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",
    report_to="none",
    dataloader_pin_memory=True,
)

print(f"📈 训练参数:")
print(f"   - 总样本数: {total_samples:,}")
print(f"   - 总步数: ~{total_steps}")

# ==============================
# ⚙️ 7. Trainer
# ==============================
print("🔹 Starting training ...")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# ==============================
# 💾 8. 保存结果
# ==============================
print("✅ Saving model ...")
model.save_pretrained(str(output_dir))
tokenizer.save_pretrained(str(output_dir))

print(f"🎉 LoRA 微调完成! 模型保存在: {output_dir}")