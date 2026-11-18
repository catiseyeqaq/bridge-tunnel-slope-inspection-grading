# 本地运行与复现

建议使用 Python 3.12 x86_64。从仓库根目录使用 `run.py`，原 `code/run.py` 是比赛打包入口。每次运行用一个新的输出目录。

| 运行方式 | 需要的资产 | 可验证的内容 |
| --- | --- | --- |
| 无权重示例 | Python、NumPy、Pillow、PyYAML | 两张合成图、示例分类头、风险门控、七字段和计时输出 |
| 真实 2B 起步 | 官方 2B 底座与同版本依赖 | 可先用合成样例验证模型载入和视觉前向 |
| 真实 2B 巡检 | 官方 2B、匹配的三份 NPZ 及旁路 JSON、照片及元数据 | 全图视觉、轨道分类、桥梁案例投票和风险筛选 |
| 27B 按需复核 | 上述资产、官方 27B、匹配的 LoRA、Linux 大显存设备 | 严重风险筛选、资源预检、复核与降级 |

## 1. 在空环境跑示例

```bash
git clone https://github.com/catiseyeqaq/bridge-tunnel-slope-inspection-grading.git
cd bridge-tunnel-slope-inspection-grading
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python run.py --demo --output-dir outputs/demo
```

Windows PowerShell 可直接用虚拟环境的解释器。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe -X utf8 run.py --demo --output-dir outputs/demo
```

看 `outputs/demo/result.json` 的两条七字段记录，以及 `infer_time.json`、`run_manifest.json`。这里的图片、像素编码器和分类头都是合成示例。输出用于验证程序流程，不能用于评估识别性能。

## 2. 配置真实模型

安装与硬件匹配的 PyTorch 和 torchvision，再安装仓库依赖。以下是 Windows/Linux 的两个安装例子，择一使用。赛题 PPU 镜像使用供应商配套 PyTorch、torchvision 和运行库。

```bash
# NVIDIA CUDA 13.0
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
# CPU
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
```

```bash
python -m pip install -r requirements.txt
python prepare_assets.py download embedding_2b models/embedding_2b
python run.py --demo --embedding-model models/embedding_2b --device cuda --output-dir outputs/demo-2b
```

CPU 运行把 `--device cuda` 换成 `--device cpu`，按小批量准备充足内存。真模型示例仍使用合成资产，只验证底座前向和集成。已有完整底座目录可跳过下载，使用 `python prepare_assets.py check embedding_2b PATH` 校验。

模型及安装来源见 [PyTorch 官方安装矩阵](https://pytorch.org/get-started/previous-versions/)、[2B 固定版本](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/tree/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda) 和 [27B 固定版本](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)。

## 3. 用比赛学习资产处理自己的图

从项目所有者获取授权的三份 NPZ。仓库 `code/code/data/` 有匹配的五份 JSON。可将该目录复制到 `local_assets/heads`，把三份 NPZ 分别放入对应子目录。

| 目录内相对路径 | 用途 |
| --- | --- |
| `rail_head/rail_head_v1.npz` | 轨道 22 类分类头 |
| `bridge_all_view/head_all.npz` | 桥梁有无病害、视角头与 698 条病例向量 |
| `training_support/support_bank.npz` | 与训练参考样本的相似度 |

`assets.lock.json` 给出上述文件和所有旁路 JSON 的大小与 SHA256，原训练图片不是推理依赖。以下的 `input` 目录及元数据由调用方提供。

```bash
python prepare_assets.py check learned local_assets/heads
python run.py --check-only --input-dir input --metadata input/metadata.json --assets local_assets/heads --embedding-model models/embedding_2b --output-dir outputs/inspection
python run.py --input-dir input --metadata input/metadata.json --assets local_assets/heads --embedding-model models/embedding_2b --device cuda --output-dir outputs/inspection
```

元数据示例使用虚构的桥名和文件名。每个相对路径指向一张输入图，文件名与路径尾部一致。记录顺序沿用元数据。

```json
[
  {"questionCategory":"桥梁","bridgeName":"示例桥","filename":"beam.jpg","imageRelativePath":"桥梁/beam.jpg"},
  {"questionCategory":"轨道","bridgeName":"","filename":"rail.jpg","imageRelativePath":"轨道/rail.jpg"}
]
```

省略 `--metadata` 时优先读取输入目录的 `metadata.json`，否则根据 `桥梁/<桥名>/` 与 `轨道/` 生成记录。平铺桥梁照片须提供可信身份映射。可用 `--fill-defect-location` 启用比赛归档中的文件名位置后处理。该选项只补有明确位置字样的桥梁 `defectLocation`。

## 4. 加上按需复核与 TongAgents

27B 原版使用 BF16。使用 Linux 及兼容 CUDA/PPU，进入复核时所用设备需至少 60 GiB 可用显存，并有足够的主机内存。LoRA 由项目所有者授权提供，目录包含配置和权重两份文件。

```bash
python -m pip install -r requirements-teacher.txt
python prepare_assets.py download teacher_27b models/teacher_27b
python prepare_assets.py check adapter local_assets/teacher_lora
python run.py --mode risk_gate_teacher --profile onsite --device cuda \
  --input-dir input --metadata input/metadata.json --assets local_assets/heads \
  --embedding-model models/embedding_2b --teacher-model models/teacher_27b \
  --adapter local_assets/teacher_lora --output-dir outputs/with-review
```

`--profile onsite` 采用线下预检时间预算。子进程只接收风险门控选中的图片，资源校验、拒绝建议或超时等状态见 `selective_review/summary.json` 或 `run_manifest.json`。若选中数为零，运行跳过教师启动。

要复现比赛中的 SDK 工具调用，在 Linux x86_64 的 Python 3.12 环境，通过赛事或项目授权渠道取得 TongAgents 2.7.19 wheel 与配套依赖。校验后在该环境安装 wheel，再给运行命令加 `--tongagents`。

```bash
python prepare_assets.py check tongagents /path/to/sdk-wheel-directory
python -m pip install --no-deps /path/to/sdk-wheel-directory/tongagents-2.7.19-cp312-cp312-linux_x86_64.whl
```

原赛题离线依赖恢复见 [比赛机部署归档](code/START_HERE.md)。通用入口默认用 Python 执行同一条判别管线，并记录所用执行方式。

## 5. 验收输出与归档结果

- `result.json` 按检查后的元数据输出七字段。出现数量、身份、标签或字段错误时，程序返回失败。
- `infer_time.json` 由当前执行计时。`run_manifest.json` 记录同一运行的 ID、文件 SHA、视觉覆盖、门控和复核状态。
- `--check-only` 校验输入覆盖和资产路径、大小及小文件 SHA。底座大文件全量 SHA 使用 `prepare_assets.py check`。
- `code/RELEASE_MANIFEST.json` 与 `code/evidence/` 是历史记录，`assets.lock.json` 是通用运行使用的外部资产清单。

使用已安装示例依赖的环境运行测试。Bash 用 `export PYTHONPATH="$PWD/code/code/src"`，PowerShell 用 `$env:PYTHONPATH = (Resolve-Path 'code/code/src').Path`。

```bash
python -X utf8 -m unittest discover -s code/code/tests
```

历史线下批次的 [运行清单](code/evidence/run_manifest_20261021T100704.json) 和 [发布回执](code/FINAL_PUBLISH_20261021.json) 对应 490 张图、6 张 27B 复核及 200.307 秒入口计时。回执记录了缺失桥名时的照片名回退。复现同次比赛文件需要当时获准使用的输入、元数据和完整运行资产。通用入口默认要求可信桥名。

本次本地验证在全新 Windows Python 3.12 环境跑通无权重示例，并通过 33 项单元和集成检查。真实 2B 在 Windows、RTX 5060、PyTorch 2.10.0 CUDA 13.0 环境中，配合校验一致的比赛学习资产完整处理了一张桥梁图和一张轨道图。27B 调度和回退通过模拟测试，真实双模型运行数据沿用上述比赛归档。
