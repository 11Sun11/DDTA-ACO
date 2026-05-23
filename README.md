# DDTA-ACO
Traffic-Aware Charge-Route Planning of Electric Vehicles using Data-Driven Ant Colony Optimization


一个用于电动汽车路径规划与充电站决策的蚁群优化（ACO）研究代码仓库，集成了基于行程时间估计（TTE）与多种充电策略的仿真、评估与对比实验脚本。

**主要功能**
- 基于双层路网的 ACO 优化器（含修复策略、非线性充电模型、费用估计）。
- 使用代理/替代模型（surrogate）进行时间估计以加速评估流程。
- 多种实验入口：完整实验、带修复的完整实验、简化对比实验、SL-ACO 消融对比。

**目录结构（概要）**
- `ACO/`：ACO 算法核心（优化器、蚂蚁、网络模型等）。
- `com/`：外部比较算法与辅助模块（RH、TOA、EVPRP、MIQCP 等）。
- `data/`：样例与数据文件（网络、实例等）。
- `models/`、`surr_train/`：用于 TTE / 代理模型的代码和训练脚本。
- `main_ACO.py`, `main_ACO_porto.py`, `main_ACO_chengdu.py`：不同数据集与实验的入口脚本。

**主要入口（快速说明）**。
- `ACO_main`：ACO 实验入口，包含度量（time、cost、runT）的计算和分析，推荐用于主要实验报告与分析。
- `ACO_MIQCP`：针对简化问题或 MIQCP 对比的轻量模式（`optimizer.simple=True`），用于速度优先的批量对比。

**快速开始**
1. 准备 Python 环境（推荐 Python 3.8+，并使用虚拟环境）：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 运行默认入口：

```
- 在Porto数据上跑：

```bash
python main_ACO_porto.py  # 默认调用 ACO_main, 可在文件尾部切换入口
```

- 在成都数据上跑：

```bash
python main_ACO_chengdu.py
```

（注：脚本中通过注释/解注释或修改最后的调用行来选择实际执行的入口函数，或直接从其它脚本 import 并调用对应函数。）

**主要运行参数（常用部分）**
- ACO 相关：`--ants_num`（蚂蚁数量），`--max_iter`（最大迭代次数），`--q0`（选择概率超参），`--pho`（信息素挥发因子）等。
- 汽车/能量相关：`--Q`（电池容量），`--init`（初始 SoC），`--minn`/`--maxx`（SoC 下界/上界），`--t_l`（最大连续驾驶小时）。
- 数据/路径：`--data`（porto/chengdu），`--data_path`、`--edge_path`、`--node_path` 等。

详细参数请查看每个 `main_*.py` 文件顶部的 `argparse` 定义。

**结果输出**
- 实验结果会写入 `Draw/data/` 目录下，以 CSV/日志形式保存适应度、运行时间、收敛曲线等。

**开发和扩展提示**
- 若要切换优化器模式（choice 参数），请查看 `ACO.ACO_optimizer.__init__` 中 `choice` 的处理逻辑（不同值会影响信息素初始化/恢复策略）。
- 若要检查或替换代理模型（surrogate），查看 `surr_train/` 与 `ACO/DDTA_ACO.py` 中关于 `surr_eval` 的调用与实现。
