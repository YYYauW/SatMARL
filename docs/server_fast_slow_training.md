# 服务器训练：Fast–Slow Opportunity Graph MARL

这套流程会依次完成环境与约束测试、三随机种子训练、同尺度 Graph
对照、时间尺度与竞合机制消融、64/128/256 星独立评估，并持续写入
JSON、CSV、检查点和浏览器训练曲线。论文实验使用的物理设定固定为
5 秒点目标观测、240 步周期和 45° FOV。

## 1. 安装

```bash
git clone https://github.com/YYYauW/SatMARL.git
cd SatMARL
git checkout agent/initial-satmarl-import

conda create -n satmarl python=3.10 -y
conda activate satmarl
python -m pip install --upgrade pip
# 先按服务器驱动版本安装 PyTorch，命令见 https://pytorch.org/get-started/locally/
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

如果 `nvidia-smi` 报 `Driver/library version mismatch`，应先重启服务器；
若仍失败，需要管理员重新安装匹配的 NVIDIA 驱动。CUDA 修复前可以把
下文的 `DEVICE=cuda` 改为 `DEVICE=cpu` 做 smoke test，但不建议用 CPU
执行完整论文训练。

## 2. 先做 smoke test

```bash
python scripts/run_fast_slow_server_pipeline.py \
  --profile smoke --device cpu \
  --run-root runs/fast_slow_smoke
```

通过标准是流水线 `status=complete`，且生成：

- `runs/fast_slow_smoke/pipeline.json`
- `runs/fast_slow_smoke/results.json`
- 每个训练目录下的 `metrics.json` 和 `checkpoints/latest.pt`
- 每个独立测试目录下的 `summary.json`、`rollout.json`、`schedule.csv`

## 3. 在 tmux 中启动完整训练

```bash
tmux new -s satmarl
conda activate satmarl
DEVICE=cuda PROFILE=paper bash scripts/start_fast_slow_server.sh
```

按 `Ctrl-b` 再按 `d` 可离开 tmux，训练不会中断。重新进入：

```bash
tmux attach -t satmarl
```

脚本会自动复用已完成结果；若进程意外中断，再运行同一命令会从
`checkpoints/latest.pt` 恢复未完成训练。

## 4. 查看实时曲线

训练服务器默认只监听 `127.0.0.1:8766`。从你的 Windows 电脑建立隧道：

```powershell
ssh -L 8766:127.0.0.1:8766 yw@服务器IP
```

浏览器打开：

`http://127.0.0.1:8766/web/fast_slow.html`

页面每 5 秒更新训练回报、完成任务数、有效任务机会率和冲突曲线，
并在测试结束后显示多种子均值、标准差、完成率和归一化冲突率。

若只能使用 ToDesk，可直接在服务器浏览器打开同一地址。确需局域网
直接访问时使用 `HOST=0.0.0.0`，同时只在防火墙中向可信 IP 开放 8766。

## 5. AAAI 对照实验

主流水线包含以下严格同预算方法：

- `fast_slow_full`：慢层每 8 步更新战略意图，快层在机会点决策；
- `graph`：无时间分层的 OASIS-Graph；
- `fast_slow_k1`：每步刷新慢层，检验时间抽象是否有效；
- `fast_slow_no_factor`：移除任务竞争因子消息；
- `fast_slow_no_bid`：移除学习型资源竞争出价。

IPPO、MAPPO、QMIX、PS-DQN 必须在相同 64 星、3,072 任务预算下重新
训练，不能用旧的 16 星检查点冒充公平主对照。可为每个种子分别运行：

```bash
python scripts/run_algorithm_suite.py \
  --base-dir runs/fast_slow_aaai/rl_baselines/seed_701 \
  --satellites 64 --tasks 3072 --planes 8 \
  --max-steps 240 --candidate-k 24 --neighbor-k 6 \
  --episodes 300 --seed 701 --eval-seed 12001 --eval-episodes 20 \
  --task-layout global_random --eval-task-layout global_random \
  --point-observation-seconds 5 --fov-deg 45 --device cuda
```

再用种子 702、703 重复。论文主表应报告种子间均值和标准差；测试场景
种子固定为 12001 起，避免训练与测试泄漏。
