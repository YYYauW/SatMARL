光学遥感卫星观测与数据交付测试 v2，2026-10-18

这是现有SatMARL项目的局部代码更新，不是完整项目。
不含任何训练权重或历史运行结果，也不修改旧实验配置。
需要现有服务器项目中的轨道生成器、目录生成器和模型评估代码。
Python >=3.10。已有学习模型评估还需要原satmarl环境中的PyTorch。

1. 通过ToDesk把zip传入服务器Downloads目录。在Ubuntu终端解压到独立文件夹：

mkdir -p ~/satmarl_transmission_v2_update
unzip -n ~/Downloads/satmarl_transmission_load_v2_20261018.zip -d ~/satmarl_transmission_v2_update
conda activate satmarl

2. 先核对现有文件：

python ~/satmarl_transmission_v2_update/install.py --project ~/SatMARL

如果提示 Local changes or a different code version，先停止，不要强制覆盖。
需要比对服务器上有差异的文件后再合并。

3. 核对通过后备份并安装：

python ~/satmarl_transmission_v2_update/install.py --project ~/SatMARL --apply

安装器只更新清单中的8个文件。原有env.py和entities.py先备份至项目backups/目录。

4. 小规模流程检查：

cd ~/SatMARL
python scripts/run_transmission_load.py --run-root runs/transmission_v2_smoke --policy priority --smoke

5. 先用一个真实模型检查兼容性，结果放单独目录：

python -u scripts/run_transmission_load.py --run-root runs/transmission_v2_model_check --model-root runs/thousand_paper_suite/training --methods full --training-seeds 701 --satellites 512 --task-counts 6144 --episodes 1 --device cuda

6. 512星固定星数、不同目标数量正式评估：

python -u scripts/run_transmission_load.py --run-root runs/transmission_v2_n512 --model-root runs/thousand_paper_suite/training --methods full,mappo,no_reservation --training-seeds 701,702 --satellites 512 --task-counts 6144,12288,24576,49152 --episodes 20 --device cuda

完整参数说明：安装后docs/transmission_load_v2_zh.md。
已有模型在新资源/载荷条件下的结果属于迁移测试；本更新包不重新训练模型。
