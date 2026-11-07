光学卫星全天任务规划事件计时版 v3

本包用于现有SatMARL项目，可直接从原基线或已安装v2版本升级，不含训练权重或历史实验结果。
Python >=3.10。正式模型测试使用服务器原satmarl环境。
新测试默认完整24小时；成像5秒；机动按角度及角速度/加速度计算；稳定和设备启动准备按真实秒计时。
成像、数传分别管理开机段，同一开机段内不对每个目标或每个数据包重复预热。

通过ToDesk把zip传到服务器Downloads后，在Ubuntu终端输入：

mkdir -p ~/satmarl_transmission_v3_update
unzip -n ~/Downloads/satmarl_transmission_day_v3_20261018.zip -d ~/satmarl_transmission_v3_update
conda activate satmarl
python ~/satmarl_transmission_v3_update/install.py --project ~/SatMARL

核对通过后：

python ~/satmarl_transmission_v3_update/install.py --project ~/SatMARL --apply

安装前会备份需要替换的文件。遇到未识别的本地修改会停止，不能强制覆盖。
安装文件清单包含12个项目文件；不删除旧配置和旧结果。

完整24小时小规模流程检查：

cd ~/SatMARL
python scripts/run_transmission_load.py --config configs/transmission_day_v3.json --run-root runs/transmission_event_v3_daycheck --policy priority --full-day-smoke

用一个旧模型进行全天兼容性检查：

python -u scripts/run_transmission_load.py --config configs/transmission_day_v3.json --run-root runs/transmission_event_v3_model_check --model-root runs/thousand_paper_suite/training --methods full --training-seeds 701 --satellites 512 --task-counts 6144 --episodes 1 --device cuda

固定512星、改变全天目标数：

python -u scripts/run_transmission_load.py --config configs/transmission_day_v3.json --run-root runs/transmission_event_v3_n512 --model-root runs/thousand_paper_suite/training --methods full,mappo,no_reservation --training-seeds 701,702 --satellites 512 --task-counts 6144,12288,24576,49152 --episodes 20 --device cuda

配置说明位于docs/transmission_day_v3_zh.md。
60秒载荷启动、420秒数传准备是具名工程工况的参考设置，均可独立修改；420秒不代表通用X频段发射机热机时间。
本包不接管训练，也不改变原训练入口。旧权重直接进入新环境时，应将结果归类为迁移测试。
