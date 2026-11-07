本方法全球均衡任务全天测试：完整服务器操作

本次只使用satmarl_full_uniform_day_test_20261018.zip。
该包整合了已验证的均衡目录更新、指标定义、完整命令和只读结果检查程序。
项目运行文件与原均衡更新包完全相同；服务器需要已安装事件计时v3基础更新。
已安装均衡更新时，安装器可能显示0 updates needed，这是正常的。

测试设计
本方法在程序中记为full；不测试MAPPO或消融方法。
卫星数量128、256、512、1024；任务数1000--10000，每1000一档。
原训练权重701、702；各设置20个24小时场景，共80个作业、1600回合。
空间固定20x50等面积网格，每增加1000个目标，每格新增1个点；小集合嵌套于大集合。
不同回合固定区域计数、改变格内位置；最大10000个点恰好覆盖40x250细格各1点。
任务范围是全球海陆，单次/多次观测请求比例70%/30%。
请求按24小时分层释放；成像5秒，机动按角度计算；90分钟内实际成像累计不超过30分钟。
观测成功率=观测成功请求数/输入请求总数；
观测及数据下传成功率=完成全部所需观测、且全部必要数据到达地面站的请求数/相同请求总数。
两项包含关系：下传成功属于观测成功。统计截至24小时场景结束。
原权重不更新；本轮是原模型在新约束下的迁移测试。

一、通过ToDesk将ZIP传到Ubuntu的/home/yw/。
以下命令均在Ubuntu终端逐块执行。

ls -lh /home/yw/satmarl_full_uniform_day_test_20261018.zip
mkdir -p /home/yw/satmarl_full_uniform_update_20261018
unzip -n /home/yw/satmarl_full_uniform_day_test_20261018.zip -d /home/yw/satmarl_full_uniform_update_20261018
conda activate satmarl
python /home/yw/satmarl_full_uniform_update_20261018/install.py --project /home/yw/SatMARL

检查通过后才执行：

python /home/yw/satmarl_full_uniform_update_20261018/install.py --project /home/yw/SatMARL --apply

若出现Local changes、different version或prerequisite missing or modified，停止安装并保留提示。
不强制覆盖、不重置项目、不删除旧结果。

二、检查两个权重和CUDA。

cd /home/yw/SatMARL
ls -lh runs/thousand_paper_suite/training/full/seed_{701,702}/checkpoints/latest.pt
python -c 'import torch; print("torch:",torch.__version__,"CUDA:",torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA is not available"; print("GPU:",torch.cuda.get_device_name(0))'

三、若尚未完成原模型兼容性检查，先单独运行一个小规模全天模型检查。
这一步是1个24小时场景；它不计入正式1600回合的结果。
已经确认原模型兼容时可以跳过。

CHECK_DIR="$PWD/runs/full_uniform_model_check_$(date +%Y%m%d_%H%M%S)"
python -u scripts/run_transmission_load.py --config configs/transmission_day_v3_full_sweep.json --run-root "$CHECK_DIR" --model-root runs/thousand_paper_suite/training --methods full --training-seeds 701 --satellites 128 --task-counts 1000 --episodes 1 --device cuda

模型检查完成后再启动正式批量。不要仅凭小规模priority规则策略检查通过就认为神经网络已经验证。

四、正式批量测试。确认同一GPU没有另一套完整批量测试后执行一次。
当前脚本逐作业运行，不会同时启动80个GPU进程。

RUN_DIR="$PWD/runs/full_uniform_day_$(date +%Y%m%d_%H%M%S)"
echo "结果目录：$RUN_DIR"
nohup python -u scripts/run_transmission_load.py --config configs/transmission_day_v3_full_sweep.json --run-root "$RUN_DIR" --model-root runs/thousand_paper_suite/training --methods full --training-seeds 701,702 --satellites 128,256,512,1024 --task-counts 1000,2000,3000,4000,5000,6000,7000,8000,9000,10000 --episodes 20 --device cuda > "$RUN_DIR.log" 2>&1 &
echo "后台进程PID：$!"
tail -n 50 -f "$RUN_DIR.log"

Ctrl+C仅退出日志查看，不停止上述后台测试。nohup允许终端关闭后继续运行。
请记下echo显示的结果目录。
如果更换终端，先重新设置RUN_DIR为该实际目录，不要再次执行带date的新目录定义。

五、查看进度和完成标准。

cat "$RUN_DIR/progress.json"

progress.json首次在一个完整作业结束后写出，初始不存在不等于运行失败。
每个作业有20个全天回合，回合结束才打印observed和delivered统计。
完成时status=complete，completed_jobs=80，total_jobs=80。

python /home/yw/satmarl_full_uniform_update_20261018/check_results.py "$RUN_DIR"

看到PASS表示80个作业、1600条回合、40组模型合并结果、24小时终点、
两类成功率的分母/包含关系、200组空间分布统计及场景文件哈希均通过检查。
检查程序只读取结果，不修改文件、不启动新测试。

六、意外退出后的继续。
先确认原进程已退出；如果提示runner.lock，不要直接删除锁或重复启动。
保留相同权重、代码、参数及输出目录，运行上面的正式命令并增加--resume，
同时将日志重定向改为追加。实际命令为：

nohup python -u scripts/run_transmission_load.py --config configs/transmission_day_v3_full_sweep.json --run-root "$RUN_DIR" --model-root runs/thousand_paper_suite/training --methods full --training-seeds 701,702 --satellites 128,256,512,1024 --task-counts 1000,2000,3000,4000,5000,6000,7000,8000,9000,10000 --episodes 20 --device cuda --resume >> "$RUN_DIR.log" 2>&1 &

脚本跳过已完整结束的作业，未完成作业从该作业的第一个回合重新计算。
代码/权重/参数或采样协议变化时须用新的输出目录，不能混合继续。

七、结果文件。
episodes.csv：1600条逐回合结果，包含两类成功数量和成功率。
grouped.csv：80组按原训练模型分别统计的结果。
grouped_across_training_seeds.csv：40组结果，合并701/702两个模型的均值。
jobs/：每个作业的完整JSON，含首个场景的成像和任务结果明细。
scenario_balance.json：20场景x10任务数量的分布检查。
manifest.json、scenario_sha256.json：配置、模型/代码和场景哈希追溯。
scenario/：本轮实际使用的任务目录和轨道缓存。
CSV成功率取0--1，论文展示百分比时乘100。

八、完整通过检查后打包本轮结果和日志，通过ToDesk传回。
不重复打包原训练权重。

ARCHIVE="${RUN_DIR}_results_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")" "$(basename "$RUN_DIR").log"
ls -lh "$ARCHIVE"
sha256sum "$ARCHIVE"

将ls显示的tar.gz文件通过ToDesk传回Windows，保留服务器原结果目录。
