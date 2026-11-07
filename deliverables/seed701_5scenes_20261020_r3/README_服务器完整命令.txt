SatMARL：24小时、训练模型701、每组5场景、16进程 — r3
日期：2026-10-20

一、本次修复内容

r2安装器把Windows混合换行版本的原始SHA256作为唯一前置身份，因此拒绝
服务器上的Git/Linux LF版文件。已对用户报告的七个文件逐一核对：原始文件
只执行CRLF到LF转换后，SHA256恰好等于服务器报告值；这七份LF字节也与
Git HEAD内容一致。它们不是参数或物理模型改变。

r3保留19个原始前置校验值，只给已证明的七个文件增加明确的LF原始SHA256
备选值。未列入的文件及未知校验值仍拒绝；不是对任意代码放宽校验。
安装器和取消器识别已核验版本；迁移准备器沿用服务器当前原始SHA，仍要求
与旧实验科学身份精确一致，不跨换行重写或替代身份，不改旧结果或服务器源文件。

本次包内原加速v2的10个源码文件保持逐字节一致；新增的兼容辅助模块与原
四个701/五场景辅助文件构成五个新文件。共15个payload文件。未安装原v2
时新增15个；已有字节一致的v2时只新增五个。已存在但内容不同的新文件仍
拒绝覆盖。此次r2安装已在前置检查失败，按报错没有复制任何文件。

二、实验设置完全保持

只测试本方法full，原训练模型701；卫星128、256、512、1024颗；任务数
1000、2000、3000、4000、5000、6000、7000、8000、9000、10000；每组
固定原场景0..4。40作业、200回合，每回合86400物理秒。申请16进程，
两张GPU各8进程，每卡最多4个1024星作业，每进程2个数值计算线程。

等面积20×50粗网格、40×250细网格和24小时释放分箱不变；同一场景各档
任务使用原任务目录的均匀嵌套前缀。5秒点目标成像、按角度计算姿态准备、
单次开机预热、1:4压缩、地面站仰角可见窗口及连续下传、滚动90分钟成像
最多30分钟等物理参数不变。原训练权重不变，不需要重新训练。

两类成功率：观测成功任务数/全部输入任务数；观测且所需数据全部下传成功
任务数/全部输入任务数。全部原25项METRICS保留。grouped.csv为5场景均值
和样本标准差(ddof=1)；grouped_across_training_seeds.csv仍为40行，模型数
为1、between-training-seed std为0。场景误差线用grouped.csv中的std。

三、上传并安装r3（全部在Ubuntu终端执行）

用ToDesk上传ZIP为：
/home/yw/satmarl_seed701_5scenes_20261020_r3.zip

不要覆盖r2解压目录，使用新的r3目录：

conda activate satmarl
mkdir -p /home/yw/satmarl_seed701_5scenes_update_20261020_r3
unzip -n /home/yw/satmarl_seed701_5scenes_20261020_r3.zip \
  -d /home/yw/satmarl_seed701_5scenes_update_20261020_r3

python /home/yw/satmarl_seed701_5scenes_update_20261020_r3/install.py \
  --project /home/yw/SatMARL

先检查输出；应看到19个前置文件、15个payload通过，已批准LF匹配7项。
随后正式新增文件：

python /home/yw/satmarl_seed701_5scenes_update_20261020_r3/install.py \
  --project /home/yw/SatMARL --apply

cd /home/yw/SatMARL

原config.py等七个文件不覆盖、不转换换行；旧模型和runs不写入。
任何未知差异仍会报错，先保留输出，不改校验值强行通过。

四、核对并取消之前的实验

根据已提供运行记录，旧并行目录为下面这个；先核对实际输出：

OLD_RUN="/home/yw/SatMARL/runs/full_uniform_parallel_20261018_081756"

python scripts/cancel_previous_transmission.py --discover

python scripts/cancel_previous_transmission.py \
  --run-root "$OLD_RUN" --inspect

确认是原80作业实验，再即时取消：

python scripts/cancel_previous_transmission.py \
  --run-root "$OLD_RUN" --apply

必须看到 Previous owner and all matching workers stopped 或
STOPPED ... no live matching owner/worker 才继续。即时取消只丢弃当前尚未
完整保存的回合，所有已保存回合、作业和旧目录都保留。精确核对UID、
脚本、运行目录、PID起始时间和父子关系，不发送SIGKILL，不批量停止Python。

如果希望等当前回合保存，用下面命令替代即时取消（可能要等待数小时）：
# python scripts/cancel_previous_transmission.py \
#   --run-root "$OLD_RUN" --after-current-episode --apply

如发现多个运行，分别核对，不要猜目录或批量停止。若旧程序已停止，inspect
应确认无活跃owner及worker，然后即可复用该目录。不要删除runner.lock。

五、准备新目录并复用符合条件的结果

SEED_RUN="$PWD/runs/full_day_seed701_5scenes_$(date +%Y%m%d_%H%M%S)"
echo "新结果目录：$SEED_RUN"

只读预览：

python scripts/prepare_transmission_701x5.py \
  --config configs/transmission_day_v3_seed701_5scenes.json \
  --run-root "$SEED_RUN" \
  --reuse-run "$OLD_RUN" \
  --model-root runs/thousand_paper_suite/training \
  --device cuda

确认通过后准备新目录：

python scripts/prepare_transmission_701x5.py \
  --config configs/transmission_day_v3_seed701_5scenes.json \
  --run-root "$SEED_RUN" \
  --reuse-run "$OLD_RUN" \
  --model-root runs/thousand_paper_suite/training \
  --device cuda --apply

如果还要复用原串行目录，可在上述两条命令都增加第二个--reuse-run：
# --reuse-run /home/yw/SatMARL/runs/full_uniform_day_20261018_074647

source目录必须全部停止；模型SHA、物理参数、任务/轨道、策略及代码要一致。
准备器仍要求旧manifest记载的每个科学源码SHA与当前原始文件精确相等，
不会迁移“原始SHA不同但规范化后相同”的目录。本次旧实验及当前文件均是
服务器LF版本，因此身份保持一致。新目录使用当前服务器原始身份，不把
Windows校验值硬写进去。配置、准备器及结果检查器与r2保持逐字节不变。
只筛选seed701、固定场景0..4；不包含702或第5..19号场景。
完整20场景作业提取前5回合后重新汇总；部分作业保留连续已保存回合并续跑。
全部原25项逐回合值保留，包括旧episode_wall_seconds；来源和文件SHA写入
迁移清单。跨目录科学结果冲突拒绝合并，不择优选分。中间缺号或场景0缺少
代表性台账时拒绝迁移。旧performance目录不迁入，不冒充新GPU运行速度。
源目录只读，不删除、不重写。不要直接改原目录episodes/seeds再续跑。

六、真实模型一致性检查

VALIDATION="$SEED_RUN.equivalence.json"

python -u scripts/check_accelerated_equivalence.py \
  --source-run "$SEED_RUN" \
  --report "$VALIDATION" \
  --device cuda --gpu 0 \
  --satellites 128,1024 --tasks 1000 \
  --transitions 40 --threads 2

看到 Passed exact real-model prefix comparison 后启动。检查使用原701权重，
小/大星座各40次事件前缀，核对动作、掩码、出价、奖励和台账。它是服务器上
真实模型的前缀比较，不是完整24小时等价性证明。报告在运行目录外，不能
覆盖已存在报告；重新检查需换报告名。

七、启动16进程，每15秒打印进度

nohup python -u scripts/run_transmission_accelerated.py \
  --config configs/transmission_day_v3_seed701_5scenes.json \
  --run-root "$SEED_RUN" --resume \
  --model-root runs/thousand_paper_suite/training \
  --validation-report "$VALIDATION" \
  --device cuda --gpus 0,1 \
  --workers-per-gpu 8 --max-large-per-gpu 4 \
  --threads-per-worker 2 --progress-seconds 15 \
  > "$SEED_RUN.log" 2>&1 &

echo "调度PID：$!"
echo "结果目录：$SEED_RUN"
echo "一致性报告：$VALIDATION"
tail -n 100 -f "$SEED_RUN.log"

Resources会给出实际槽位，申请16；受当时CPU/RAM/VRAM分配限制。--gpus为
PyTorch可见设备逻辑编号，不能擅自清空CUDA_VISIBLE_DEVICES等调度器限制。
进度打印jobs .../40、episodes .../200、running/queued/failed、每进程
sim .../24h、事件数和最近完整回合obs、obs+downlink。last_done不是当前
未完成回合成功率。Ctrl+C退出tail不停止nohup后台任务。

八、重新开终端、监控、暂停和恢复

conda activate satmarl
cd /home/yw/SatMARL
# SEED_RUN="/home/yw/SatMARL/runs/这里填实际新结果目录"
# VALIDATION="$SEED_RUN.equivalence.json"

python scripts/run_transmission_accelerated.py --run-root "$SEED_RUN" --status
watch -n 10 "python scripts/run_transmission_accelerated.py --run-root '$SEED_RUN' --status"

另一个终端看GPU：
nvidia-smi -l 5

新计算回合的耗时分解：
python scripts/report_accelerated_performance.py --run-root "$SEED_RUN"

普通暂停，等待每个在运行回合保存：
python scripts/transmission_accelerated_control.py --run-root "$SEED_RUN" --drain --apply

等drained且调度及子进程全退出，再用第七节相同命令恢复，保持SEED_RUN及
VALIDATION不变，日志重定向改为 >> "$SEED_RUN.log" 2>&1 &。不要重建日期
变量，不要重新准备同一个已存在目录。正式runner不加--reuse-run。

降为12进程：
python scripts/transmission_accelerated_control.py --run-root "$SEED_RUN" \
  --workers-per-gpu 6 --max-large-per-gpu 3 --apply

恢复申请16进程：
python scripts/transmission_accelerated_control.py --run-root "$SEED_RUN" \
  --workers-per-gpu 8 --max-large-per-gpu 4 --apply

调整不改科学设置，减并发在回合边界生效。异常锁先保留输出，不盲目删除。

九、完整结果检查与拷回

python scripts/check_transmission_701x5.py "$SEED_RUN"

应通过40作业/200唯一回合、40组统计、原25项字段与均值/样本std、单模型
between-seed std=0、两类率分母与包含关系、86400秒和滚动成像预算、数据
守恒、5份CSV的50档均匀分布/优先级/任务类型/24小时分箱、9项资产SHA、
轨道24h+lookahead覆盖、原模型及当前服务器源码SHA、迁移清单来源核验。
下传<=已生成<=已承诺；已承诺=下传+待传。未结束成像的已承诺数据可以令
已生成<已承诺，不要求二者相等。

原80作业/1600回合check_results.py不适用本次方案。
关键文件：episodes.csv、grouped.csv、grouped_across_training_seeds.csv、
jobs/*.json、scenario_balance.json、migration.json、manifest.json。

RESULT_ARCHIVE="/home/yw/satmarl_seed701_5scenes_results_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$RESULT_ARCHIVE" \
  -C "$(dirname "$SEED_RUN")" "$(basename "$SEED_RUN")"
ls -lh "$RESULT_ARCHIVE"
echo "调度日志：$SEED_RUN.log"
echo "一致性报告：$VALIDATION"

通过ToDesk把结果包、调度日志和一致性报告拷回；原权重无需重复打包。

十、证据边界

本机Windows仅有NumPy，无PyTorch/CUDA。合成测试验证安装保护、代码身份、
迁移与统计，不是正式实验成绩。Linux信号控制使用模拟进程测试，服务器
inspect必须现场核对。没有访问或停止服务器实际进程；用户执行取消命令后
才会停止。实际显存峰值、吞吐和完成时间仍以服务器运行结果为准。
