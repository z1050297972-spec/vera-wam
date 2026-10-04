# VERA + SO-ARM101 复现指南

在 Ubuntu 上用 **SO-ARM101** 接 **VERA**（video-to-action 策略）。VERA 本体在 `vera-main/`；本指南覆盖它之外的部分：环境、机械臂接入、标定、摄像头、遥操作。

**实测配置**：Ubuntu 26.04 · 2 × RTX 3090 24GB · SO-ARM101 Pro（主臂 + 从臂）· 2 × JYU2C-2083 摄像头 · Miniconda3 在 `~/miniconda3`。

---

## 0. 先读：两个环境不能合并

| 环境 | Python | torch | 用途 |
|---|---|---|---|
| `vera` | 3.11 | 2.6.0+cu124 | VERA 本体、机械臂工具 `soarm/` |
| `lerobot` | 3.12 | 2.10.0+cu126 | 遥操作、录数据 |

**两重硬冲突**：VERA 钉死 `torch==2.6.0` 而 lerobot 要 `torch>=2.7`；lerobot 要求 Python ≥3.12 且源码用了 PEP 695 语法，在 3.11 上连解析都失败。

所以 `soarm/` **零 lerobot / 零 torch 依赖**（只拉 `feetech-servo-sdk` 和 `pyserial`），装在 `vera` 环境里独立可用，标定产物与 lerobot 完全兼容。

**权限**：串口需要 `dialout` 组，`sudo usermod -aG dialout $USER` 后**必须重新登录**（组成员在登录时固定）。临时可用 `sg dialout -c 'python -m soarm arms'`。下文命令都假设已登录进组。

---

## 1. 安装

```bash
# vera 环境
conda create -n vera python=3.11 -y && conda activate vera
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
git clone git@github.com:sizhe-li/VERA.git && cd VERA
pip install -e ".[idm,video,eval]" && pip install lpips      # ★ lpips 必须补装，见下
pip install feetech-servo-sdk
python -c "import vera, vera.policy, vera.idm, vera.server; print('ok')"

# lerobot 环境（遥操作/录数据）
conda create -n lerobot python=3.12 -y && conda activate lerobot
git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot && pip install -e ".[feetech]"
```

之后用绝对路径调 lerobot 的命令即可：`~/miniconda3/envs/lerobot/bin/lerobot-teleoperate`。

> **`lpips` 为什么必须补装**：VERA 钉死 `torchmetrics==0.11.4`，该版本的 `torchmetrics.image` 是条件导出 —— 只有装了 `lpips` 才导出 `LearnedPerceptualImagePatchSimilarity`，而 `vera/utils/logging_utils.py` 无条件导入它。不补就报 `ImportError: cannot import name 'LearnedPerceptualImagePatchSimilarity'`，且装任何 extra 都不会带上它。

> 不要用 apt 装 NVIDIA 驱动或 `cuda`；驱动装好即可，PyTorch 自带 CUDA runtime。数据集很大，留 300GB+。

---

## 2. 硬件检查

### 2.1 ⚠️ 主臂从臂供电电压不同

Pro 版两条臂的舵机规格不同，需要**两个电源**（外观一样，只看外壳电压标签区分）：

| | 舵机 | 供电 | 实测 | EEPROM 电压上限 |
|---|---|---|---|---|
| **主臂 leader** | 7.4V | **5V** | 5.2V | 12.0V |
| **从臂 follower** | 12V | **12V** | 12.2V | 14.0V |

**接错的后果**：主臂接 12V 会过压，6 个舵机全部报电压错误（`Status` 位 `0x01`）。而 **lerobot 此时的报错完全指错方向** —— 它会丢弃带错误位的舵机，于是过压的舵机在它眼里等于不存在，报成 `Missing motor IDs`，看起来像"没插电"。**所以诊断一律用 `soarm check`，不要只信 lerobot 的报错。**

> 标准版套装两条臂都是 7.4V，两个都用 5V。判据看 `Max_Voltage_Limit`：14.0V 是 12V 舵机，8.0V 附近是 7.4V 舵机。

### 2.2 端口：必须用 by-id 路径

两条臂各有一块 WCH CH343 适配器（`1a86:55d3`），枚举成 `/dev/ttyACM*`。**编号由枚举顺序决定，重新插拔就可能互换、主从会反过来**，所以不要用 `/dev/ttyACM0`。用稳定路径：

```bash
python -m soarm arms        # 自动发现 + 按电压判主从，缓存到 ~/.cache/soarm/arms.json
```

```
  ✅ 自动判定: leader = 5B3D048490 (7.4V 舵机)  follower = 5B3D042389 (12V 舵机)
```

判定失败时（标准版两条臂规格相同，或某条没上电），拔掉其中一条的 USB 看哪个节点消失，手写缓存：

```bash
mkdir -p ~/.cache/soarm && echo '{"leader":"序列号","follower":"序列号"}' > ~/.cache/soarm/arms.json
```

### 2.3 自检

```bash
python -m soarm check all      # 逐个读电压/温度/错误位
```

`✅ 全部正常` → 可以标定。`★错误位 0x01` → 供电接错（见 2.1）。`❌ 无响应` → 舵机没上电或总线断开，用 `python -m soarm probe follower` 进一步查。

---

## 3. 标定

每条臂两趟，`--id` 要和遥操作的 `--robot.id` / `--teleop.id` 一致：

```bash
python -m soarm measure  follower                             # ① 手推每个关节走满行程（按 x 结束，无时限）
python -m soarm finalize follower --id my_follower --dry-run  # ② 空跑
python -m soarm finalize follower --id my_follower            # 写入

python -m soarm measure leader && python -m soarm finalize leader --id my_leader
python -m soarm compare                 # 两条臂的 2047位置 都应是 50%
```

**不需要像 lerobot 那样"标定前把两条臂摆成同一姿态"**。lerobot 的零点取自"你摆的那个位置"，所以必须摆准（它官方文档原话就是 "ensure that the leader and follower arms have the same position values when they are in the same physical position"）；我们的零点取自**各自实测行程的中点**，主从对应关系是自动成立的。

**只有一个关节需要摆位**：`wrist_roll` 是整圈旋转、没有行程中点，零点取自 `finalize` 那一刻的读数 —— 所以**两条臂的 `wrist_roll` 要摆成同一个物理方向**（比如夹爪开口都朝上），再跑 `finalize`。其余 5 个关节摆得准不准都不影响。想让工具逐关节引导你摆位时才用 `python -m soarm mid <role>`（可选项）。

**这个零点错位的症状与修复**：采数据时 observation 的最低限位和 action 对不上（实测 action 到 −98.9°、observation 只到 −80.1°，差 18.8° —— 从臂比主臂提前顶住，那一段 action 从臂物理上执行不了）。修它只动从臂这一个关节：

```bash
python -m soarm fix-wrist --id my_follower            # 两臂手腕推到同一对方向，各按一次 x（先加 --dry-run 可只测不写）
```

它测两个共同方向上的读数差（纯零点差是常数，两端应一致）再写入 EEPROM **和 lerobot 的标定 JSON** —— 两处必须一起改，只改舵机的话下次连接会被文件里的旧值写回去。改完把两臂手腕摆到同一方向，显示的度数应该一致。

### 三个坑

**① 写 `Goal_Position` 会让固件自动开力矩**（实测 6/6）。而 Goal 在 SRAM、上电默认 0，所以每次断电重启后开力矩，臂会朝刻度 0 猛冲。遥操作前先 `python -m soarm prep follower`（它先对齐 Goal 再关力矩）。

**② 行程可能跨过编码器 0 点**，直接记 min/max 会算成接近 360°（实测 353.8°，真值 187°）。本工具按最短弧展开已绕开，但你仍要保证**推满**，否则行程不对。

**③ `finalize` 输出的 `2047位置` 那一列应该全是 50%**。某项偏得多 = 那条臂 `measure` 没推满，重跑。

```bash
python -m soarm verify follower --id my_follower          # 只读：位置 vs 标定量程
python -m soarm verify follower --id my_follower --move   # 小幅运动测试（会动，先清空工作空间）
```

---

## 4. 主从对齐（可选的校验 / 恢复工具）

**标定做对了，日常用不上这一章。** 因为零点取自各自实测行程的中点，主从对应关系已经自动成立（见第 3 章）。`align` 只在两种时候需要：

```bash
python -m soarm align --dry-run  # ① 校验：看各关节偏差是否都接近 0
python -m soarm align            # ② 恢复：从臂被手动挪过 / 断电重启过之后，拉回主臂姿态
```

- **校验**：这是对标定质量的独立检验。某个关节差得多，说明那条臂 `measure` 没测准，回去重跑。
- **恢复**：`align` 结束时**从臂保持力矩、停在主臂姿态**，可以直接起遥操作 —— 但**不要再跑 `prep`**，关掉力矩从臂会因重力下垂，对齐就白做了。

（lerobot 没有这个功能：它的主从对应关系只在标定那一刻建立，之后臂被动过就只能重新标定。）

> **实测坑**：带负载的关节有明显静差（从臂 elbow 托着前臂时，舵机停住的位置比命令值差 **30 刻度 = 2.6°**）。每周期只把命令值往前推一小步，这一步整个落在静差里 → 舵机认为"已经到了"→ **关节永远不动**。所以工具内部是**积分式**下发：命令值持续朝目标累积，直到实测位置追上。这是本工具和 lerobot 的 `max_relative_target` 最实质的区别。

---

## 5. 摄像头

```bash
python -m soarm cam --list       # 有几只、分别能不能出帧
python -m soarm cam --lock       # 锁曝光模式/白平衡/对焦
python -m soarm cam              # ★ 锁完再确认：应显示"满足要求"且无"还没锁的项"
python -m soarm cam --simul      # 多相机同时开流体检
python -m soarm cam --preview    # 本地预览 http://127.0.0.1:8099
```

**采数据前必须 `--lock` 再 `cam` 确认。** 出厂是"光圈优先自动曝光"，静态场景 10 秒画面亮度能漂 **42 级** —— 光流和视频模型分不清"物体在动"和"整幅画面变亮"，锁定后极差 0.2 级。

**这些设置会自己丢**：拔插一定回出厂值，实测还出现过没拔插也退回的情况。所以别假定锁上了。

采集用 **MJPG 1280×720**（实测 29.8 fps 零掉帧；YUYV 720p 只有 9.9 fps，带宽不够）。`--preview` 会独占摄像头，录数据前 `pkill -f "soarm cam"`。

---

## 6. 遥操作

```bash
python -m soarm prep follower      # 对齐 Goal 并关力矩（断电重启后必做）

~/miniconda3/envs/lerobot/bin/lerobot-teleoperate \
  --robot.type=so101_follower --robot.id=my_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D042389-if00 \
  --robot.max_relative_target=10 \
  --teleop.type=so101_leader --teleop.id=my_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D048490-if00
```

序列号随机器不同，打印自己的：`python -c "from soarm import port_of; print(port_of('follower')); print(port_of('leader'))"`。

> 遥操作启动时从臂会朝主臂姿态移动，`--robot.max_relative_target` 只是**限速不是限位**。姿态差得多，启动那几秒就会走多远 —— 嫌冲的话先 `python -m soarm align` 把从臂带过去（第 4 章，可选）。

> ⚠️ 必须在有 `dialout` 组的 shell 里跑，并关掉 client 端 keepalive（`ping_interval=None`）。缺组时报的不是权限错误：pyserial 的 `PermissionError` 被 lerobot 的 `except OSError` 吞成 `ConnectionError: Could not connect on port`，端口路径通常是对的，别照着提示去换端口。

---

## 7. VERA 接入（进行中）

**已完成**：环境 ✅ / 机械臂与标定 ✅ / 遥操作 ✅ / 双摄像头 ✅ / MimicGen planner checkpoint（3.2GB）✅

**VERA 是两段式**：视频规划器（世界模型，看画面幻想未来，**与机械臂无关**）→ 雅可比 IDM（逐具身训练，把光流翻译成关节动作）。

接 SO-ARM 需要四件事，**唯一绕不过去的是数据**（IDM 必须用自己采的数据训，别家 checkpoint 维度不对）：

| | 内容 | 状态 |
|---|---|---|
| ① 采数据 | 遥操作录轨迹 → 打包成 VERA 格式（JPEG + MegaFlow 光流 + 关节轨迹）| ✅ 13 集已录并打包 |
| ② 训 IDM | 新增 dataset yaml + config，6 维关节 + 夹爪 | ✅ 配置已就绪，待开训 |
| ③ 服务端 | 加 `soarm` embodiment 到 `adapter_factory` 注册表 | 待做 |
| ④ 机械臂侧 | 写 `SoArmBackend`（实现 `get_state` / `apply_action`）| 待做 |

**当前任务：抓起方块放到另一个方块上**（MimicGen 的经典任务，与 planner 的训练域一致）。

### 7.1 先修环境（两处，不修跑不起来）

```bash
pip install -U "huggingface_hub>=1.5,<2"      # transformers 5.17 要求；旧版会报 is_offline_mode 缺失
pip install --no-deps --ignore-requires-python zstandard git+https://github.com/cvg/megaflow.git
pip install --no-deps natsort
```

三条都**不能带上 torch**：megaflow 声明 `torch>=2.7`，直接 `pip install` 会把 VERA 钉死的 torch 2.6.0 换成 2.14 并拖进整套 CUDA 13，环境报废。装之前先 `--dry-run` 看清单。

另外本机 **git 全局配了 `http.proxy=127.0.0.1:7897` 但代理没在跑**，`git clone` / `pip install git+...` 会失败（curl 不受影响）。临时绕过：

```bash
export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=http.proxy GIT_CONFIG_VALUE_0= \
       GIT_CONFIG_KEY_1=https.proxy GIT_CONFIG_VALUE_1=
```

### 7.2 打包（lerobot 数据集 → VERA 格式）

IDM **不能直接吃 mp4**，必须先打包（光流只有 packed 路径实现了）。13 集、约 8 分钟/集：

```bash
export VERA_DATA_PREFIX=/home/shawn/桌面/vera-wam/data
cd ~/桌面/vera-wam/vera-main
# 两张卡并行（第二段要另开一个终端，--device cuda:1 --start-index 7 --num-episodes 6）
python scripts/data/pack_lerobot.py \
  --source-root $VERA_DATA_PREFIX/soarm-pickplace \
  --output-root $VERA_DATA_PREFIX/datasets/jacobian/soarm_packed \
  --views desk --temporal-stride 3 --flow-resolution 256 256 \
  --device cuda:0 --start-index 0 --num-episodes 7 --skip-index
# 全部打完后跑一次（不带 --skip-index）写 index.json
python scripts/data/pack_lerobot.py --source-root ... --output-root ... --views desk
```

**打完必须验证**（多集切片出过错位且不报错，见排错表）：

```bash
python scripts/data/check_packed.py \
  --source-root $VERA_DATA_PREFIX/soarm-pickplace \
  --packed-root $VERA_DATA_PREFIX/datasets/jacobian/soarm_packed \
  --views desk --image-size 128 128 --fps 30 --temporal-stride 3 \
  --stats-out ../measure/soarm_stats.yaml
```

它逐集比对打包帧与 lerobot 原始帧（每集取首/中/末三帧，**搜索 ±3 帧偏移并要求偏移 0 严格最优**——绝对阈值区分不了"错一帧"和 JPEG 噪声）、核对 `traj_state` 与 parquet 逐位一致、检查 flow 条目数 = (T−1)×视图数，并输出归一化常数。当前结果：**39/39 帧对齐、13 集轨迹差 0.00000**。

- **`--views desk`（只手腕视角）是刻意的**：固定机位那个视角的光流是噪声——方块在 256×256 里只有 ~11 像素宽，0.1 秒内位移远小于一个像素，实测 `|f|` 均值 0.05 px、只有 1% 的像素超过 0.5 px（手腕视角是 5.38 px / 73%）。提高分辨率也救不回来。
- `--temporal-stride 3`（30→10 fps）必须在算光流**之前**做，这样光流和 `du` 描述同一时间步，与发布配置的 du/flow 量级一致。

### 7.3 训 IDM

```bash
export VERA_DATA_PREFIX=/home/shawn/桌面/vera-wam/data
cd ~/桌面/vera-wam/vera-main
python -m vera.main --config-name=config_jacobian_soarm_vggt wandb.mode=disabled
```

配置在 `vera-main/vera/configurations/`：`dataset/soarm_packed.yaml`（含实测的 `action_abs_scale` / `oflow_abs_scale`）、`algorithm/model/soarm_vggt_jacobian.yaml`（**`command_dim: 6`** —— 动作维度只在这里出现，不会从数据推断）、`config_jacobian_soarm_vggt.yaml`。已实测：单卡 3090、batch 4、4 个 worker、**3.4 it/s**。

**先跑一次过拟合自检**（几分钟，验证数据真能驱动模型，而不只是"能跑"）：

```bash
python scripts/data/overfit_check.py --batches 2 --steps 150 --lr 1e-4
```

它取固定的几个 batch 反复梯度下降并打印损失。实测 150 步 **0.73 → −1.88**；损失不降就说明数据或接线有问题，比等到训练结束才发现便宜得多。

两个约束：每路画面会被 resize 到 **128×128**（发布的 MimicGen 配方是方形，loader 直接拉伸、不保宽高比）；IDM 出的是**关节增量 du**，下发要用第 4 节的积分式，不能每步写 `Goal = 上次Goal + du`。

### 7.4 为跑通这条链路改掉的四处

都是实测踩出来的，不是预防性改动：

1. **冻结 VGGT 骨干**（`freeze_aggregator: true`）。12.6 亿参数不冻结时，AdamW 的 fp32 主权重+梯度+两个动量约 20 GB，3090 上 batch 2 就 OOM；冻结后只训 DPT 解码器（3280 万参数，82 个模块 train / 1380 个 eval），batch 4 轻松。发布的 PushT IDM 用的是同一个取舍。
2. **DataLoader worker 改用 `spawn`**。默认 `fork` 会让子进程继承父进程已初始化的 CUDA 上下文，worker 里任何 CUDA 调用都会 `CUDA error: initialization error`，PyTorch 只在稍后报成 `DataLoader worker exited unexpectedly`。`base_data_module.py` 现在默认 `multiprocessing_context: spawn`，可按 split 覆盖回 `fork`。
3. **`worker_init_fn` 改成可 pickle 对象**。原本是个 lambda，`spawn` 无法序列化（`Can't pickle local object ...<lambda>`）。
4. **`JACOBIAN_COLORMAP` 加 `soarm`**（6 色）。可视化调色板按具身名索引，缺键会在第一次验证时 `KeyError` 直接崩。同理 `image_jacobian.py` 里有一处 `self.logger.experiment.log` 没做空守卫，`wandb.mode=disabled` 时 `self.logger` 是 None，会打崩整个验证过程。

> checkpoint 每个 **4 GB**（含全部参数），`save_top_k: 1` + `save_last` → 同时占约 8 GB。磁盘紧张时调小 `every_n_train_steps` 或关掉 `save_last`。

---

## 8. 排错

| 现象 | 处理 |
|---|---|
| lerobot 报 `Missing motor IDs` | **多半是过压**（2.1），用 `soarm check` 看错误位 |
| lerobot 报 `Could not connect on port` 但端口存在 | dialout 权限，报错被吞了，重新登录 |
| `check` 报无响应 | 舵机没上电或总线断开，用 `probe` 进一步查 |
| 通电后掰不动臂 | 力矩被留着：`soarm prep <role>` 会先对齐 Goal 再关力矩 |
| `measure` 报行程 > 300° | **有另一个进程在读同一条总线**（`pgrep -af soarm`），清掉后重测 |
| `compare` 某关节不是 50% | 那条臂 `measure` 没推满，重跑 |
| `align` 中某关节纹丝不动 | 顶到限位或被挡住，工具会标 `★没动（顶住/被挡住）` |
| `align` 显示 `★主臂超量程` | 主臂那个关节摆到自己行程外了，目标被夹在从臂端点 |
| 对齐完起遥操作，从臂又往下掉 | 中间跑过 `prep`。`align` 之后直接起遥操作 |
| 画面随手臂动作整体变亮 | 自动曝光没锁，`soarm cam --lock` 后再 `cam` 确认 |
| `cam` 报打不开但设备在 | 别的程序占着，最常见是 `--preview` 还开着 |
| `hg download` 卡住 / `Fetching 0 files` | 直连 huggingface.co 在本网络不通，用 `HF_ENDPOINT=https://hf-mirror.com` |
| 打包跑完但训练结果很怪 | **先跑 `check_packed.py`**：多集 mp4 切片极易静默错位（本项目就发生过 6 集中 4 集错位、却能正常加载） |
| 训练报 `Can't pickle local object ...<lambda>` | DataLoader 用了 `spawn` 但 `worker_init_fn` 是 lambda，换成可 pickle 对象 |
| 训练报 `CUDA error: initialization error` / worker 意外退出 | fork 出来的 worker 继承了父进程的 CUDA 上下文，设 `multiprocessing_context=spawn` |
| 训练 OOM（12.6 亿参数的模型） | `algorithm.model.freeze_aggregator=true` 只训解码器 |
| `wandb.mode=disabled` 时验证崩溃 | 已修（`image_jacobian.py` 加了空守卫）；若仍崩，看是不是缺该具身的 `JACOBIAN_COLORMAP` 配色 |
| Hydra 报 `not in struct` | 键名写错或该键不存在（例：`val_every_n_step` 在 `experiment.validation` 下，不在 `training` 下） |