# VERA + SO-ARM101 复现指南

在 Ubuntu 上用 **SO-ARM101** 机械臂复现 **VERA**（video-to-action 策略）的完整流程。全部操作在**终端**里执行。

VERA 本体在 `vera-main/`（原仓库，仿真任务开箱可用）。本指南覆盖 VERA 之外的部分：环境、机械臂接入、标定 —— 也就是"让真实机械臂能被驱动、能采数据"这一段。

**适用配置**（本指南实测于此）：

| 项 | 配置 |
|---|---|
| 系统 | Ubuntu 26.04 LTS |
| GPU | 2 × RTX 3090 24GB，驱动 595.84 |
| 机械臂 | SO-ARM101 Pro（主臂 + 从臂），当前**单臂**使用 |
| 摄像头 | 2 个 USB 摄像头（JoyandAI JYU2C-2083，同型号），序列号 `…2605004`（俯拍桌面）+ `…2605007`（广角全景）；可同时开流 |
| conda | Miniconda3 装在 `~/miniconda3` |

---

## 目录

- [0. 先读：两个环境，不能合并](#0-先读两个环境不能合并)
- [1. 环境安装](#1-环境安装)
- [2. 机械臂硬件](#2-机械臂硬件)
- [3. 标定（两趟法）](#3-标定两趟法)
- [4. 验证与遥操作](#4-验证与遥操作)
- [5. 换台机器复现](#5-换台机器复现)
- [6. 排错](#6-排错)

---

## 0. 先读：两个环境，不能合并

这是本项目最大的环境坑，先说清楚，否则后面必卡。

| 环境 | Python | torch | 用途 |
|---|---|---|---|
| `vera` | 3.11 | **2.6.0+cu124** | VERA 本体、机械臂工具（`soarm/`）|
| `lerobot` | 3.12 | 2.10.0+cu126 | 遥操作、录数据 |

**两重硬冲突，装不进同一个环境**：

1. **torch**：VERA 在 `pyproject.toml` 里钉死 `torch==2.6.0`（flash-attn 的 ABI 绑定），lerobot 要求 `torch>=2.7,<2.12`。
2. **Python**：lerobot 声明 `requires-python = ">=3.12"`，且源码用了 **PEP 695 语法**（`type NameOrID = str | int`，在 `motors_bus.py` 里）。在 3.11 上**连解析都失败**，不是依赖声明问题，绕不过去。

**所以本指南的工具分两边**：

- `soarm/` 这个包只依赖 `feetech-servo-sdk`（纯 Python，只拉 `pyserial`），**不需要 lerobot 或 torch**，因此装在 `vera` 环境里就能用。标定产物与 lerobot 完全兼容。
- 遥操作/录数据才需要切到 `lerobot` 环境。

### `soarm/` 包结构

按功能拆成独立模块，可以只当命令行用，也可以 `import` 到自己的脚本里：

| 文件 | 职责 |
|---|---|
| `constants.py` | 常量、寄存器地址、角色定义、缓存与标定文件路径 |
| `encoding.py` | 符号幅值编码、刻度↔角度换算（**与 lerobot 逐位等价**）|
| `ports.py` | 适配器发现、按电压判定主从、端口解析与缓存 |
| `bus.py` | `Arm` 类：串口会话与寄存器读写 |
| `hardware.py` | `check_arm` / `deep_probe` / `read_state` / `prep_goal_safe` |
| `calibration.py` | 两趟标定（`measure_travel` / `show_angles_to_midpoints` / `finalize_two_pass`）、`verify_arm`、`compare_arms` |
| `align.py` | 主从姿态对齐：读主臂位姿，把从臂驱动到同一个姿态（`align_arms`）|
| `camera.py` | 摄像头：多设备发现 / 设置锁定 / 单机与多路体检 / 存帧 / 本地预览（`run_camera`、`check_simultaneous`）|
| `ui.py` | 终端实时刷新（非 tty 时退化为定期打印）|
| `cli.py` | 命令行入口与子命令 |
| `soarm.py` | 兼容旧写法的薄壳，等价于 `python -m soarm` |

命令行在仓库根目录运行：

```bash
sg dialout -c "python -m soarm --help"          # 推荐
sg dialout -c "python soarm/soarm.py --help"    # 等价
```

当库用：

```python
from soarm import Arm, check_arm, prep_goal_safe, measure_travel, ticks_to_deg
```

> **为什么不把两者合一**：唯一的理论路径是新建 Python 3.12 环境 + torch 2.6.0 + VERA + lerobot（`--no-deps --ignore-requires-python`）。不推荐 —— 要把 VERA 整套重依赖（vggt/deepspeed 等）在 3.12 上重装，解析失败风险高，且 lerobot 会跑在它未声明的 torch 版本上。而且 VERA 的真实机械臂部署本来就是**两台机器**（GPU 跑 server、机械臂主机跑 runner，靠 msgpack over websocket 通信），环境隔离是设计的一部分，不是缺陷。

---

## 1. 环境安装

### 1.1 检查系统与驱动

```bash
sg dialout -c "nvidia-smi"                     # 应列出所有卡
sg dialout -c "lsb_release -a"                 # 系统版本
sg dialout -c "df -h /"                        # VERA 检查点和数据集很大，建议留 300GB+
```

在原生 Ubuntu 上**不要**用 apt 装 NVIDIA 驱动，也不要单独 `apt install cuda` —— 装好驱动即可，PyTorch 自带 CUDA runtime。

### 1.2 Miniconda

```bash
sg dialout -c "cd ~ && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
sg dialout -c "ls -lh Miniconda3-latest-Linux-x86_64.sh"        # 应约 140 MB

sg dialout -c "bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3"
sg dialout -c "$HOME/miniconda3/bin/conda init bash"
sg dialout -c "exec bash"                                       # 或关掉终端重开
```

### 1.3 vera 环境

```bash
sg dialout -c "conda create -n vera python=3.11 -y"
sg dialout -c "conda activate vera && pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124"

# 校验
sg dialout -c "conda activate vera && python -c \"import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())\""
# 期望: 2.6.0+cu124 True 2
```

### 1.4 VERA 本体

```bash
sg dialout -c "git clone git@github.com:sizhe-li/VERA.git"
sg dialout -c "cd VERA && pip install -e '.[idm,video]'"
sg dialout -c "cd VERA && pip install -e '.[eval]'"

# ★ 必须补装，否则 import vera.idm 失败（VERA 依赖清单漏了它）
sg dialout -c "cd VERA && pip install lpips"
```

> **为什么必须补 `lpips`**：VERA 钉死 `torchmetrics==0.11.4`，该版本的 `torchmetrics.image` 是**条件导出** —— 只有装了 `lpips` 才会导出 `LearnedPerceptualImagePatchSimilarity`；而 `vera/utils/logging_utils.py` 无条件导入这个类，于是整条导入链断掉：
>
> ```
> ImportError: cannot import name 'LearnedPerceptualImagePatchSimilarity' from 'torchmetrics.image'
> ```
>
> 作者其实知道这类问题（专门为核心依赖补了 `torch-fidelity` 并注释 "torchmetrics' FID module imports it at module scope"），只是漏了 LPIPS 对应的包。`lpips` 不在 `pyproject.toml` 的任何 extra 里，装 `[idm,video]` 和 `[eval]` 都不会带上它。补装只增加 `lpips==0.1.4` 一个包，不动 torch。

校验：

```bash
sg dialout -c "conda activate vera && python -c \"import vera, vera.policy, vera.idm, vera.server; print('vera ok')\""
```

### 1.5 机械臂工具依赖

```bash
sg dialout -c "conda activate vera && pip install feetech-servo-sdk"
sg dialout -c "conda activate vera && python -c \"import scservo_sdk; print('sdk ok')\""
```

只增加 `feetech-servo-sdk` 和 `pyserial` 两个包，torch 不受影响。

### 1.6 lerobot 环境（遥操作/录数据用，独立环境）

```bash
sg dialout -c "conda create -n lerobot python=3.12 -y"
sg dialout -c "conda activate lerobot && git clone https://github.com/huggingface/lerobot.git ~/lerobot"
sg dialout -c "cd ~/lerobot && pip install -e '.[feetech]'"
sg dialout -c "conda activate lerobot && python -c \"import lerobot; from lerobot.robots.so_follower import SO101Follower; print('lerobot ok')\""
```

用绝对路径调用它的命令即可，不必先 `conda activate`：

```bash
sg dialout -c "~/miniconda3/envs/lerobot/bin/lerobot-teleoperate --help"
```

---

## 2. 机械臂硬件

### 2.1 ⚠️ 主臂和从臂的供电电压不同

**这是最容易浪费半天的坑。**

SO-ARM101 **Pro 版**的主臂和从臂用**不同电压规格**的舵机，因此需要**两个不同的电源适配器**：

| | 舵机规格 | 减速比 | 正确供电 | 舵机 EEPROM 电压上限 |
|---|---|---|---|---|
| **主臂 leader** | **7.4V** | 1/191、1/345、1/147 混合 | **5V**（实测 5.2V）| 12.0V（ID2 仅 8.0V）|
| **从臂 follower** | **12V** | 全部 1/345 | **12V**（实测 12.2V）| 14.0V |

两个适配器**外观一样**（5.5×2.1mm 圆头 DC），只能看外壳印的电压标签区分。

**接错的后果**：主臂接 12V 会过压 —— 主臂舵机 EEPROM 里的 `Max_Voltage_Limit` 是 12.0V，实测 12.2–12.4V 超过它，6 个舵机全部持续报电压错误（`Status` 位 `0x01`）。

**最坑的地方**是 lerobot 此时的报错完全指错方向：

```
RuntimeError: FeetechMotorsBus motor check failed on port '...':
Missing motor IDs: - 1 (expected model: 777) ...
```

看起来像"没插电"或"ID 不对"，**实际是过压**。原因是 lerobot 的 `broadcast_ping()` / `_assert_motors_exist()` 会**丢弃带错误状态的舵机**，过压的舵机在它眼里等于不存在；而原始协议不做这个过滤，能看到全部 6 个和错误位。

**所以诊断一律用 `soarm.py check`**，不要只信 lerobot 的报错。

> 标准版套装两条臂都是 7.4V，两个电源都用 5V。判断方法：看舵机的 `Max_Voltage_Limit` —— 14.0V 是 12V 舵机，8.0V 附近是 7.4V 舵机。

### 2.2 识别 USB 与串口

两条臂各有一块 USB 转总线适配器，芯片都是沁恒 WCH CH343（`1a86:55d3`），枚举成 `/dev/ttyACM*`。

```bash
sg dialout -c "lsusb"                      # 应出现两条 1a86:55d3
sg dialout -c "ls -l /dev/ttyACM*"         # 是 ttyACM 不是 ttyUSB
sg dialout -c "ls -l /dev/serial/by-id/"   # ★ 带序列号的稳定路径
```

**不要用 `/dev/ttyACM0` / `/dev/ttyACM1`**：编号由枚举顺序决定，重新插拔就可能互换，主从会反过来。用 `by-id` 路径或下面的 udev 别名。

### 2.3 串口权限

```bash
sg dialout -c "id"                       # 看有没有 dialout
sg dialout -c "sudo usermod -aG dialout $USER"
```

`usermod` **成功时不输出任何东西**，这是正常的，用 `getent group dialout` 确认。

**必须重新登录**才会在当前会话生效（组成员在登录时固定）。临时不重登：

```bash
sg dialout -c "python -m soarm check all"
```

> **Jupyter 用户的特殊要求**：如果从桌面图标启动 JupyterLab，那个进程**没有** dialout 组，notebook 里的所有串口操作都会失败。用带组的 shell 启动：
> ```bash
> sg dialout -c "~/miniconda3/envs/vera/bin/jupyter lab"
> ```
> 或者干脆重新登录一次。

### 2.4 udev 固定端口名（可选但推荐）

```bash
sg dialout -c "sudo tee /etc/udev/rules.d/99-soarm.rules > /dev/null <<'EOF'
SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"1a86\", ATTRS{idProduct}==\"55d3\", ATTRS{serial}==\"5B3D048490\", SYMLINK+=\"soarm_leader\", GROUP=\"dialout\", MODE=\"0660\"
SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"1a86\", ATTRS{idProduct}==\"55d3\", ATTRS{serial}==\"5B3D042389\", SYMLINK+=\"soarm_follower\", GROUP=\"dialout\", MODE=\"0660\"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/soarm_*"
```

序列号换成你自己的（`ls /dev/serial/by-id/` 能看到）。不想依赖 dialout 组就把 `MODE="0660"` 改成 `MODE="0666"`。

### 2.5 自动发现两条臂

```bash
sg dialout -c "cd <本仓库> && python -m soarm arms"
```

它扫描适配器、读每个端口的舵机电压画像，**按电压判定主从**（12V 舵机的是从臂、7.4V 的是主臂），并把结果缓存到 `~/.cache/soarm/arms.json`。之后所有命令都读这个缓存，不需要再指定序列号。

```
  序列号 5B3D042389  ->  /dev/ttyACM1
     6 个舵机   实测 12.2V   电压上限 14.0V   12V 舵机
  序列号 5B3D048490  ->  /dev/ttyACM0
     6 个舵机   实测 5.2V   电压上限 11.3V   7.4V 舵机

✅ 自动判定: leader = 5B3D048490 (7.4V 舵机)  follower = 5B3D042389 (12V 舵机)
```

自动判定失败时（标准版两条臂规格相同，或某条没上电），拔掉其中一条臂的 USB、看哪个节点消失即可确定映射，然后手写缓存：

```bash
mkdir -p ~/.cache/soarm
echo '{"leader":"序列号","follower":"序列号"}' > ~/.cache/soarm/arms.json
```

### 2.6 硬件自检

```bash
sg dialout -c "python -m soarm check all"
```

逐个读舵机的电压、温度、错误位：

```
    shoulder_pan     12.2V  限值[4.0,14.0]V  31°C
    ...
  ✅ 全部正常，无错误状态
```

| 输出 | 含义 | 处理 |
|---|---|---|
| `✅ 全部正常` | 通信、电压、温度都正常 | 可以标定 |
| `❌ 无响应：一个舵机都找不到` | 舵机没上电，或总线线缆/接头断开 | 查 DC 插头、适配板开关、3 芯总线、臂内菊花链第一节 |
| `★错误位 0x01 (电压)` | 供电电压不符合该臂规格 | 见 2.1 |
| `PermissionError` | 不在 dialout 组 | 见 2.3 |

无响应时进一步看：

```bash
sg dialout -c "python -m soarm probe follower"   # 两种协议 × 全部波特率
```

（其中 14400 / 128000 / 250000 会被 CH343 适配器拒绝并标"跳过"，正常。）

### 2.7 摄像头（2 × JYU2C-2083）

```bash
python -m soarm cam                      # 体检（所有相机）：设置 / 实测帧率·抖动·亮度稳定性
python -m soarm cam --lock               # 锁曝光模式·白平衡·对焦（★ 会自己失效，锁完要用 cam 确认）
python -m soarm cam --list               # 列出所有相机（序列号 + 能不能出帧）
python -m soarm cam --simul              # 多相机同时开流的体检（见下面"多相机"）
python -m soarm cam --preview            # 本地实时预览 http://127.0.0.1:8099（★ 独占摄像头，录数据前先关）
python -m soarm cam --snap frame.jpg     # 存一帧（多相机时按序列号加后缀）

python -m soarm cam --device /dev/video0     # 只操作这一只（默认对所有相机操作）
python -m soarm cam --lock --brightness 32   # 画面太暗时加数字增益（牺牲对比度，尽量靠打光）
python -m soarm cam --seconds 10             # 体检时长（默认 5 秒）
```

体检输出长这样（退出码 0 = 满足 VERA 采集要求，1 = 有问题）：

```
  当前的自动项:
    曝光模式            3（光圈优先）
    ...
  ⚠️  还没锁的项（会让画面整体漂移，VERA 会把它当运动）:
     - 曝光模式=3（光圈优先）→ 画面亮度会自己漂
     → 跑:  python -m soarm cam --lock
  实测:
  实际分辨率 1280×720   MJPG   29.8 fps   抖动 2.0ms（最慢 37ms）   读失败 0
  亮度 128.6   极差 0.2   清晰度（Laplacian 方差）52
  ✅ 满足 VERA 采集要求（帧率、亮度稳定性都过关）
```

**和机械臂端口同一个坑：别写 `/dev/video0`。** `video0/video1` 按枚举顺序排，插拔/重启会换位（`video1` 是 metadata 节点，读不出帧）。上面这些命令不加 `--device` 时自动用稳定路径：

```bash
ls -l /dev/v4l/by-id/     # 本机两只:
#   usb-JoyandAI_JYU2C-2083_JYU2C-2083-2605004-video-index0 -> ../../video0   俯拍桌面
#   usb-JoyandAI_JYU2C-2083_JYU2C-2083-2605007-video-index0 -> ../../video2   广角全景
```

> ⚠️ **`--preview` 会独占摄像头**：开着的时候 lerobot 录数据、其它 `cam` 子命令都读不到画面。
> 录数据前先 `pkill -f "soarm cam"`。

**要锁什么（`--lock` 干的事）** —— 不锁的话画面会整体漂移，而光流/视频模型分不清"物体在动"和"整幅画面变亮/变黄了"：

| 控制项 | 出厂 | 锁成 | 实测依据 |
|---|---|---|---|
| `auto_exposure` | 3（光圈优先，自动）| **1（手动 = 不再自己调）** | 自动模式静态场景 10 秒亮度漂 **42 级**；锁成手动后极差 **0.2 级** |
| `white_balance_automatic` | 1 | **0**（+ 色温，本机 4600K）| 自动白平衡会让整幅画面色温漂 |
| `focus_auto_continuous` | 1 | 0 | 本机 `focus_absolute` 只有 0..15，是桩控制（无对焦马达）：锁前锁后清晰度 112→113，无实际差别，锁上只是省心 |
| `power_line_frequency` | 1 | 1（50Hz）| 已是正确值，抗灯光频闪条带，不动 |

> ⚠️ **三个坑（都是实测的）**
>
> 1. **这些设置会丢** —— 拔插一定回出厂默认（自动曝光/自动白平衡）；实测还出现过**没拔插也自己退回**的情况（`2605007` 的曝光悄悄回到光圈优先，反复开关流不会导致、像是链路 reset）。所以采集流程是 **`--lock` 之后再跑一次 `cam` 确认**，别假定锁上了。
> 2. **`exposure_time_absolute` 在本机是桩控制**：78 与 10000 实测画面**完全不变**（亮度 179.4 vs 179.3）。所以别指望用它调亮度 —— 亮度靠**环境光**：`--lock` 锁的是"模式"（不再自动补偿），如果房间光变了，画面亮度就跟着变。
> 3. **`brightness` 是数字增益，本机有效但会牺牲对比度**：设 64 把亮度从 179 抬到 228，同时清晰度 77 → 67。尽量靠打光，不要靠它。

**帧率/格式实测（MJPG，静态场景，10 秒）**：

| 配置 | 实测 fps | 帧间隔抖动 | 说明 |
|---|---|---|---|
| **MJPG 1280×720** | **29.8** | **2.0ms**（最慢 37ms）| ★ 默认用这一档，零掉帧 |
| MJPG 640×480 | 24.8 | 1.3ms | 反而更慢（该模式的驱动默认帧率低）|
| MJPG 1920×1080 | 24.7 | 1.0ms | 带宽够但没更快 |
| YUYV 640×480 | 24.7 | 1.3ms | 未压缩，带宽吃紧 |
| YUYV 1280×720 | 9.9 | 2.0ms | ★ 带宽不够，别用 |

#### 多相机：两只同型号相机可以同时跑（实测 30 秒稳定）

本机接了 2 只同型号 JYU2C-2083：

| 序列号 | 稳定路径 | 视角 | 单独跑 | 双路同时 |
|---|---|---|---|---|
| `…2605004` | `/dev/v4l/by-id/…-2605004-video-index0` | 俯拍桌面（夹爪在画面里）| 29.8 fps | **24.7 fps** |
| `…2605007` | `/dev/v4l/by-id/…-2605007-video-index0` | 广角全景（人 / 桌子 / 机械臂）| 29.8 fps | **29.7 fps** |

**双路同时开流 30 秒实测**：两只都稳定，零读失败、抖动 1.3–2.0ms：

```
$ python -m soarm cam --simul --seconds 30
    ✅ 2605004       24.7 fps   读失败 0     分段 fps: 24.7 → 24.7 → 24.7 → 24.7
    ✅ 2605007       29.7 fps   读失败 0     分段 fps: 29.7 → 29.7 → 29.7 → 29.7 → 29.7
  ✅ 多路可以同时跑。
```

代价是**两只共享 USB 2.0 的等时带宽**：协商到较低 altsetting 的那只从 29.8 掉到 24.7 fps。VERA 控制循环只要 15Hz，24.7 完全够。

两只都插在同一条 USB 2.0 总线上（端口 1-3.1 / 1-3.2，同一个 hub，本机只有一个控制器 `00:14.0`）。**但实测能跑** —— 相机流接口虽然提供高带宽等时档（端点描述符最高 `alt=11: 3×1020 字节/微帧`），驱动会按实际需求协商，两条流挤得下。**不需要 PCIe USB 卡，也不用换相机。**

> ⚠️ **曾经出现过"一只完全读不到帧"**（另一只 0 fps、读失败数十万次），当时判断是带宽不够 —— **那个结论是错的**。原因更可能是链路/驱动的临时状态（那段时间我正用裸 ioctl 改过相机的格式和流状态）。现在连测多轮都正常。所以：
>
> **换机器或换接线后，跑一次 `python -m soarm cam --simul` 亲自确认**，别照抄任何结论（包括这段）。

> ⚠️ **锁定的设置会自己丢失。** 实测 `2605007` 在没被拔插的情况下退回过出厂值（曝光回到光圈优先）。反复开关流不会让它复位（各测 4 次都保持），所以更像是链路 reset 一类的事件。因此采集流程应该是：`--lock` 之后**再用 `cam` 确认一遍**（它会列出还没锁的项），别假定锁上了。

> `--preview` 会把所有相机放在同一页（每路一张卡 + 实时帧率 + 读失败计数），双路能不能跑一眼就看出来。

另外：
- **单帧成本**：1280×720 MJPG 解码 + resize 到 128×192 + BGR→RGB ≈ **40ms**（全分辨率 resize 是主要开销）。VERA 控制循环 15Hz、预算 67ms，够用但没有很多余量。
- **帧格式**：cv2 给的是 BGR，VERA 的契约是 **uint8 HxWx3 RGB**，`frame[..., ::-1]` 转一下（参考 `vera-main/vera/controller/nora_camera_reader.py`）。
- 相机视野：桌面方块 + **夹爪本体都在画面里**（夹爪在画面下缘），正是 IDM 需要的"工具-物体接触区"。

---

## 3. 标定（两趟法）

### 3.1 ⚠️ 先知道 STS3215 的三个行为

这三个都是**实测**出来的，直接决定操作顺序，搞错会导致机械臂猛冲或变硬掰不动。

| 行为 | 后果 |
|---|---|
| **写 `Goal_Position` 会让固件自动开启力矩** | 即使刚发过关闭指令。实测 6/6 全中；只写 Goal、不碰 `Torque_Enable`。（`Homing_Offset` / `Max_Position_Limit` / `Operating_Mode` / `Lock` 都**不会**，逐个测过）|
| **`Goal_Position` 在 SRAM，上电默认 0**，且力矩开启时舵机**朝 Goal 驱动**（不是保持当前位置）| 上电后开力矩会朝刻度 0 猛冲 |
| **手动移动机械臂不会更新 `Goal_Position`** | 掰过臂之后 Goal 就是"过时的旧姿态"，差值可能极大（实测有 1938 刻度 ≈ 170°）|

**推论**：

- 做任何 lerobot 操作（遥操作/录数据）之前，先跑 `soarm.py prep`；
- **每次给机械臂断电重启后都要重跑**（Goal 在 SRAM，掉电清零）；
- 标定工具会自己保证臂是松的（每 2 秒重发一次关力矩）。

### 3.2 为什么是两趟

lerobot 原版流程是「把关节摆到行程中间按回车」，而"中间"被实现成**舵机自己的半圈刻度 2047**。两者不是一回事：2047 是舵机转整圈的一半，而关节的机械行程只是整圈的一小段。所以对行程偏在一侧的关节，读数 2047 可能离**机械中点**很远。

实测后果：`2047位置` 一列会出现从臂夹爪 32%、主臂夹爪 0% 这种零点完全不对应的情况。

还有更隐蔽的一条：按固定角度自动停止采集行程会**截断行程更宽的关节**。

两趟法解决这两个问题：

| 趟 | 命令 | 做什么 |
|---|---|---|
| 1 | `measure` | 手推测出**每个关节的真实行程** |
| 2 | `mid` | 摆位（体检 + 定 `wrist_roll` 的零位，见下）|
| 3 | `finalize` | 写入。输出 `2047位置` 供核对 |

数学上，零点**直接取自实测的行程中点**：

```
mid_raw = 行程中点在原始刻度的读数
o       = mid_raw − 2047        写进 Homing_Offset
量程    = [2047 − W/2, 2047 + W/2]       W = 实测行程宽度
```

写完偏移后，**行程中点**读数恰好是 2047，量程恒定居中 —— 所以 `2047位置` 永远精确是 50%。

> **这意味着你不必把关节摆得准。** 偏移量只由实测行程决定，位置只要落在测量到的行程内就行（`finalize` 会检查这一点；跑出去了说明行程数据过时了，重跑 `measure`）。
>
> 早期版本把偏移定成 `pos − 2047`（强制"当前位置 = 中点"），结果位置一偏量程就被推出 `[0,4095]`，报"偏移后量程超出"—— 现在没有这个问题了。

**`wrist_roll` 是例外**：整圈旋转没有行程可测，它的零位只能取自**你摆的那个位置**。所以第二趟仍然要跑，主要是为了它 —— 而且主从必须摆成同一个物理方向。

`wrist_roll` 是整圈旋转关节，**没有行程端点**，所以不参与第一趟测量，量程固定 0–4095。但它的**零位仍然要定**（第二趟和其余关节一起摆），见 3.4 节。

### 3.3 第一趟：测真实行程

```bash
sg dialout -c "python -m soarm measure follower"
```

**用手把每个关节从一端推到另一端**，走满真实行程。推满之后**按 `x` 结束**（Ctrl-C 也可以，同样保留已测数据）。**没有时间限制**，推多久由你决定。

- 力矩是关的，可以放心用手推（工具每 2 秒会重发一次关力矩指令）；
- `wrist_roll` 不用推；它在表里不出现，因为整圈旋转没有行程端点（量程固定 0–4095）。**这是唯一一个不出现在第一趟的关节**；
- 所有关节推完后按 `x`。

```
  关节                 下限(raw)    上限(raw)       已测宽度       距下限
  ------------------------------------------------------------
  shoulder_pan           1345        3100       154.2°       75.3°
  ...
```

```
  关节                 行程宽度       距下限      当前(raw)  备注
  ------------------------------------------------------------
  shoulder_lift       187.1°        75.3°         1983  跨过 0 点
  ...
```

**角度都是相对行程算的，不是舵机整圈角度**：

| 列 | 含义 |
|---|---|
| `行程宽度` | 该关节**真正走过的角度**（0–250° 量级，不可能接近 360°）|
| `距下限` | 当前位置离行程下限多远，范围 0 到 `行程宽度` |
| `当前(raw)` | 编码器原始刻度 |
| `备注` | 行程跨过编码器 0 点时会标出来 |

### ⚠️ 两个必须绕开的坑

**坑一：不能拿原始刻度直接当角度。** `刻度 / 4096 × 360` 算的是**舵机转整圈**的角度。关节的机械行程只是整圈的一小段，所以一个只能转 154° 的关节，它的绝对刻度完全可能落在整圈的 170°~320° 那一段 —— 看到"接近 360°"就是这个原因。工具里所有面向用户的角度都相对行程（下限或中点）算。

**坑二：行程可能跨过编码器 0 点。** 编码器读数是 0..4095 的单圈值。如果某个关节的行程从 4095 继续转到 0 再往上（取决于舵机盘的安装齿位，很常见），那么直接记 `min`/`max` 会得到 `min≈0, max≈4095`，**宽度算成接近 360°**。实测在小臂上犯过这个错：`shoulder_lift` 量出 353.8°，而真值是 **187°**（和早期 lerobot 标定记录一致）。

正确做法是按**最短弧累积位移**（把行程"展开"到实数轴上），宽度取展开后的极差 —— 与起点在哪里无关：

```
采样 3950 → 4095 → 0 → 1983
  错误：max-min = 4095           → 359.9°
  正确：展开后极差 = 2129        → 187.1°
```

工具内部就是这么算的。附带的好处是：写完 `Homing_Offset` 之后量程会自动落到区间中部（因为行程中点被锚定到 2047），所以**跨 0 点这件事不会传递到舵机的 `Min/Max_Position_Limit`** 上。

> 行程文件带格式版本号。旧格式（按 raw min/max 记录）会被明确拒绝并要求重新测量 —— 它算出的宽度是错的，不能拿去标定。

结果**直接记到仓库里的 `measure/` 目录**，第二趟（`mid`）从这里读：

```
measure/
├── follower.json     # 从臂的行程测量结果
└── leader.json       # 主臂的
```

这样结果直接可见、方便取用和检查，不用去翻 `~/.cache`。文件里是**原始刻度**（每台机器的装配齿位不同，数值必然不同），所以 `measure/` 已加进 `.gitignore` —— 它是本地产物，不该提交。想让别人复用你的测量结果就把那行删掉。

路径可用环境变量覆盖：`SOARM_MEASURE_DIR=/path/to/dir`。

> 机器配置（两条臂的序列号）仍在 `~/.cache/soarm/arms.json`，那是自动发现的结果、不是测量产物，所以留在用户目录。

两个自动检查：

- 行程 **不到 20°** → 告警（没推到位或机械卡住）；
- 行程 **超过 300°** → 报错（单个关节转不了这么多，读数被错误包污染了），并明确提示**不要用这份数据往下做标定**。

工具对每次采样做 **3 点中位数滤波**，抑制半双工总线的偶发错误包（lerobot 自己的代码注释也承认这种丢包存在）—— 否则一次坏读数就会污染整个行程范围，而且无声无息（实测见过 `min=5, max=4094` 这种整圈跳动）。真出现这种告警时，先确认**没有别的程序同时在读同一条总线**，然后重测。

### 3.4 第二趟：摆到各自中点

```bash
sg dialout -c "python -m soarm mid follower"
```

和 `measure` 一样：**没有计时也没有时限**，摆好后**按 `x` 结束**（Ctrl-C 也可以）。5 个行程关节都进入容差（默认 ±8°）时会自动结束。

> 因为零点取自实测行程中点（见 3.2），这一趟对 5 个行程关节**不是必须的** —— 它主要做两件事：确认臂在测量范围内、给 `wrist_roll` 定零位。摆不准不影响标定结果。

```
  关节                   相对中点     行程位置  位置条（| 中点  o 当前）  状态
  ------------------------------------------------------------------------------
  shoulder_pan       -70.0°      -1%  [o-----------|------------]  偏离中点
  shoulder_lift     +176.1°     144%  [------------|-----------o]  ★已超出测量行程
  ...
  wrist_roll              —        —                             无行程可对；raw 486，将写入偏移 -1561
  gripper            +18.5°      64%  [------------|---o--------]  偏离中点
```

| 列 | 含义 |
|---|---|
| `相对中点` | 当前位置相对该关节行程中点的角度偏移（带符号）|
| `行程位置` | 落在行程的百分之几，0% = 下限、100% = 上限 |
| `位置条` | 把行程画成一条线，`\|` 是中点、`o` 是当前位置 —— 一眼看出往哪边调、还差多少 |
| 状态 | `OK` 在中点附近（±8°）／`偏离中点` 继续调／`★已超出测量行程` 读数跑出第一趟范围，重跑第一趟 |

> 位置条的坐标轴是**该关节的实际行程**，不是编码器整圈 —— 一个只能转 187° 的关节会铺满整条线，而不是挤在整圈的一小段里。

**`wrist_roll` 也要摆，而且主从要用同一个物理方向。**

它没有行程中点可对，所以那一行只显示当前 raw 和"将写入的偏移"，不参与 `OK` 判定。但它的零位**会**写进舵机（`Homing_Offset`），而且是在你按 x/结束那一刻的读数上算的。

所以：**把两条臂的 `wrist_roll` 摆成同一个物理方向**（比如夹爪开口都竖直朝上），再去分别跑 `mid` + `finalize`。否则主从的 `wrist_roll` 零点对不上，遥操作时手腕旋转会一直偏 —— 这是唯一一个"没有行程也要对零位"的关节。

如果那一行提示偏移超出 ±2047，把手腕转一下再记录即可（整圈旋转可以随便转）。

### 3.5 第三趟：写入标定

```bash
# 先空跑看结果，不写任何东西
sg dialout -c "python -m soarm finalize follower --id my_follower --dry-run"


# 确认无误后写入
sg dialout -c "python -m soarm finalize follower --id my_follower"
```

输出：

```
  关节              位置      偏移             最终量程       宽度    2047位置
  ----------------------------------------------------------------------
  shoulder_pan     2222    +175 [ 1170, 2925]   154.2°       50%
  ...

  2047 在各关节量程中的位置: 50% 50% 50% 50% 50% 50%   （应接近 50%）
```

**`2047位置` 全部接近 50% 就是标定正确的标志。**

`--id` 决定文件名（lerobot 靠它找标定）：

- 从臂：`~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`
- 主臂：`~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/<id>.json`

**务必给 id**（本指南用 `my_follower` / `my_leader`）。不指定会存成 `None.json`，两条同型号的臂会互相覆盖。

写入前会自动校验：偏移是否超出 ±2047（超了舵机会报 `ValueError`）、偏移后量程是否越界、行程是否小于 20°。**不通过就拒绝写入并说明哪个关节有问题**，不会留下坏标定。

### 3.6 标主臂

把 `follower` 换成 `leader`，`--id` 换成 `my_leader`，重复 3.3–3.5：

```bash
sg dialout -c "python -m soarm measure  leader"
sg dialout -c "python -m soarm mid      leader"
sg dialout -c "python -m soarm finalize leader --id my_leader"
```

> **两条臂不需要手工摆成同一个姿态。** 因为归一化零点是各自**实测行程的中点**，只要各自把行程推满，中点自然对应。这是两趟法比原版流程省心的地方。

### 3.7 检查主从对齐

```bash
sg dialout -c "python -m soarm compare"
```

```
  关节                  从臂行程     主臂行程     从2047位置     主2047位置  判定
  ------------------------------------------------------------------------
  shoulder_pan        238°     202°         50%         50%  ✓ 对齐
  gripper             128°     107°         32%          0%  ✗ 差太多，建议重标
```

两趟标定做对了应该都是 **50%**。某个关节差距大 = 那一趟没推满，重跑该臂的 `measure`。

### 3.8 把从臂对齐到主臂（以主臂为基准）

**读主臂的位姿，把从臂驱动到同一个姿态。** 标定（3.7）保证两条臂的零点对得上，这一步保证它们**此刻的姿态**也对得上 —— 遥操作启动时从臂会朝主臂姿态移动，`--robot.max_relative_target` 只是限速、不是限位（4.4），先对齐过启动就只是小范围微调。

```bash
sg dialout -c "python -m soarm align"            # 读主臂 → 从臂对齐过去
sg dialout -c "python -m soarm align --dry-run"  # 只看偏差，不动任何东西
```

主臂是基准、只读：它的力矩全程关着，**用手摆**；从臂由工具驱动跟随。运行后从臂立即开始朝主臂姿态移动。

```
主臂 → 从臂 对齐    从臂跟随主臂    容差 ±5.0°

  关节            主臂位置 从臂位置   需移动   位置条                       状态
  ------------------------------------------------------------------------------
  shoulder_pan         52%      48%    +9.5°   [------------o|-----------]  偏离
  shoulder_lift        50%      50%    -0.4°   [------------o------------]  OK
  elbow_flex           46%      61%   -29.0°   [-----------|---o---------]  偏离
  ...
  gripper              18%      22%    -5.1°   [----|o-------------------]  偏离

  对齐 2/6 个关节（偏差在 ±5.0° 内算对齐）
```

| 列 | 含义 |
|---|---|
| `主臂位置` / `从臂位置` | 归一化到**各自行程**的百分比（0% = 行程下限，100% = 上限）—— 和 lerobot 用的是同一组标定数 |
| `需移动` | 从臂还要转多少度，`+` 是往它自己量程的上限方向 |
| `位置条` | 坐标轴是**从臂的行程**：`\|` 是主臂位置映射过来的目标，`o` 是从臂现在的位置 |
| 状态 | `OK` 在容差内／`偏离`／`★主臂超量程`（主臂已超出从臂能到的地方，只能到端点）|

对上之后按 `x` 结束：从臂**停在当前姿态并保持力矩**，可以直接起遥操作（4.4）。按 `x` 是停住，不会继续追目标；结束时那个力矩**不要用 `prep` 关掉** —— 关掉从臂会因重力下垂，对齐就白做了。

每个控制周期最多动 `--max-step` 度（默认 2°，约 40°/s），不会为了追上主臂猛冲；`--limp` 让从臂结束时松手。

**为什么命令值是"积分"推上去的（实测行为）**：带负载的关节有明显静差 —— 从臂 elbow 托着前臂时，舵机停住的位置比命令值差 **30 刻度（2.6°）**，wrist_flex 差 15 刻度：重力压着关节，舵机自己走不到命令值。所以如果每周期只把命令值往前推一小步（1~2°），这一步整个落在静差里 → 舵机认为"已经到了" → **关节永远不动**（实测：每周期 +1° 推 elbow 纹丝不动、负载 0；一次写 13° 立刻就动、负载 132）。因此工具内部是**积分式**：命令值持续朝目标累积，直到**实测位置**追上目标，静差被自动补掉（到位后残差 ≤ 2 刻度 ≈ 0.18°）；同时命令值最多领先实测位置 90 刻度，被挡住时不会无限积累。遥操作（lerobot）没暴露这个问题，只是因为它的 `max_relative_target` 默认是 10 度。

**方向自检**（第一次跑之前做一遍）：把两条臂用手摆成同一个物理姿态，位置要**偏离中点**（推到行程的一端最好）—— 行程中点是左右对称的，镜像在那里也是 0，看不出问题。都接近 0 说明主从方向一致；某个关节显示接近**整段行程宽度**的值，就是该关节主从装配方向相反，先查舵机盘装配（lerobot 里主从方向不一致是靠标定里的 `drive_mode` 表达，本工具一律按 0 处理，遇到非 0 会直接拒绝而不是猜）。

`wrist_roll` 是整圈旋转关节（3.4），按**最短弧**算，不会为了差 10° 转一整圈。

不指定 id 时用该角色目录下**唯一**的那个标定文件；有多个就用 `--follower-id my_follower --leader-id my_leader` 指明遥操作真正在用的那个。

---

## 4. 验证与遥操作

### 4.1 验证标定（只读）

```bash
sg dialout -c "python -m soarm verify follower --id my_follower"
```

阶段 1 读各关节位置，和标定量程对比 —— 检查标定数据是否自洽、当前姿态是否在行程内。

### 4.2 验证标定（会运动）

```bash
# 确认周围无人无障碍后再跑
sg dialout -c "python -m soarm verify follower --id my_follower --move"
```

阶段 2 让每个关节**朝量程中心**移动 8° 再返回，验证力矩、方向、标定映射。它自己控制时序（先 Goal := 当前位置再开力矩），不会猛冲；结束时关力矩。

### 4.3 开力矩前的安全预处理

```bash
sg dialout -c "python -m soarm state follower"    # 先看力矩/Goal/位置
sg dialout -c "python -m soarm prep  follower"    # 把 Goal 对齐当前位置并关力矩
```

`state` 里看到 `⚠️ 开力矩后会朝 Goal 猛冲` 就一定要先 `prep`。**每次断电重启后都要重跑 `prep`。**

> `prep` 的顺序是「写 Goal → 再关力矩」。反过来无效，因为写 Goal 会让固件自己把力矩打开。

### 4.4 遥操作（切到 lerobot 环境）

运行前：

1. 先在 `vera` 环境跑完 `soarm.py prep`；
2. 把**从臂和主臂摆成相近的姿态**（遥操作启动时从臂会朝主臂的姿态移动）—— 用 3.8 的 `soarm.py align`：主臂用手摆，从臂自动跟过去；它结束时从臂保持力矩，所以第 1 步的 `prep` 就不必再跑了；
3. 两条臂都接**正确电压**的电源；
4. 工作空间内无手无障碍。

```bash
sg dialout -c "~/miniconda3/envs/lerobot/bin/lerobot-teleoperate \
  --robot.type=so101_follower --robot.id=my_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D042389-if00 \
  --robot.max_relative_target=10 \
  --teleop.type=so101_leader --teleop.id=my_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D048490-if00"
```

默认用**完整的 `by-id` 路径**：不依赖 udev 规则，插拔顺序变了也对得上，而且和 `soarm.py` 内部解析出来的是同一个节点。序列号随机器不同，两条路径可以直接打印，别照抄上面那串：

```bash
sg dialout -c "python -c \"from soarm import port_of; print(port_of('follower')); print(port_of('leader'))\""
```

> 装了 2.4 的 udev 规则的话，可改写成短别名 `/dev/soarm_follower`、`/dev/soarm_leader`，等价但更好敲。

> ⚠️ **这条命令必须在有 `dialout` 组的 shell 里跑。** 缺组时端口打不开，但报的不是权限错误：pyserial 抛的 `PermissionError` 是 `OSError` 的子类，被 lerobot 的 `except (FileNotFoundError, OSError, serial.SerialException)` 一并转成 `ConnectionError: Could not connect on port '...'`，真实原因（`Errno 13 Permission denied`）就此消失，只剩一句让你去跑 `lerobot-find-port` 的误导提示 —— 端口路径通常是对的，别照着它去换端口。自查方法：`ls -l /dev/serial/by-id/` 看得到节点、`soarm.py check` 也正常，那就是权限。注销重新登录即可，或临时 `sg dialout -c '整条 lerobot 命令'`。

`--robot.max_relative_target=10` 是**限速**：每个控制周期把目标相对当前位置的差值钳到 ±10 度。它避免启动瞬间猛冲，但**不是限位**，臂最终还是会走到主臂姿态 —— 所以第 2 条不能省。

### 4.5 和 VERA 的衔接

**当前进度**：环境 ✅ / 机械臂与标定 ✅ / 遥操作 ✅ / 双摄像头（可同时开流）✅ / **SO-ARM 接入 VERA ⬜ 未开始**

VERA 已有的 embodiment 只有 `pusht` / `mimicgen` / `allegro` / `droid`（见 `vera-main/vera/server/`），**没有 SO-ARM**，需要自己加。参考 `start_server_allegro.py`。

VERA 真实机械臂的架构是**两台机器**：

```
GPU 主机（vera 环境）              机械臂主机（lerobot 环境）
WAN + VGGT-J IDM 推理   <--ws-->   读状态 / 下发动作 / 收画面
                        msgpack
```

**IDM 输出的是关节增量 `du`**（不是绝对角度），机械臂主机把它加到当前 `q` 上得到绝对目标再下发。这正好和"两个环境不能共存"对应：GPU 侧跑 VERA，机械臂侧跑 lerobot，只通过 websocket+msgpack 通信，不共享进程。

接入 SO-ARM 需要：① 按 VERA 数据格式采轨迹（`docs/DATA_GENERATION.md`，用 lerobot）；② 为 SO-ARM 训一个基于雅可比的 IDM（`TRAINING.md` 的 Stage 1）；③ 仿 `start_server_allegro.py` 加服务端 embodiment；④ 写机械臂主机的 ws runner。

> **摄像头标定**在训 IDM 时才需要，形式和直觉不同：内参 3×3 要**按图像尺寸归一化**（`vera/utils/convention.py` 的 `denormalize_intrinsics` 会把 fx、cx 乘回 width），外参 4×4 是 **cam2world**（相机到世界，`vera/utils/geometry.py` 的参数名就是 `cam2world`，内部再 `transform_world2cam`）。方向搞反会导致光流全错。VERA 的仿真 embodiment 这些参数是**按视角配置合成生成**的（`vera/utils/camera_utils.py`），真机需要自己提供。

**先跑通仿真确认 VERA 本身没问题**（不需要机械臂）：

```bash
sg dialout -c "conda activate vera && python -m vera.server.start_vera_server --embodiment pusht --port 8820 --vis-port 8821"
# 然后打开 vera-main/examples/pusht_dfot_stack.ipynb → Run All
```

带 `--vis-port` 时浏览器打开 `http://localhost:8821/` 能实时看两阶段流水线。

---

## 5. 换台机器复现

### 5.1 必须改的

| 项 | 本机值 | 要做什么 |
|---|---|---|
| 两条臂的 USB 序列号 | `5B3D048490`（主）/ `5B3D042389`（从）| **代码不用改** —— `soarm.py arms` 自动发现并判定。但 4.4 的 lerobot 命令行里是写死的 `by-id` 路径，要换成自己的（跑一次 4.4 里的 `port_of` 那行打印，或照 2.4 装 udev 别名） |
| 机械臂型号 | SO-ARM101 Pro | 标准版两条臂都是 7.4V 舵机，电压画像相同 → 自动判定失效，按 2.5 手动确定映射 |
| 供电电压 | 主臂 5V / 从臂 12V | 取决于套装版本。判据：`check` 输出 `限值[4.0,14.0]V` 是 12V 舵机，`[4.0,8.0]V` 是 7.4V |
| 标定 id | `my_follower` / `my_leader` | 可自选，但**必须与 lerobot 命令的 `--robot.id` / `--teleop.id` 一致** |
| 摄像头节点 | `…JYU2C-2083-2605004-video-index0` / `…2605007-video-index0` | **不用改代码** —— `python -m soarm cam` 自动按 `/dev/v4l/by-id` 找全部相机。相机数量可以不同（1 只也能用）；双路是否跑得动用 `python -m soarm cam --simul` 确认（见 2.7）|
| conda / 仓库路径 | `~/miniconda3`、本仓库 | 换成自己的 |

### 5.2 不用改的

寄存器地址、`Homing_Offset` 编码公式、标定文件格式与路径规则、两趟标定流程、三个舵机力矩行为、两个环境的划分、`soarm.py` 本身。

### 5.3 完整顺序

```bash
# 1. 硬件：两条臂 + 各自的电源（注意电压可能不同）+ 摄像头
# 2. 环境：按第 1 章（vera 环境含 pip install lpips；另建 lerobot 环境）
# 3. 权限
sg dialout -c "sudo usermod -aG dialout $USER"      # 然后重新登录
# 4. 识别与自检
sg dialout -c "python -m soarm arms"
sg dialout -c "python -m soarm check all"     # 两条臂都应 ✅
# 5. 标定（每条臂三趟）
sg dialout -c "for ROLE in follower leader; do python -m soarm measure  \$ROLE; python -m soarm mid      \$ROLE; python -m soarm finalize \$ROLE --id my_\$ROLE; done"
sg dialout -c "python -m soarm compare"       # 都应是 50%
# 6. 摄像头（VERA 采数据前必做；设置会自己失效，所以锁完要再确认一次）
python -m soarm cam --list                # 有几只、分别能不能出帧
python -m soarm cam --simul               # 多路能不能同时开流
python -m soarm cam --lock                # 锁曝光/白平衡/对焦
python -m soarm cam                       # 确认：应显示"满足 VERA 采集要求"且无"还没锁的项"
# 7. 验证与遥操作
sg dialout -c "python -m soarm verify follower --id my_follower"
# 遥操作前的姿态对齐，二选一（3.8）：
#   手动：sg dialout -c "python -m soarm prep follower"  →  关力矩，然后用两只手把两条臂摆到相近姿态
#   自动：sg dialout -c "python -m soarm align"          →  读主臂位姿、从臂自动跟过去；不必再 prep
# 然后按 4.4 跑 lerobot-teleoperate
```

---

## 6. 排错

### 串口与设备

| 现象 | 原因 | 处理 |
|---|---|---|
| `ls /dev/ttyUSB*` 报 no such file | 正常，适配器枚举成 ttyACM | 用 `ls /dev/ttyACM*` |
| 插拔后 ttyACM 编号变了 | 枚举顺序变化 | 用 `by-id` 路径或 2.4 的 udev 别名 |
| `arms` 报"打不开串口：权限不足" | 当前进程不在 dialout 组 | 重新登录，或 `sg dialout -c '...'` |
| `arms` 报"无法按电压自动判定主从" | 两条臂规格相同，或某条没上电 | 拔插确定映射，按 2.5 手写缓存 |
| Jupyter 里所有串口操作都失败 | JupyterLab 从桌面启动，没有 dialout 组 | `sg dialout -c '...jupyter lab'`，或重新登录 |
| lerobot 报 `Could not connect on port '...'`，可端口是对的（`ls -l /dev/serial/by-id/` 看得到、`soarm.py check` 也正常）| **仍是 dialout**，只是真因被吞了：pyserial 的 `PermissionError` 属于 `OSError`，被 `motors_bus.py` 的 `except (FileNotFoundError, OSError, serial.SerialException)` 一并转成 `ConnectionError`，`Errno 13 Permission denied` 就此消失，只剩一句叫你去跑 `lerobot-find-port` 的误导提示 | 重新登录（推荐）、`sg dialout -c '整条命令'`，或把 2.4 的 udev 规则里 `MODE="0660"` 改成 `0666` 以彻底免组 |

### 通信与硬件

| 现象 | 原因 | 处理 |
|---|---|---|
| lerobot 报 `Missing motor IDs` | **可能是电压错误**（见 2.1），不一定是没插好 | 先跑 `soarm.py check` 看错误位 |
| `check` 报无响应但电源灯亮 | 灯可能接在 USB 侧，不代表舵机总线有电 | 查 DC 插头是否插紧、适配板有无电源开关 |
| 一个舵机（如夹爪）单独报错 | 该舵机线缆或接头问题 | 检查它前后两段 3 芯线 |
| 全部波特率都无响应 | 舵机总线没电 | 用 `probe` 确认，然后查供电 |
| `Incorrect status packet!` | Feetech 半双工总线偶发丢包（lerobot 自己的代码注释也承认）| 重跑那一步，不是硬件故障 |
| **通电后用手掰不动机械臂** | 力矩被留在开启状态。两个来源：(a) 任何程序写了 `Goal_Position`（会自动开力矩）；(b) lerobot 被中断/崩溃没走到 `disconnect()` | `python -m soarm prep <role>` 会先对齐 Goal 再关力矩 |
| 开力矩时机械臂猛地甩动 | `Goal_Position` 是过时的旧姿态 | 先 `state` 看差值，再 `prep` |

### 标定

| 现象 | 原因 | 处理 |
|---|---|---|
| `2047位置` 某关节明显偏离 50% | 第二趟没摆到该关节的中点 | 重跑 `mid`，确认显示 `OK` 再 `finalize` |
| `★已超出测量行程` | 读数跑出第一趟范围 | 重跑 `measure` |
| `❌ 校验未通过 ... 偏移超出 ±2047` | 该关节的实际中点太靠近编码器量程顶端 | 该关节零位装配偏得厉害；换安装角度重装舵机盘 |
| `measure` 报某关节行程不到 20° | 没推到位，或机械卡住 | 重推，两端都推到限位；手推不动就先跑 `prep` |
| `measure` 报某关节行程不到 20°，或 `mid`/`finalize` 说数据不可用 | measure 没推到位，或跑到一半退出了 | 重新 `measure`，每个关节都从一端推到另一端 |
| `measure` 报某关节行程超过 300° | 读数被错误包污染，或**有另一个进程同时在读同一条总线** | 确认没有别的程序占用串口（`pgrep -af soarm`），然后重测 |
| `measure` 的 `距下限` 一直是 0 | 该关节没动过 | 正常 —— 还没推它；推到两端后这个值会跟着变 |
| `measure` 报某关节行程不到 20° | 没推到位，或机械卡住 | 重推，两端都推到限位 |
| `compare` 某关节 `✗ 差太多` | 两条臂行程宽度差太多（某趟没推满）| 两条臂都重跑 `measure` |
| 标定后 lerobot 报 `no calibration registered` | 该臂 JSON 不存在或 id 不符 | 检查 `~/.cache/huggingface/lerobot/calibration/` 下的文件名与 `--robot.id` 是否一致 |
| lerobot 说 `Missing motor IDs` 但 `check` 正常 | 标定文件 id 与命令不符 | 同上 |

### 主从对齐（3.8）

| 现象 | 原因 | 处理 |
|---|---|---|
| `align` 报"需要交互终端" | 输出被重定向或管道了 | 在终端里跑（要能按 `x`）；`--dry-run` 不受限制 |
| `align` 报某条臂"还没有标定文件" | 该臂没标定，或 id 给错 | 先按 3.5/3.6 标定，或 `--follower-id` / `--leader-id` 指明在用的 id |
| 某关节"需移动"一直是接近整段行程宽度的值 | 该关节主从装配方向相反（镜像） | 先做 3.8 的方向自检，再查舵机盘装配；别硬驱 |
| 显示 `★主臂超量程` | 主臂摆到了自己的行程之外，映射过来的目标落在从臂量程外 | 把主臂摆回行程内；从臂只能到端点 |
| 按 `x` 结束但从臂没到位 | `x` 是**停在当前姿态**，不会继续追目标 | 再跑一次，让它多走一会儿 |
| 对齐完起遥操作，从臂又往下掉 | 中间跑过 `prep`，把关掉的力矩松开、并让臂因重力下垂 | `align` 之后直接起遥操作（3.8）|
| `align` 一跑从臂就动，来不及反应 | 这是设计：读主臂、跟过去，不等待确认 | 先 `--dry-run` 看一遍偏差；真跑起来按 `x` 就停 |
| 某关节"需移动"一直不动、位置也不变 | 顶到限位或被挡住 → 工具会标 `★没动（顶住/被挡住）` | 看该关节是不是压在桌面/线缆上；把姿态挪开一点再跑 |
| 从臂某关节顶着不动，那一行标 `★主臂超量程` | 主臂那个关节摆到了自己行程之外，目标被夹在从臂端点 | 把主臂那个关节往回摆进行程内 |
| 从臂保持力矩后想用手掰 | 它正通电保持姿态 | `python -m soarm prep follower` |

### 摄像头（2.7）

| 现象 | 原因 | 处理 |
|---|---|---|
| `cam` / `--snap` 报"打不开"，但设备明明在 | 别的程序占着摄像头 —— **最常见是本工具自己的 `--preview` 还开着** | `pkill -f "soarm cam"` 再试；录数据前也记得先关预览 |
| 预览里某一路是黑的 / 帧率 0 | 该路没开流成功 | 看那一路的"读失败"计数；跑 `python -m soarm cam --simul` 单独确认 |
| `--lock` 跑过了，`cam` 仍报"还没锁的项" | 设置自己丢了（实测：没拔插也发生过）| 重跑 `--lock`，再跑 `cam` 确认，两条都要过 |
| 画面整体忽明忽暗 | 自动曝光没锁（出厂是光圈优先，会自己凑亮度）| `python -m soarm cam --lock`，并确认 `曝光模式 = 1（手动）` |
| 整幅画面随手臂动作变亮/变黄 | 同上，或自动白平衡没关 | 同上；确认 `自动白平衡 = 0` |
| 帧率只有 ~24 而非 ~30 | 双路同时开流时共享 USB 2.0 等时带宽（24.7 也够 VERA 的 15Hz）| 正常，不用管 |
| `Cannot open camera by name` 警告 | 同"打不开"：设备被占 | 同第一行 |
| 换机器/换接线后双路跑不起来 | 与 USB 拓扑有关 | 先跑 `--simul`；确实不行再考虑 PCIe USB 卡或换 USB 3.0 相机（2.7 有说明）|

### VERA 环境

| 现象 | 原因 | 处理 |
|---|---|---|
| `ImportError: cannot import name 'LearnedPerceptualImagePatchSimilarity'` | VERA 漏装 `lpips` | `pip install lpips` |
| `torch.cuda.is_available()` 为 False | 装成 CPU 版 wheel | 确认版本带 `+cu124` 后缀 |
| `ModuleNotFoundError: lerobot` | 在 `vera` 环境里调用了 lerobot | 切到 `lerobot` 环境 |
| `import vera` 报 `pkg_resources` 相关错 | `setuptools>=81` 删了 `pkg_resources` | `pip install "setuptools<81"` |
| VGGT 装不上 | git 依赖被挡 | `pip install "git+https://github.com/facebookresearch/vggt.git"` |
| flash-attn 编译失败 | 可选依赖 | 忽略，WAN 会回退 SDPA |
