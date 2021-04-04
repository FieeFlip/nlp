文中所有教学代码和日志见：
*https://link.zhihu.com/?target=https%3A//github.com/BIGBALLON/distribuuuu/tree/master/tutorial*

文中提到的框架见：
*https://link.zhihu.com/?target=https%3A//github.com/BIGBALLON/distribuuuu*

入门编程代码

https://mp.weixin.qq.com/s/G-uLl3HXzFJOW03nA7etig

# 分布式并行训练概述

最常被提起，容易实现且使用最广泛的，莫过于**数据并行(Data Parallelism)** 技术，其核心思想是将大batch划分为若干小barch分发到不同device并行计算，解决单GPU显存不足的限制。以此同时，当单GPU无法放下整个模型时，我们还需考虑**模型并行 (Model / Pipeline Parallelism)**。如考虑将模型进行纵向切割，不同的Layers放在不同的device上。或是横向切割，通过矩阵运算进行加速。当然，还有一些非并行的技术或者技巧，用于解决训练效率或者训练显存不足等问题，这里草率地将当前深度学习的大规模分布式训练技术分为如下三类：

- **Data Parallelism** (数据并行)

- - Naive：每个worker存储一份model和optimizer，每轮迭代时，将样本分为若干份分发给各个worker，实现并行计算
  - **ZeRO**: Zero Redundancy Optimizer，微软提出的数据并行内存优化技术，核心思想是保持Naive数据并行通信效率的同时，尽可能降低内存占用

- **Model/Pipeline Parallelism** (模型/管道并行)

- - **Naive**: 纵向切割模型，将不同的layers放到不同的device上，按顺序进行正/反向传播
  - **GPipe**：小批量流水线方式的纵向切割模型并行
  - **Megatron-LM**：Tensor-slicing方式的模型并行加速

- **Non-parallelism approach** (非并行技术)

- - Gradient Accumulation: 通过梯度累加的方式解决显存不足的问题，常用于模型较大，单卡只能塞下很小的batch的并行训练中
  - CPU Offload: 同时利用 CPU 和 GPU 内存来训练大型模型，即存在GPU-CPU-GPU的 transfers 操作
  - etc.：还有很多不一一罗列(如Checkpointing,Memory Efficient Optimizer等)

**DeepSpeed**，微软在2020年开源的一个基于PyTorch分布式训练 library，让训练百亿参数的巨大模型成为可能，其提供的 3D-parallelism (DP+PP+MP)的并行技术组合。

### 单机多卡DP

```python
# you would like to speed up training with the minimum code change.
net = nn.DataParallel(net)
```

### 多机多卡DDP

- 进程组的概念

- - **group**：进程组，大部分情况下DDP的各个进程是在同一个进程组下
  - **world size**：总的进程数量(原则上一个process占用一个GPU是较优的)
  - **rank**：当前进程的序号，用于进程间通讯，rank = 0 的主机为 master 节点
  - **local_rank**：当前进程对应的GPU号

举个例子 ：4台机器(每台机器8张卡)进行分布式训练， 通过 init_process_group() 对进程组进行初始化， 初始化后 可以通过 get_world_size() 获取到 world size，在该例中为32， 即有32个进程，其编号为0-31, 通过 get_rank() 函数可以进行获取 在每台机器上，local rank均为0-8，这是 local rank 与 rank 的区别， local rank 会对应到实际的GPU ID上 (单机多任务的情况下注意CUDA_VISIBLE_DEVICES的使用，控制不同程序可见的GPU device)。

- DDP基本用法

- - 使用 torch.distributed.init_process_group 初始化进程组
  - 使用 torch.nn.parallel.DistributedDataParallel 创建分布式并行模型
  - 创建对应的 DistributedSampler
  - 使用 torch.distributed.launch / torch.multiprocessing 或 slurm 开始训练

- 集体通信的使用

- - 将各卡数据进行汇总，分发，平均等操作，需要使用集体通讯操作，比如算accuracy或者总loss时候需要用到allreduce， 参考
  - torch.distribute
  - NCCL-Woolley
  - scaled_all_reduce

- 不同启动方式的用法

- - torch.distributed.launch：mnmc_ddp_launch.py
  - torch.multiprocessing：mnmc_ddp_mp.py
  - Slurm Workload Manager：mnmc_ddp_slurm.py

```python
"""
(MNMC) Multiple Nodes Multi-GPU Cards Training
    with DistributedDataParallel and torch.distributed.launch
Try to compare with [snsc.py, snmc_dp.py & mnmc_ddp_mp.py] and find out the differences.
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP

BATCH_SIZE = 256
EPOCHS = 5


if __name__ == "__main__":

    # 0. set up distributed device
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")

    # 1. define network
    net = torchvision.models.resnet18(pretrained=False, num_classes=10)
    net = net.to(device)
    # DistributedDataParallel
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    # 2. define dataloader
    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    # DistributedSampler
    # we test single Machine with 2 GPUs
    # so the [batch size] for each process is 256 / 2 = 128
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
    )

    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.01 * 2,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )

    if rank == 0:
        print("            =======  Training  ======= \n")

    # 4. start to train
    net.train()
    for ep in range(1, EPOCHS + 1):
        train_loss = correct = total = 0
        # set sampler
        train_loader.sampler.set_epoch(ep)

        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

            if rank == 0 and ((idx + 1) % 25 == 0 or (idx + 1) == len(train_loader)):
                print(
                    "   == step: [{:3}/{}] [{}/{}] | loss: {:.3f} | acc: {:6.3f}%".format(
                        idx + 1,
                        len(train_loader),
                        ep,
                        EPOCHS,
                        train_loss / (idx + 1),
                        100.0 * correct / total,
                    )
                )
    if rank == 0:
        print("\n            =======  Training Finished  ======= \n")

"""
usage:
>>> python -m torch.distributed.launch --help
exmaple: 1 node, 4 GPUs per node (4GPUs)
>>> python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=22222 \
    mnmc_ddp_launch.py
[init] == local rank: 3, global rank: 3 ==
[init] == local rank: 1, global rank: 1 ==
[init] == local rank: 0, global rank: 0 ==
[init] == local rank: 2, global rank: 2 ==
            =======  Training  ======= 
   == step: [ 25/49] [0/5] | loss: 1.980 | acc: 27.953%
   == step: [ 49/49] [0/5] | loss: 1.806 | acc: 33.816%
   == step: [ 25/49] [1/5] | loss: 1.464 | acc: 47.391%
   == step: [ 49/49] [1/5] | loss: 1.420 | acc: 48.448%
   == step: [ 25/49] [2/5] | loss: 1.300 | acc: 52.469%
   == step: [ 49/49] [2/5] | loss: 1.274 | acc: 53.648%
   == step: [ 25/49] [3/5] | loss: 1.201 | acc: 56.547%
   == step: [ 49/49] [3/5] | loss: 1.185 | acc: 57.360%
   == step: [ 25/49] [4/5] | loss: 1.129 | acc: 59.531%
   == step: [ 49/49] [4/5] | loss: 1.117 | acc: 59.800%
            =======  Training Finished  =======
exmaple: 1 node, 2tasks, 4 GPUs per task (8GPUs)
>>> CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="10.198.189.10" \
    --master_port=22222 \
    mnmc_ddp_launch.py
>>> CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="10.198.189.10" \
    --master_port=22222 \
    mnmc_ddp_launch.py
            =======  Training  ======= 
   == step: [ 25/25] [0/5] | loss: 1.932 | acc: 29.088%
   == step: [ 25/25] [1/5] | loss: 1.546 | acc: 43.088%
   == step: [ 25/25] [2/5] | loss: 1.424 | acc: 48.032%
   == step: [ 25/25] [3/5] | loss: 1.335 | acc: 51.440%
   == step: [ 25/25] [4/5] | loss: 1.243 | acc: 54.672%
            =======  Training Finished  =======
exmaple: 2 node, 8 GPUs per node (16GPUs)
>>> python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="10.198.189.10" \
    --master_port=22222 \
    mnmc_ddp_launch.py
>>> python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="10.198.189.10" \
    --master_port=22222 \
    mnmc_ddp_launch.py
[init] == local rank: 5, global rank: 5 ==
[init] == local rank: 3, global rank: 3 ==
[init] == local rank: 2, global rank: 2 ==
[init] == local rank: 4, global rank: 4 ==
[init] == local rank: 0, global rank: 0 ==
[init] == local rank: 6, global rank: 6 ==
[init] == local rank: 7, global rank: 7 ==
[init] == local rank: 1, global rank: 1 ==
            =======  Training  ======= 
   == step: [ 13/13] [0/5] | loss: 2.056 | acc: 23.776%
   == step: [ 13/13] [1/5] | loss: 1.688 | acc: 36.736%
   == step: [ 13/13] [2/5] | loss: 1.508 | acc: 44.544%
   == step: [ 13/13] [3/5] | loss: 1.462 | acc: 45.472%
   == step: [ 13/13] [4/5] | loss: 1.357 | acc: 49.344%
            =======  Training Finished  ======= 
"""
```

### Launch / Slurm 调度方式

这里单独用代码 imagenet.py 讲一下不同的启动方式。我们来看一下这个 `setup_distributed` 函数：

- 通过 srun 产生的程序在环境变量中会有 SLURM_JOB_ID， 以判断是否为slurm的调度方式
- **rank**通过 SLURM_PROCID 可以拿到
- **world size**实际上就是进程数， 通过 SLURM_NTASKS 可以拿到
- IP地址通过 `subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")` 巧妙得到，栗子来源于 MMCV
- 否则，就使用launch进行调度，直接通过 os.environ["RANK"] 和 os.environ["WORLD_SIZE"] 即可拿到 rank 和 world size

```python
# 此函数可以直接移植到你的程序中，动态获取IP，使用很方便
# 默认支持launch 和 srun 两种方式
def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
```

那提交任务就变成很自然的事情:

```python
# ======== slurm 调度方式 ========
# 32张GPU，4个node，每个node8张卡，8192的batch size，32个进程
# see：https://github.com/BIGBALLON/distribuuuu/blob/master/tutorial/imagenet.py
slurm example: 
    32GPUs (batch size: 8192)
    128k / (256*32) -> 157 itertaion
>>> srun --partition=openai -n32 --gres=gpu:8 --ntasks-per-node=8 --job-name=slrum_test \
    python -u imagenet.py
[init] == local rank: 7, global rank: 7 ==
[init] == local rank: 1, global rank: 1 ==
[init] == local rank: 4, global rank: 4 ==
[init] == local rank: 2, global rank: 2 ==
[init] == local rank: 6, global rank: 6 ==
[init] == local rank: 3, global rank: 3 ==
[init] == local rank: 5, global rank: 5 ==
[init] == local rank: 4, global rank: 12 ==
[init] == local rank: 1, global rank: 25 ==
[init] == local rank: 5, global rank: 13 ==
[init] == local rank: 6, global rank: 14 ==
[init] == local rank: 0, global rank: 8 ==
[init] == local rank: 1, global rank: 9 ==
[init] == local rank: 2, global rank: 10 ==
[init] == local rank: 3, global rank: 11 ==
[init] == local rank: 7, global rank: 15 ==
[init] == local rank: 5, global rank: 29 ==
[init] == local rank: 2, global rank: 26 ==
[init] == local rank: 3, global rank: 27 ==
[init] == local rank: 0, global rank: 24 ==
[init] == local rank: 7, global rank: 31 ==
[init] == local rank: 6, global rank: 30 ==
[init] == local rank: 4, global rank: 28 ==
[init] == local rank: 0, global rank: 16 ==
[init] == local rank: 5, global rank: 21 ==
[init] == local rank: 7, global rank: 23 ==
[init] == local rank: 1, global rank: 17 ==
[init] == local rank: 6, global rank: 22 ==
[init] == local rank: 3, global rank: 19 ==
[init] == local rank: 2, global rank: 18 ==
[init] == local rank: 4, global rank: 20 ==
[init] == local rank: 0, global rank: 0 ==
            =======  Training  ======= 
   == step: [ 40/157] [0/1] | loss: 6.781 | acc:  0.703%
   == step: [ 80/157] [0/1] | loss: 6.536 | acc:  1.260%
   == step: [120/157] [0/1] | loss: 6.353 | acc:  1.875%
   == step: [157/157] [0/1] | loss: 6.207 | acc:  2.465%


# ======== launch 调度方式 ========
# nproc_per_node: 每个node的卡数
# nnodes: node数量
# node_rank：node编号，从0开始
# see: https://github.com/BIGBALLON/distribuuuu/blob/master/tutorial/mnmc_ddp_launch.py
distributed.launch example: 
    8GPUs (batch size: 2048)
    128k / (256*8) -> 626 itertaion
>>> python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=22222 \
    imagenet.py
[init] == local rank: 0, global rank: 0 ==
[init] == local rank: 2, global rank: 2 ==
[init] == local rank: 6, global rank: 6 ==
[init] == local rank: 5, global rank: 5 ==
[init] == local rank: 7, global rank: 7 ==
[init] == local rank: 4, global rank: 4 ==
[init] == local rank: 3, global rank: 3 ==
[init] == local rank: 1, global rank: 1 ==
            =======  Training  ======= 
   == step: [ 40/626] [0/1] | loss: 6.821 | acc:  0.498%
   == step: [ 80/626] [0/1] | loss: 6.616 | acc:  0.869%
   == step: [120/626] [0/1] | loss: 6.448 | acc:  1.351%
   == step: [160/626] [0/1] | loss: 6.294 | acc:  1.868%
   == step: [200/626] [0/1] | loss: 6.167 | acc:  2.443%
   == step: [240/626] [0/1] | loss: 6.051 | acc:  3.003%
   == step: [280/626] [0/1] | loss: 5.952 | acc:  3.457%
   == step: [320/626] [0/1] | loss: 5.860 | acc:  3.983%
   == step: [360/626] [0/1] | loss: 5.778 | acc:  4.492%
   == step: [400/626] [0/1] | loss: 5.700 | acc:  4.960%
   == step: [440/626] [0/1] | loss: 5.627 | acc:  5.488%
   == step: [480/626] [0/1] | loss: 5.559 | acc:  6.013%
   == step: [520/626] [0/1] | loss: 5.495 | acc:  6.520%
   == step: [560/626] [0/1] | loss: 5.429 | acc:  7.117%
   == step: [600/626] [0/1] | loss: 5.371 | acc:  7.580%
   == step: [626/626] [0/1] | loss: 5.332 | acc:  7.907%
```

## 0X04 完整框架 Distribuuuu

Distribuuuu 是我闲(没)来(事)无(找)事(事)写的一个完整的纯DDP分类训练框架，足够精简且足够有效率。支持launch和srun两种启动方式，可以作为新手学习和魔改的样板工程。

```python
# 1 node, 8 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train_net.py --cfg config/resnet18.yaml
# see srun --help 
# and https://slurm.schedmd.com/ for details

# example: 64 GPUs
# batch size = 64 * 128 = 8192
# itertaion = 128k / 8192 = 156 
# lr = 64 * 0.1 = 6.4

srun --partition=openai-a100 \
     -n 64 \
     --gres=gpu:8 \
     --ntasks-per-node=8 \
     --job-name=Distribuuuu \
     python -u train_net.py --cfg config/resnet18.yaml \
     TRAIN.BATCH_SIZE 128 \
     OUT_DIR ./resnet18_8192bs \
     OPTIM.BASE_LR 6.4
```