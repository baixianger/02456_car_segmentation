- title: "How to running Jupiter Notebook with interactive jobs on NVIDIA GPUs from DTU HPC"
  author:
- author:白翔
- date: 2021-12-31

# How to running Jupiter Notebook with interactive jobs on NVIDIA GPUs from DTU HPC

如何在学校的GPU节点上运行jupyter notebook

## 1 Start service

启动jupyter笔记本

### 1.1 Access the DTU HPC service using an SSH tunnel

通过ssh连接学校服务器

```bash
## ssh tunnel
ssh s213120@login2.gbar.dtu.dk
```

```bash
## type in your **passwords**
```

### 1.2 Running interactively on GPUs

申请GPU资源

```shell
## Tesla V100 node with 16GB memory
voltash 
## Tesla V100-SXM2 node with 32GB memory
sxm2sh
## A100-GPU node with 40GB memory
a100sh
```

- Use `nvidia-smi` to monitor which GPUs are currently occupied

  在终端中运行 `nvidia-smi`查看GPU资源
- If use the latest GPU, please install the newest Pytorch or TensorFlow (or will occur incompatibility):

  如果使用最新GPU, 注意跟新pytorch和tensorflow(检查兼容的cuda版本), A100核心只兼容cuda11:

  `pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

### 1.3 Running Jupyter Notebook

启动jupyter笔记本并设置访问端口和访问地址

```shell
jupyter notebook --no-browser --port=40000 --ip=$HOSTNAME
```

## 2 SSH Port Forwarding

ssh端口映射

```shell
ssh s213120@login2.gbar.dtu.dk -g -L8080:n-62-12-19:40001 -N
ssh s213120@login2.gbar.dtu.dk -g -L8080:n-62-20-9:40001 -N
```

![1640907053963.png](image//1640907053963.png)

## 3 Starting a browser

在本地浏览器访问端口映射后的jupyter服务

Start your local browser and write in the address bar:

```shell
http://localhost:8080
## get token from the start log of jupyter service
## 密码可以从jupyter的启动日志中获得
http://localhost:8080/?token=<get-from-starting-Log>
```

Check CUDA / 检查CUDA

Run python env / 启动python环境后 :

```python
import torch
print(torch.cuda.is_available())
```
