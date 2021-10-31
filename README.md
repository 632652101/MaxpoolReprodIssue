# MaxpoolReprodIssue

### Step 1: Clone

```
# Clone this repo.
git clone https://github.com/632652101/MaxpoolReprodIssue.git
cd Visualize-PP
exprot PYTHONPATH=./
```



### Step 2: 检查前向对齐

```
python forward_alexnet_pp.py
python forward_alexnet_torch.py
python check_forward.py
```



### Step 3: 运行主函数代码

torch 和 paddle 版本前几个跌打误差不大， 后面随着iteration的真大，两者的结果无法对齐。

```
python method_2_pp_alexnet.py
python method_2_torch_alexnet.py
python check_res.py
```





网络的权重文件可以[在此](https://pan.baidu.com/s/1HkRrEsjpn1iQMAYVeNSeAQ)下载（百度网盘提取码： wgc6）下载， 下载后放到 weights/ 下。