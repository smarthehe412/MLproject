### Using single-label models for multi-label image classification

需要环境：torch 2.3.0, 对应 CUDA 与 CUDNN

首先配置 fiftyone

```bash
pip install fiftyone
```

随后配置其他库，如 tqdm，PIL，pandas，numpy，torchsummary，torchvision

运行 `get_labels.py`，下载数据集并从数据集获得 labels.json

每个文件夹是一个模型的主体（attention，fastercnn-resNet50 已被弃用）。

resNet-50 训练：将 train.py 复制到项目根目录，运行获得 model_complete.pth

resNet-50-reduced 训练：将 train.py 和 labels.json 一同复制到项目根目录，运行获得 model_complete.pth

GCN, GCN-resNet-101 训练：将 train.py 和 get_adjacency.py 一同复制到项目根目录，运行 get_adjacency.py 获得 adjacency_file.pkl，然后运行 train.py 获得 model_complete.pth

评价：获得 model_complete.pth 后，复制对应模型的 evaluate.py 到项目根目录并运行，获得 result.json

计算与可视化：获得 result.json 后，运行项目根目录下的 calc.py，获得各项分数，同时启动 fiftyone 客户端，可在 http://localhost:5151/ 上查看 validation split 的图片。在终端上每次点击 enter 会获得下一张图片对应的 predicted label 与 ground truth.