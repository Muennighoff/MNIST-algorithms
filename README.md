# MNIST-algorithms

先安装 PyTorch, Numpy和Pandas：
`pip install -r requirements.txt`

解压数据：
`unzip ./data/*.zip -d ./data/`

训练，验证和测试 - 比如：
`python entry.py --model CNN --dropout_proba 0.1 --init_func normal --batch_size 64 --lr 1e-4`
可以看entry.py的argparse就用任何参数。

重现pdf中的表, 如果不用GPU，VIT模型比较慢：
`bash table.sh`
