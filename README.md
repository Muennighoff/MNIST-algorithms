# 人工智能与商业创新第3次作业

先安装 PyTorch, Numpy和Pandas：
`pip install -r requirements.txt`

解压数据：
（数据是本来的MNST数据集：https://www.kaggle.com/oddrationale/mnist-in-csv）
```
unzip /content/mnist_test.csv.zip -d ./MNIST-algorithms/data/test/
unzip /content/mnist_train.csv.zip -d ./MNIST-algorithms/data/train/
mv ./MNIST-algorithms/data/test/mnist_test.csv ./MNIST-algorithms/data/test.csv
mv ./MNIST-algorithms/data/train/mnist_train.csv ./MNIST-algorithms/data/train.csv
```

训练，验证和测试 - 比如：
`python entry.py --model CNN --dropout_proba 0.1 --init_func normal --batch_size 64 --lr 1e-4`
可以看entry.py的argparse就用任何参数。

重现pdf中的表, 如果不用GPU，VIT模型比较慢：
`bash table.sh`

