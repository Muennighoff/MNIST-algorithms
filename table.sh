#!/bin/bash

echo "现在运行表格中的所有10个实验"

echo "第一： MLP-Base"
python entry.py --model MLP --dropout_proba 0.0 --exp MLP-Base