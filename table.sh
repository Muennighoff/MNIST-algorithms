#!/bin/bash

echo "现在运行表格中的所有10个实验 ; 如果用GPU就更快"

echo "第一： MLP-Base"
python entry.py --model MLP --dropout_proba 0.0 --exp MLP-Base

echo "第二： CNN-Base"
python entry.py --model MLP --dropout_proba 0.0 --exp CNN-Base

echo "第三： VIT-Base"
python entry.py --model MLP --dropout_proba 0.0 --exp VIT-Base


echo "第四： MLP-Dropout"
python entry.py --model MLP --dropout_proba 0.1 --exp MLP-Dropout

echo "第五： CNN-Dropout"
python entry.py --model MLP --dropout_proba 0.1 --exp CNN-Dropout

echo "第六： VIT-Dropout"
python entry.py --model MLP --dropout_proba 0.1 --exp VIT-Dropout


