#!/bin/bash

echo "现在运行表格中的所有21个实验 ; 如果用GPU就更快"

echo "第一： MLP-Base"
python entry.py --model MLP --dropout_proba 0.0 --exp MLP-Base

echo "第二： CNN-Base"
python entry.py --model CNN --dropout_proba 0.0 --exp CNN-Base

echo "第三： VIT-Base"
python entry.py --model VIT --dropout_proba 0.0 --exp VIT-Base

########################

echo "第四： MLP-Dropout"
python entry.py --model MLP --dropout_proba 0.1 --exp MLP-Dropout

echo "第五： CNN-Dropout"
python entry.py --model CNN --dropout_proba 0.1 --exp CNN-Dropout

echo "第六： VIT-Dropout"
python entry.py --model VIT --dropout_proba 0.1 --exp VIT-Dropout

########################

echo "第七： MLP-Dropout-Zeros"
python entry.py --model MLP --dropout_proba 0.1 --init_func zeros --exp MLP-Dropout-Zeros

echo "第八： CNN-Dropout-Zeros"
python entry.py --model CNN --dropout_proba 0.1 --init_func zeros --exp CNN-Dropout-Zeros

echo "第九： VIT-Dropout-Zeros"
python entry.py --model VIT --dropout_proba 0.1 --init_func zeros --exp VIT-Dropout-Zeros

########################

echo "第十： MLP-Dropout-Uniform(0,1)"
python entry.py --model MLP --dropout_proba 0.1 --init_func uniform --uniform_low 0 --uniform_high 1 --exp MLP-Dropout-Uniform01

echo "第十一： CNN-Dropout-Uniform(0,1)"
python entry.py --model CNN --dropout_proba 0.1 --init_func uniform --uniform_low 0 --uniform_high 1 --exp CNN-Dropout-Uniform01

echo "第十二： VIT-Dropout-Uniform(0,1)"
python entry.py --model VIT --dropout_proba 0.1 --init_func uniform --uniform_low 0 --uniform_high 1 --exp VIT-Dropout-Uniform01

########################

echo "第十三： MLP-Dropout-Uniform(-1,1)"
python entry.py --model MLP --dropout_proba 0.1 --init_func uniform --uniform_low -1 --uniform_high 1 --exp MLP-Dropout-Uniform-11

echo "第十四： CNN-Dropout-Uniform(-1,1)"
python entry.py --model CNN --dropout_proba 0.1 --init_func uniform --uniform_low -1 --uniform_high 1 --exp CNN-Dropout-Uniform-11

echo "第十五： VIT-Dropout-Uniform(-1,1)"
python entry.py --model VIT --dropout_proba 0.1 --init_func uniform --uniform_low -1 --uniform_high 1 --exp VIT-Dropout-Uniform-11

########################

echo "第十六： MLP-Dropout-Uniform(-0.1,0.1)"
python entry.py --model MLP --dropout_proba 0.1 --init_func uniform --uniform_low -0.1 --uniform_high 0.1 --exp MLP-Dropout-Uniform-0.10.1

echo "第十七： CNN-Dropout-Uniform(-0.1,0.1)"
python entry.py --model CNN --dropout_proba 0.1 --init_func uniform --uniform_low -0.1 --uniform_high 0.1 --exp CNN-Dropout-Uniform-0.10.1

echo "第十八： VIT-Dropout-Uniform(-0.1,0.1)"
python entry.py --model VIT --dropout_proba 0.1 --init_func uniform --uniform_low -0.1 --uniform_high 0.1 --exp VIT-Dropout-Uniform-0.10.1

########################

echo "第十九： MLP-Dropout-Xavier"
python entry.py --model MLP --dropout_proba 0.1 --init_func xavier --exp MLP-Dropout-Xavier

echo "第二十： CNN-Dropout-Xavier"
python entry.py --model CNN --dropout_proba 0.1 --init_func xavier --exp CNN-Dropout-Xavier

echo "第二十一： VIT-Dropout-Xavier"
python entry.py --model VIT --dropout_proba 0.1 --init_func xavier --exp VIT-Dropout-Xavier

########################

echo "第二十二：CNN-Dropout-BatchNorm"
python entry.py --model CNN2 --dropout_proba 0.1 --exp CNN-Dropout-BatchNorm