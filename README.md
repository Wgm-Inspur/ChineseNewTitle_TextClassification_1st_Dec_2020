# 飞桨常规赛：中文新闻文本标题分类 - 12月第1名方案

![GitHub forks](https://img.shields.io/github/forks/GT-ZhangAcer/PythonRepository-Template?style=for-the-badge) ![GitHub Repo stars](https://img.shields.io/github/stars/GT-ZhangAcer/PythonRepository-Template?style=for-the-badge) 

## 1、单模预测

根据官方提供的训练集和验证集，采用多种策略，不断优化参数，提高单个模型的准确率。
### 1.1、使用roberta-wwm-ext-large
``` 
使用训练集（train.txt）训练模型，参数如下：
batch_size = 300，max_seq_length = 48，epochs = 4，seed = 1024；优化器上选择AdamW优化器，learning_rate = 4e-5，warmup_proportion = 0.1，weight_decay = 0.0

在训练集上的准确率：99.03%
使用验证集（dev.txt）进行验证，准确率：99.62%
对测试集进行预测，提交结果的分值：89.08。
将结果文件重命名为：result89.08.txt，保存在“single_result”目录。
``` 

### 1.2、使用nezha-large-wwm-chinese
``` 
使用训练集（train.txt）训练模型，参数如下：
batch_size = 256，max_seq_length = 48，epochs = 4，seed = 1024；优化器上选择AdamW优化器，learning_rate = 4e-5，warmup_proportion = 0.1，weight_decay = 0.0

在训练集上的准确率：99.02%。
使用验证集（dev.txt）进行验证，准确率：99.53%。
对测试集进行预测，提交结果的分值：88.96。
将结果文件重命名为：result88.96.txt，保存在“single_result”目录。
```

### 1.3、使用skep_ernie_1.0_large_ch
``` 
使用训练集（train.txt）训练模型，参数如下：
batch_size = 300，max_seq_length = 48，epochs = 4，seed = 1024；优化器上选择AdamW优化器，learning_rate = 4e-5，warmup_proportion = 0.1，weight_decay = 0.0

在训练集上的准确率：98.43%。
使用验证集（dev.txt）进行验证，准确率：99.24%。
对测试集进行预测，提交结果的分值：88.82。
将结果文件重命名为：result88.82.txt，保存在“single_result”目录。
```

### 1.4、使用bert-wwm-ext-chinese
``` 
使用训练集（train.txt）训练模型，参数如下：
batch_size = 300，max_seq_length = 48，epochs = 6，seed = 1024；优化器上选择AdamW优化器，learning_rate = 4e-5，warmup_proportion = 0.1，weight_decay = 0.0

在训练集上的准确率：98%。
使用验证集（dev.txt）进行验证，准确率：98.94%。
对测试集进行预测，提交结果的分值：88.62。
将结果文件重命名为：result88.62.txt，保存在“single_result”目录。
```

### 1.5、使用macbert-large-chinese
``` 
使用训练集（train.txt）训练模型，参数如下：
batch_size = 300，max_seq_length = 48，epochs = 4，seed = 1024；优化器上选择AdamW优化器，learning_rate = 4e-5，warmup_proportion = 0.1，weight_decay = 0.0

在训练集上的准确率：98.37%。
使用验证集（dev.txt）进行验证，准确率：99.19%。
对测试集进行预测，提交结果的分值：88.75。
将结果文件重命名为：result88.75.txt，保存在“single_result”目录。
```

### 1.6、使用huhuiwen/mengzi-bert-base
``` 
使用训练集（train.txt）训练模型，参数如下：
batch_size = 300，max_seq_length = 48，epochs = 4，seed = 1024；优化器上选择AdamW优化器，learning_rate = 4e-5，warmup_proportion = 0.1，weight_decay = 0.0

在训练集上的准确率：97.77%。
使用验证集（dev.txt）进行验证，准确率：98.63%。
对测试集进行预测，提交结果的分值：88.64。
将结果文件重命名为：result88.64.txt，保存在“single_result”目录。
```

### 1.7、使用junnyu/hfl-chinese-electra-180g-base-discriminator
``` 
使用训练集（train.txt）训练模型，参数如下：
batch_size = 300，max_seq_length = 48，epochs = 4，seed = 1024；优化器上选择AdamW优化器，learning_rate = 4e-5，warmup_proportion = 0.1，weight_decay = 0.0

在训练集上的准确率：97.44%。
使用验证集（dev.txt）进行验证，准确率：98.31%。
对测试集进行预测，提交结果的分值：88.28。
将结果文件重命名为：result88.28.txt，保存在“single_result”目录。
```

## 2、单模融合

采用等权投票法、加权投票法进行集成学习，提高模型的效果。
### 2.1、等权投票法进行融合
``` 
采用相等的权重融合1.1至1.7的结果文件，提交结果的分值：89.494。
结果文件重命名为：result_merge89.494.txt，保存在“merge_result”目录。
``` 

### 2.2、加权投票法进行融合
``` 
采用权重158、146、132、112、125、114、78融合1.1至1.7的结果文件，提交结果的分值：89.527。
结果文件重命名为：result_merge89.527.txt，保存在“merge_result”目录。
```

## 3、直推式学习

采用直推式学习生成伪标签数据集，扩充训练集的规模。
### 3.1、伪标签生成
``` 
采用1.1至1.7的模型处理测试集，将预测值相同的结果视为伪标签，包含77076条文本。
伪标签文件重命名为：fakeData1.csv，保存在“fake_data”目录。
``` 

### 3.2、模型训练
``` 
采用roberta-wwm-ext-large，按照下述参数在训练集（train.txt）上进行训练：
batch_size = 300，max_seq_length = 48，epochs = 4，seed = 1024；优化器上选择AdamW优化器，learning_rate = 4e-5，warmup_proportion = 0.1，weight_decay = 0.0

在训练集上的准确率：99.11%
使用验证集（dev.txt）进行验证，准确率：99.60%
对测试集进行预测，提交结果的分值：90.01, 将结果文件重命名为：result90.01.txt，保存在“fake_result”目录。

在已训练模型基础上继续训练，参数调整如下：
batch_size = 300，max_seq_length = 48，epochs = 6，seed = 0；优化器上选择AdamW优化器，learning_rate = 1e-5，warmup_proportion = 0.1，weight_decay = 0.0

在训练集上的准确率：99.53%
使用验证集（dev.txt）进行验证，准确率：99.76625%
对测试集进行预测，提交结果的分值：90.06447。
将结果文件重命名为：result90.06447.txt，保存在“fake_result”目录。
```

