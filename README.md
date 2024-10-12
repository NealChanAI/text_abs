## 运行

实验环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.9.7

(如果是Windows，请用bert4keras>=0.9.8)

首先请在`snippets.py`中修改相关路径配置，然后再执行下述代码。

训练代码：
```bash
#! /bin/bash

python extract_convert.py
python extract_vectorize.py

for ((i=0; i<15; i++));
    do
        python extract_model.py $i
    done

python seq2seq_convert.py
python seq2seq_model.py
```

预测代码
```python
from final import *
summary = predict(text, topk=3)
print(summary)
```
