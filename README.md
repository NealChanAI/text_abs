## 运行


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
