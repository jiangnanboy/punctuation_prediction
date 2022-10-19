#### 中文句子标点符号预测

对一个没有标点符号的句子预测标点，主要预测逗号、句号以及问号（，。？）

###### 给句子添加标点符号

请下载模型 [pun_model.onnx]，将模型放入model/ernie_onnx目录下。

链接：https://pan.baidu.com/s/1l62YmuU3giNPkT2TonZRKA 
提取码：sy12

```
def onnx_infer(sess, tokenizer, sent):
    tokenized_tokens = tokenizer(sent)
    input_ids = np.array([tokenized_tokens['input_ids']], dtype=np.int64)
    token_type_ids = np.array([tokenized_tokens['token_type_ids']], dtype=np.int64)
    result = sess.run(
        output_names=None,
        input_feed={"input_ids": input_ids,
                    "token_type_ids": token_type_ids}
    )[0]
    return result, input_ids
```

输出结果：
```
sent: 从小我有个梦想这个梦想是我想当一个科学家 -> result: 从小我有个梦想，这个梦想是我想当一个科学家。
------------------------------------------------
sent: 中国的首都是北京我爱我的祖国 -> result: 中国的首都是北京。我爱我的祖国。
------------------------------------------------
sent: 早上起来穿衣吃饭后我就上学了在路上碰见了许久不见的一个朋友 -> result: 早上起来，穿衣吃饭后，我就上学了，在路上碰见了许久不见的一个朋友。

```