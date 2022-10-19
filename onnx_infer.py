import re

from transformers import BertTokenizer
import numpy as np
import onnx
import onnxruntime as ort

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

def clean_text(text, punc_list):
    text = text.lower()
    text = re.sub('[^A-Za-z0-9\u4e00-\u9fa5]', '', text)
    text = re.sub(f'[{"".join([p for p in punc_list][1:])}]', '', text)
    return text

# 后处理识别结果
def post_process(tokenizer, input_ids, result, punc_list):
    seq_len = len(input_ids[0])
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0][1:seq_len - 1])
    labels = result[1:seq_len - 1].tolist()
    assert len(tokens) == len(labels)
    text = ''
    for t, l in zip(tokens, labels):
        text += t
        if l != 0:
            text += punc_list[l]
    return text

if __name__ == '__main__':
    # load onnx model
    onnx_path = 'D:\\project\\pycharm_workspace\\punctuation_prediction\\model\\ernie_onnx\\pun_model.onnx'
    model = onnx.load(onnx_path)
    sess = ort.InferenceSession(bytes(model.SerializeToString()))
    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained("D:\\project\\pycharm_workspace\\punctuation_prediction\\model\\pun_models_pytorch")
    # punc
    punc_list = []
    punc_list.append('')
    punc_list.append('，')
    punc_list.append('。')
    punc_list.append('？')

    sent = '从小我有个梦想这个梦想是我想当一个科学家'
    sent = '中国的首都是北京我爱我的祖国'
    sent = '早上起来穿衣吃饭后我就上学了在路上碰见了许久不见的一个朋友'
    sent = clean_text(sent, punc_list)
    result, input_ids = onnx_infer(sess, tokenizer, sent)
    text = post_process(tokenizer, input_ids, result, punc_list)
    print(text)

