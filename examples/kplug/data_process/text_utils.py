# coding=utf-8
# author: xusong <xusong28@jd.com>
# time: 2020/9/7 15:01


""" 数据清洗

## 参考

- https://github.com/CLUEbenchmark/CLUEPretrainedModels/blob/master/create_pretraining_data.py#L200
-
"""

import sys

sys.path.append('../../../data/vocab/')

from langconv import Converter


cn_punc = '，。；：？！（）～｜'  #
def q2b(uchar, skip_cn_punc=False):
    # 有时，希望保留全角中文标点，例如cn_punc。
    if skip_cn_punc and uchar in cn_punc:
        return uchar
    inside_code = ord(uchar)
    if inside_code == 12288:  # 全角空格直接转换
        inside_code = 32
    elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
        inside_code -= 65248
    return chr(inside_code)

def str_q2b(ustring, skip_cn_punc=False):
    """ 全角转半角 """
    return ''.join([q2b(uchar, skip_cn_punc) for uchar in ustring])


class TextProcessor:
    """
    专门为jd.vocab的数据清洗

    - 长度限制
    - 纠错  ㎜->6mm 5cm ㎝
    - 标点纠错
    - 多个空格
    - 繁体简体转化。google原始bert词典的话，就不用

    """

    def __init__(self):
        self.converter = Converter('zh-hans')

    def clean(self, text):
        """
        :param text:
        :return:
        """
        text = str_q2b(text, skip_cn_punc=True)  # 全角->半角
        text = self.converter.convert(text)  # 繁体->简体
        return text


def test():
    text = 'Ｌ，。T;；,桖 T恤。。 ∨领 V领 含格品三挡四挡\u200e新鲜大个的车厘子，第二\u200e天' \
           '鬼塚虎 （PHILIPS） 在4%～100%范围 透出光采 光采紧致 常用95－110度｛｜｝～“”' \
           '双层的压摺网纱裙摆，格纹针织百褶 抽摺裙、箱型摺裙、百摺裙、罗伞摺裙，裙摆采用荷叶边捏摺造型，捏摺灯笼袖，摺皱' \
           '血脉偾张,女孩7岁称髫年'
    p = TextProcessor()
    print(p.clean(text))


if __name__ == "__main__":
    test()
