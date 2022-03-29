# coding=utf-8
# author: xusong <xusong28@jd.com>
# time: 2021/9/17 17:43

import torch
from transformers import BertTokenizer, BertConfig
from modeling_kplug import KplugForMaskedLM

model_dir = "../../models/hugging_face/kplug/"

tokenizer = BertTokenizer.from_pretrained(model_dir)
model = KplugForMaskedLM.from_pretrained(model_dir)

input_ids = torch.tensor(tokenizer.encode("这款连[MASK]裙真漂亮", add_special_tokens=True)).unsqueeze(0)
outputs = model(input_ids)


# fill mask
from transformers import FillMaskPipeline
from transformers import MODEL_FOR_MASKED_LM_MAPPING
MODEL_FOR_MASKED_LM_MAPPING[BertConfig] = KplugForMaskedLM
fill_masker = FillMaskPipeline(model=model, tokenizer=tokenizer)
outputs = fill_masker(f"这款连[MASK]裙真漂亮")
print(outputs)

