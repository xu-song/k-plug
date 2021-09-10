
# hack/patch
from .tasks import fairseq_task_patch
from .models import bart_patch

# 通用
from .tasks import sentence_multilabel
from .criterions import multilabel_bce_criterion

# 加强版Transformer
# from .models import transformer_plus

# MASS
from .models import transformer_mass
from .tasks import multitask_s2s_mass


# K-PLUG
from .models import transformer_kplug
from .tasks import multitask_s2s_kplug, sequence_tagging, sentence_prediction_bert
from .criterions import auto_criterion, sequence_tagging


# SAGCopy
from .models import transformer_sagcopy
from .criterions import label_smoothed_cross_entropy_with_guidance


# KGNet
from .models import transformer_kgnet
from .tasks import multitask_s2s_kgnet
from .tasks import multitask_s2s_kgnet_for_labeling    # 利用编码器进行NLU
from .tasks import multitask_s2s_kgnet_for_labeling_with_decoder   # 利用解码器进行NLU


# Denosing Teacher Forcing
from .models import transformer_denosing_tf
from .tasks import denosing_teacher_forcing



# MPNet
# from .models import mpnet
# from .tasks import masked_permutation_lm
# from .criterions import masked_permutation_criterion


# BART



"""
ProphetNet  
"""
# from .models import transformer_ngram
# from .modules import ngram_multihead_attention
# from .criterions import ngram_criterions
