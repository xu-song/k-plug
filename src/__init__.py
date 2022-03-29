
# hack/patch
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
