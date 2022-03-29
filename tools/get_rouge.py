import files2rouge
import sys
from tqdm import tqdm, trange
ref = sys.argv[1]     # tgt
result = sys.argv[2]  # hyp

ref_list = open(ref).readlines()
result_list = open(result).readlines()
f_ref = open('ref_file', 'w')
f_result = open('hyp_file', 'w')

for i in trange(len(ref_list)):
    ref = ''.join(ref_list[i].strip().split())
    result = ''.join(result_list[i].strip().split())
    w2id = {}
    idx = 0
    for w in ref + result:
        if w not in w2id:
            w2id[w] = str(idx)
            idx += 1
    ref_ids = [w2id[w] for w in ref]
    result_ids = [w2id[w] for w in result]
    f_ref.write(' '.join(ref_ids) + '\n')
    f_result.write(' '.join(result_ids) + '\n')

f_ref.close()
f_result.close()
files2rouge.run('hyp_file', 'ref_file')

