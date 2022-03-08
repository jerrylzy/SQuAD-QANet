import numpy as np
from ujson import load as json_load
from ujson import dump as json_dump

num_samples = 512
# with open('data/dev_eval.json', 'r') as fh:
#     gold_dict = json_load(fh)
data = np.load('data/train.npz')
# dev = np.load('data/dev.npz')
data_len = len(data['context_idxs'])
# indices = np.random.choice(data_len, num_samples, replace=False)

# smaller_gold_dict = { key: gold_dict[key] for key in gold_dict if int(key) < 100}
# with open('data/smaller_dev_eval.json', 'w') as fh:
#     json_dump(smaller_gold_dict, fh)

np.savez('data/smaller_train.npz',
         context_idxs=data['context_idxs'][:num_samples],
         context_char_idxs=data['context_char_idxs'][:num_samples],
         ques_idxs=data['ques_idxs'][:num_samples],
         ques_char_idxs=data['ques_char_idxs'][:num_samples],
         y1s=data['y1s'][:num_samples],
         y2s=data['y2s'][:num_samples],
         ids=data['ids'][:num_samples])

# np.savez('data/smaller_dev.npz',
#          context_idxs=data['context_idxs'][:100],
#          context_char_idxs=data['context_char_idxs'][:100],
#          ques_idxs=data['ques_idxs'][:100],
#          ques_char_idxs=data['ques_char_idxs'][:100],
#          y1s=data['y1s'][:100],
#          y2s=data['y2s'][:100],
#          ids=data['ids'][:100])
