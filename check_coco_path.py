import os
import pickle

src = './dcama_mask_mixing/data/splits/coco/trn/fold0.pkl'
new_src = './dcama_mask_mixing/data/splits/new_coco/trn/fold0.pkl'

data = pickle.load(open(src, 'rb'))

new_dict = {}

source = './data/coco/'
for key, value in data.items():
    new_list = []

    for name in value:
        img_path = source + name

        if os.path.exists(img_path):
            new_list.append(name)
        else:
            print(img_path)

    new_dict[key] = new_list

with open(new_src, 'wb') as handle:
    pickle.dump(new_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
