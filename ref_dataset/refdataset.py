import os

import numpy as np
import torch.utils.data as data
from PIL import Image
from torch.utils.data.dataloader import default_collate

from .refer import REFER


class Refdataset(data.Dataset):
    def __init__(self,
                 refer_data_root='./data',
                 dataset='refcoco',
                 splitBy='unc',
                 split='train',
                 image_transforms=None,
                ):

        self.images_transforms = image_transforms
        self.root = refer_data_root
        self.refer = REFER(refer_data_root, dataset, splitBy)
        self.split = split
        print('Preparing dataset ......')
        print(dataset, split)
        print(refer_data_root, dataset, splitBy)

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        
        self.ref_ids = ref_ids

        self.all_sentences = []
        self.all_category_id = []
        for index, x in enumerate(self.ref_ids):
            
            ref = self.refer.Refs[x]

            sentences_raw_for_ref = []

            for i,(sents,sent_id) in enumerate(zip(ref['sentences'],ref['sent_ids'])):
                sentence_raw = sents['sent']
                sentences_raw_for_ref.append(sentence_raw)
            
            self.all_sentences.append(sentences_raw_for_ref)
            self.all_category_id.append(ref['category_id'])
        
        #category_ids
        self.classes_ids = [None] * 91

        with open(os.path.join(self.root, 'label_orgid.txt')) as file:
            for line in file:
                category_id, class_name = line.strip().split(' ', 1)
                self.classes_ids[int(category_id)] = class_name

        with open(os.path.join(self.root, "labels.txt")) as f:
            self.all_class_name = f.readlines()
        self.all_class_name = [t.strip() for t in self.all_class_name]


    def __len__(self):
        return len(self.ref_ids)
    
    def __getitem__(self, index) :
        the_ref_id = self.ref_ids[index]
        the_img_id = self.refer.getImgIds(the_ref_id)
        the_img = self.refer.Imgs[the_img_id[0]]

        img_path = os.path.join(self.refer.IMAGE_DIR, the_img['file_name'])

        img = Image.open(img_path).convert("RGB")
        ref = self.refer.Refs[the_ref_id]

        ref_mask = np.array(self.refer.getMask(ref)['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1
        org_gt = annot
        annot = Image.fromarray(annot.astype(np.uint8), mode='P')


        if self.images_transforms is not None:
            h, w = ref_mask.shape
            img, target = self.images_transforms(img, annot)
            target = target.unsqueeze(0)
        
        sentence = self.all_sentences[index]
        category_id = self.all_category_id[index]
        class_name = self.classes_ids[category_id]

        if self.split == 'train':
            chosen_sentence = np.random.choice(sentence)
        else:
            chosen_sentence = sentence # list of sentences

        batch = {
            'query_img':img,
            'query_mask':target,
            'query_idx':index,
            'sentence':chosen_sentence,
            'category_id':category_id,
            # 'class_id':class_id,
            'class_name':class_name,
            "image_filename":the_img['file_name'],
            'org_gt':org_gt,
        }
        
        return batch
    
def collate_fn(batch):
    batched_data = {}
    for key in batch[0].keys():
        if key == 'sentence' and isinstance(batch[0][key], list):
            batched_data[key] = [d[key] for d in batch]
        elif key == 'org_gt':
            batched_data[key] = [d[key] for d in batch]
        else:
            batched_data[key] = default_collate([d[key] for d in batch])

    return batched_data
