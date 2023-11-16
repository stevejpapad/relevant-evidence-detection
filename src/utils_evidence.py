import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ast import literal_eval
from unidecode import unidecode
from torch.utils.data import DataLoader

def str_to_list(df, list_columns = ['entities', 'q_detected_labels', 'captions', 'titles', 'images_paths', 'images_labels'] ):
    
    df[list_columns] = df[list_columns].fillna('[]')

    for column in list_columns:
        df[column] = df[column].apply(literal_eval)
        
    return df

def process_string(input_str):
    input_str = input_str.replace('&#39;', ' ')
    input_str = input_str.replace('<b>','')
    input_str = input_str.replace('</b>','')
    input_str = unidecode(input_str)  
    return input_str

def fetch_evidence_split(evidence_path):

    train_paths = pd.DataFrame(json.load(open(evidence_path + 'dataset_items_train.json'))).transpose()
    valid_paths = pd.DataFrame(json.load(open(evidence_path + 'dataset_items_val.json'))).transpose()
    test_paths = pd.DataFrame(json.load(open(evidence_path + 'dataset_items_test.json'))).transpose()

    train_paths = train_paths.reset_index().rename(columns={'index': 'match_index'})
    valid_paths = valid_paths.reset_index().rename(columns={'index': 'match_index'})
    test_paths = test_paths.reset_index().rename(columns={'index': 'match_index'})
    
    return train_paths, valid_paths, test_paths

def remove_duplicates_preserve_order(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

def idx_captions(df, prefix):
    
    captions_tuples = []
    
    for idx, caption_list in zip(df['match_index'], df['captions']):

        if isinstance(caption_list, list) and len(caption_list) > 0:
                        
            for i in range(len(caption_list)):
            
                captions_tuples.append((idx, idx + '_' + str(i), caption_list[i]))

        else:
            captions_tuples.append((idx, '-1', ''))     
        
    return pd.DataFrame(captions_tuples, columns=['match_index', 'X_item_index', 'X_caption'])

def idx_images(df, prefix):
    
    images_tuples = [] 
    
    for idx, images_list in zip(df['match_index'], df['images_paths']):

        if isinstance(images_list, list) and len(images_list) > 0:
            
            for i in range(len(images_list)):
            
                images_tuples.append((idx, idx + '_' + str(i), images_list[i]))
            
        else:
            images_tuples.append((idx, idx + '_0', ''))       
        
    return pd.DataFrame(images_tuples, columns=['match_index', 'X_item_index', 'X_image_path'])

def load_evidence_file(input_data, evidence_path):
    
    count_captions = 0
    all_samples = []

    for (row) in tqdm(input_data.itertuples(), total=input_data.shape[0]):

        texts_path = evidence_path + 'merged_balanced' + '/' + row.inv_path
        images_path = evidence_path + 'merged_balanced' + '/' + row.direct_path

        inv_annotation = json.load(open(texts_path + '/inverse_annotation.json'))
        q_detected_labels = json.load(open(texts_path + '/query_detected_labels'))

        img_detected_labels = json.load(open(images_path + '/detected_labels'))
        direct_annotation = json.load(open(images_path + '/direct_annotation.json'))

        images_keep_ids = json.load(open(images_path + '/imgs_to_keep_idx'))['index_of_images_tokeep']
        all_image_paths = [images_path + '/' + x + '.jpg' for x in images_keep_ids]

        titles = []

        if 'all_fully_matched_captions' in inv_annotation.keys():    
            titles = [x.get('title', '') for x in inv_annotation['all_fully_matched_captions']]


        if os.path.isfile(texts_path + '/captions_info'):

            count_captions += 1

            caption = json.load(open(texts_path + '/captions_info'))
            caption_keep_ids = json.load(open(texts_path + '/captions_to_keep_idx'))
            keep_captions = [caption['captions'][x] for x in caption_keep_ids['index_of_captions_tokeep']]

        else:
            keep_captions = []
    
        
        processed_list = [process_string(x) for x in keep_captions]
        keep_captions = remove_duplicates_preserve_order(keep_captions) # processed_list
        
        sample = {
            'match_index': row.match_index,
            'entities': inv_annotation['entities'],
            'entities_len': len(inv_annotation['entities']),
            'q_detected_labels': q_detected_labels['labels'],
            'q_detected_labels_len': len(q_detected_labels['labels']),
            'captions': keep_captions,
            'len_captions': len(keep_captions),
            'titles': titles,
            'titles_len': len(titles),
            'images_paths': all_image_paths,
            'len_images': len(all_image_paths),
            'images_labels': [img_detected_labels[img_id]['labels'] for img_id in images_keep_ids]
        }

        all_samples.append(sample)

    return all_samples

def load_merge_evidence_w_data(input_data, data_paths, evidence_path):
    
    data_evidence = load_evidence_file(data_paths, evidence_path)
    data_evidence = pd.DataFrame(data_evidence)

    data_evidence.match_index = data_evidence.match_index.astype('int')
    data_merge = pd.merge(data_evidence, input_data, on='match_index', how='right')
    
    return data_merge


class ImageIteratorSource(torch.utils.data.Dataset):
    def __init__(
        self,
        input_data,
        vis_processors,
    ):
        self.input_data = input_data
        self.vis_processors = vis_processors

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        current = self.input_data.iloc[idx]
        
        img_path = current.X_image_path
        idx = current.X_item_index
                
        try:
            
            image = Image.open(img_path)
            image = image.convert('RGB')
        
            max_size = 400

            width, height = image.size

            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            image = image.resize((new_width, new_height))

            img = self.vis_processors["eval"](image)

            return idx, img
        
        except Exception as e:
            print(e)
            print(idx)
    
class TextIteratorSource(torch.utils.data.Dataset):
    def __init__(
        self,
        input_data,
        txt_processors
    ):
        self.input_data = input_data
        self.txt_processors = txt_processors

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        current = self.input_data.iloc[idx]
        
        txt = self.txt_processors["eval"](current.X_caption)
        idx = current.X_item_index

        return idx, txt
    
def prepare_source_dataloaders(image_data, text_data, vis_processors, txt_processors, batch_size, num_workers, shuffle):

    img_dg = ImageIteratorSource(image_data,  vis_processors)

    img_dataloader = DataLoader(
        img_dg,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    txt_dg = TextIteratorSource(text_data,  txt_processors)

    txt_dataloader = DataLoader(
        txt_dg,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return img_dataloader, txt_dataloader
    