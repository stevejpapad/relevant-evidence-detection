import os
import json
import torch
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from string import digits
from ast import literal_eval
from utils import load_visual_news, load_features, load_training_dataset
from utils_evidence import (fetch_evidence_split, load_merge_evidence_w_data, str_to_list, process_string, idx_captions, idx_images, prepare_source_dataloaders)
from lavis.models import load_model_and_preprocess
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def load_merge_evidence_data(evidence_path, data_path, data_name):

    print("Load evidence paths")
    train_paths, valid_paths, test_paths = fetch_evidence_split(evidence_path)
    
    print("Load", data_name)
    train_data, valid_data, test_data = load_training_dataset(data_path, data_name)

    train_data = train_data.reset_index().rename(columns={'index': 'match_index'})
    valid_data = valid_data.reset_index().rename(columns={'index': 'match_index'})
    test_data = test_data.reset_index().rename(columns={'index': 'match_index'})

    print("Prepare train - evidence")
    train_merge = load_merge_evidence_w_data(train_data, train_paths, evidence_path)
    print("Save data+evidence", data_path + 'news_clippings/merged_balanced_train.csv')
    train_merge.to_csv(data_path + 'news_clippings/merged_balanced_train.csv')  

    print("Prepare valid - evidence")
    valid_merge = load_merge_evidence_w_data(valid_data, valid_paths, evidence_path)
    print("Save data+evidence", data_path + 'news_clippings/merged_balanced_valid.csv')
    valid_merge.to_csv(data_path + 'news_clippings/merged_balanced_valid.csv')
    
    print("Prepare test - evidence")
    test_merge = load_merge_evidence_w_data(test_data, test_paths, evidence_path)
    print("Save data+evidence", data_path + 'news_clippings/merged_balanced_test.csv')        
    test_merge.to_csv(data_path + 'news_clippings/merged_balanced_test.csv')


def extract_features_for_evidence(data_path, output_path, data_name_X, encoder='CLIP', choose_encoder_version='ViT-B/32', choose_gpu=0, batch_size=256, num_workders=16, shuffle = False):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_gpu)
    encoder_version = choose_encoder_version.replace('-', '').replace('/', '')    

    device = torch.device(
        "cuda:"+str(choose_gpu) if torch.cuda.is_available() else "cpu"
    )

    if encoder == 'CLIP' and choose_encoder_version == "ViT-L/14":
        model, vis_processors, txt_processors = load_model_and_preprocess(name="clip_feature_extractor", 
                                                                          model_type="ViT-L-14", 
                                                                          is_eval=True, 
                                                                          device=device)     
        
    elif encoder == 'CLIP' and choose_encoder_version == "ViT-B/32":
        model, vis_processors, txt_processors = load_model_and_preprocess(name="clip_feature_extractor", 
                                                                          model_type="ViT-B-32", 
                                                                          is_eval=True, 
                                                                          device=device)   
    else:
        raise("Choose one of the available encoders")


    print("Model loaded")
    model.to(device)
    model.eval()

    if "VERITE" in data_name_X:
        verite_df = pd.read_csv(data_path + '/VERITE_with_evidence.csv', index_col=0)
        verite_df = str_to_list(verite_df, list_columns=['captions', 'images_paths'])
        verite_df['match_index'] = verite_df.index.astype(str).tolist()        
        all_X_captions = idx_captions(verite_df, 'VERITE')
        all_X_images = idx_images(verite_df, 'VERITE')

        missing_images = all_X_images[all_X_images.X_image_path =='']
        all_X_images = all_X_images[~all_X_images.X_item_index.isin(missing_images.X_item_index)]
        
    else:
        # Load and process data
        print("Load data")
        vn_data = load_visual_news(data_path)

        train_data = pd.read_csv(data_path + 'news_clippings/merged_balanced_train.csv', index_col=0)
        valid_data = pd.read_csv(data_path + 'news_clippings/merged_balanced_valid.csv', index_col=0)
        test_data = pd.read_csv(data_path + 'news_clippings/merged_balanced_test.csv', index_col=0)

        train_data = train_data.merge(vn_data[['id', 'caption']])
        valid_data = valid_data.merge(vn_data[['id', 'caption']])
        test_data = test_data.merge(vn_data[['id', 'caption']])

        train_data = train_data.merge(vn_data[['image_id', 'image_path']])
        valid_data = valid_data.merge(vn_data[['image_id', 'image_path']])
        test_data = test_data.merge(vn_data[['image_id', 'image_path']])

        del vn_data

        train_data = str_to_list(train_data)
        valid_data = str_to_list(valid_data)
        test_data = str_to_list(test_data)

        train_data.match_index = 'train_' + train_data.match_index.astype(str)
        valid_data.match_index = 'valid_' + valid_data.match_index.astype(str)
        test_data.match_index = 'test_' + test_data.match_index.astype(str)

        train_X_captions = idx_captions(train_data, 'train')
        valid_X_captions = idx_captions(valid_data, 'valid')
        test_X_captions = idx_captions(test_data, 'test')

        train_X_images = idx_images(train_data, 'train')
        valid_X_images = idx_images(valid_data, 'valid')
        test_X_images = idx_images(test_data, 'test')

        all_X_captions = pd.concat([train_X_captions, valid_X_captions, test_X_captions])
        all_X_images = pd.concat([train_X_images, valid_X_images, test_X_images])
        missing_images = all_X_images[all_X_images.X_image_path =='']
        all_X_images = all_X_images[~all_X_images.X_item_index.isin(missing_images.X_item_index)]
    
    img_dataloader, txt_dataloader = prepare_source_dataloaders(all_X_images, 
                                        all_X_captions, 
                                        vis_processors, 
                                        txt_processors,  
                                        batch_size, 
                                        num_workders, 
                                        shuffle)

    text_ids, image_ids, all_text_features, all_visual_features = [], [], [], []
    
    if not os.path.isdir(output_path + 'temp_visual_features'):
        os.makedirs(output_path + 'temp_visual_features')

    print("Extract features from image evidence. Save the batch as a numpy file")
    batch_count = 0
    with torch.no_grad():

        for idx, img in tqdm(img_dataloader):

            sample = {
                "image": img.to(device)
            }

            image_features = model.extract_features(sample)                
            image_features = image_features.reshape(image_features.shape[0], -1).cpu().detach().numpy()
            np.save(output_path + 'temp_visual_features/' + data_name_X + '_' + encoder.lower() + '_' + str(batch_count), image_features) 

            batch_count += 1
            image_ids.extend(idx)

            del sample
            del image_features
            del img

    print("Save: ", output_path)
    image_ids = np.stack(image_ids)
    np.save(output_path + data_name_X + "_image_ids_" + encoder_version +".npy", image_ids)    
    
    print("Load visual features (numpy files) and concatenate into a single file")
    image_embeddings = []
    for batch_count in range(img_dataloader.__len__()):

        print(batch_count, end='\r')
        image_features = np.load(output_path + 'temp_visual_features/' + data_name_X + '_' + encoder.lower() + '_' + str(batch_count) + '.npy') 
        image_embeddings.extend(image_features)
    
    image_embeddings = np.array(image_embeddings)
    image_ids = np.load(output_path + data_name_X + "_image_ids_" + encoder_version +".npy")
    np.save(output_path + data_name_X + '_' + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy", image_embeddings) 
    
    print("Extract features from text evidence")
    with torch.no_grad():
        for idx, txt in tqdm(txt_dataloader):

            sample = {
                "text_input": txt
            }

            clip_features = model.extract_features(sample)
            text_features = clip_features
            text_features = text_features.reshape(text_features.shape[0], -1)
            all_text_features.extend(text_features.cpu().detach().numpy())
            text_ids.extend(idx)
        
    all_text_features = np.stack(all_text_features)
    
    np.save(output_path + data_name_X + '_' + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy", all_text_features)    
    text_ids = np.stack(text_ids)
    np.save(output_path + data_name_X + "_text_ids_" + encoder_version +".npy", text_ids)
    
    
def calc_sim(q_emb, X_items):
    
    cos_sim = cosine_similarity(q_emb.reshape(1, -1), X_items) # Calculate the cosine similarity
    cos_sim = pd.Series(cos_sim[0], index=X_items.index) # Convert the cosine similarities to a pandas Series
    cos_sim = cos_sim.sort_values(ascending=False) # Sort the cosine similarities in descending order   

    return cos_sim


def rank_X_items(input_df, image_cols, text_cols, X_image_embeddings, X_text_embeddings, image_embeddings, text_embeddings):

    img_most_similar_items = []
    txt_most_similar_items = []
    
    for (row) in tqdm(input_df.itertuples(), total=input_df.shape[0]):
              
        match_index_img = [string for string in image_cols if string.startswith(row.match_index + '_')]
        match_index_txt = [string for string in text_cols if string.startswith(row.match_index + '_')]
        
        if match_index_img != []:
            img_items = X_image_embeddings[match_index_img].T
            q_img_emb = image_embeddings[row.image_id].values
            img_cos_sim = calc_sim(q_img_emb, img_items)
            img_most_similar_items.append(
                {
                    'img_ranked_items': img_cos_sim.index.tolist(), 
                    'img_sim_scores': img_cos_sim.values.tolist()
                }
            )
            
        else:
            img_most_similar_items.append(
                {
                    'img_ranked_items': [], 
                    'img_sim_scores': []
                }
            )
                        
        if match_index_txt != []:
                       
            txt_items = X_text_embeddings[match_index_txt].T
            q_txt_emb = text_embeddings[row.image_id].values
            txt_cos_sim = calc_sim(q_txt_emb, txt_items)
            txt_most_similar_items.append(
                {
                    'txt_ranked_items': txt_cos_sim.index.tolist(), 
                    'txt_sim_scores': txt_cos_sim.values.tolist()
                }
            )
        else:
            txt_most_similar_items.append(
                {
                    'txt_ranked_items': [], 
                    'txt_sim_scores': []
                }
            )           
            
    return pd.DataFrame(img_most_similar_items), pd.DataFrame(txt_most_similar_items)


def build_faiss_index(data):
    index = faiss.IndexFlatL2(data.shape[1])  # L2 distance (Euclidean)
    index.add(data)
    return index

def find_similar_row_indices(query_row, index, k):
    D, I = index.search(query_row.reshape(1, -1), k)
    similar_row_indices = I[0]

    return similar_row_indices

def calculate_most_similar(input_data, split_name, image_embeddings, text_embeddings, evidence_path, dataset_name, encoder, encoder_version, K=12):
    
    index_images = build_faiss_index(image_embeddings[input_data.image_id].values.T)
    index_texts = build_faiss_index(text_embeddings[input_data.id].values.T)
    
    all_similar_items = []

    for i, sample in tqdm(input_data.iterrows(), total=input_data.shape[0]):    

        similar_rows_images = find_similar_row_indices(image_embeddings[sample.image_id].values, index_images, K)    
        similar_rows_texts = find_similar_row_indices(text_embeddings[sample.id].values, index_texts, K)    
        
        keep_ids = []
        keep_image_ids = []
        
        for j in similar_rows_images[1:]:
            neg_sample = input_data.iloc[j]
            neg_sample_id = neg_sample.id
            neg_sample_image_id = neg_sample.image_id 
            if neg_sample_id != sample.id and neg_sample_image_id != sample.image_id:
                keep_image_ids.append(neg_sample_image_id)

        for j in similar_rows_texts[1:]:
            neg_sample = input_data.iloc[j]
            neg_sample_id = neg_sample.id
            neg_sample_image_id = neg_sample.image_id 
            if neg_sample_id != sample.id and neg_sample_image_id != sample.image_id:
                keep_ids.append(neg_sample_id)                
                
        all_similar_items.append({
            'id': sample.id,
            'image_id': sample.image_id,        
            'most_similar_id': keep_ids,
            'most_similar_image_id': keep_image_ids,        
        })

        out_path = evidence_path + dataset_name + "_most_similar_items_" + split_name + "_" + encoder.replace("-", "").replace("/", "").lower() + "_" + encoder_version.replace("-", "").replace("/", "").lower()
        pd.DataFrame(all_similar_items).to_csv(out_path + ".csv")


def prepare_negative_evidence(input_data, split_name, evidence_path, dataset_name, encoder, encoder_version):

    all_negative_texts = []
    all_negative_images = []

    for (i, row) in tqdm(input_data.iterrows(), total=input_data.shape[0]):

        img_match_index = None
        txt_match_index = None

        negative_texts = []
        negative_images = []

        most_similar_ids = literal_eval(row.most_similar_id)
        most_similar_image_ids = literal_eval(row.most_similar_image_id)    

        for most_similar_id in most_similar_ids[:5]:

            most_similar_id_item = input_data[input_data.id == most_similar_id].iloc[0]
            img_match_index = [x for x in most_similar_id_item.img_ranked_items if x.startswith(most_similar_id_item.match_index + '_')]
            negative_images.extend(img_match_index)

        for most_similar_image_id in most_similar_image_ids[:5]:

            most_similar_image_id_item = input_data[input_data.image_id == most_similar_image_id].iloc[0]
            txt_match_index = [x for x in most_similar_image_id_item.txt_ranked_items if x.startswith(most_similar_image_id_item.match_index + '_')]
            negative_texts.extend(txt_match_index)

        all_negative_texts.append(negative_texts)    
        all_negative_images.append(negative_images)

    out_negative_images_path = evidence_path + dataset_name + "_negative_images_" + split_name + "_" + encoder.replace("-", "").replace("/", "").lower() + "_" + encoder_version.replace("-", "").replace("/", "").lower() + ".json"
    out_negative_texts_path = evidence_path + dataset_name + "_negative_texts_" + split_name + "_" + encoder.replace("-", "").replace("/", "").lower() + "_" + encoder_version.replace("-", "").replace("/", "").lower() + ".json"

    with open(out_negative_images_path, 'w') as file:
        json.dump(all_negative_images, file)

    with open(out_negative_texts_path, 'w') as file:
        json.dump(all_negative_texts, file)
        

def re_rank_evidence(data_path, data_name, data_name_X, output_path, encoder='CLIP', choose_encoder_version='ViT-B/32'):
    print("Load data")
    vn_data = load_visual_news(data_path)

    train_data = pd.read_csv(data_path + 'news_clippings/merged_balanced_train.csv', index_col=0)
    valid_data = pd.read_csv(data_path + 'news_clippings/merged_balanced_valid.csv', index_col=0)
    test_data = pd.read_csv(data_path + 'news_clippings/merged_balanced_test.csv', index_col=0)

    train_data = train_data.merge(vn_data[['id', 'caption']])
    valid_data = valid_data.merge(vn_data[['id', 'caption']])
    test_data = test_data.merge(vn_data[['id', 'caption']])

    train_data = train_data.merge(vn_data[['image_id', 'image_path']])
    valid_data = valid_data.merge(vn_data[['image_id', 'image_path']])
    test_data = test_data.merge(vn_data[['image_id', 'image_path']])

    train_data[['id', 'image_id', 'match_index']] = train_data[['id', 'image_id', 'match_index']].astype('str')
    valid_data[['id', 'image_id', 'match_index']] = valid_data[['id', 'image_id', 'match_index']].astype('str')
    test_data[['id', 'image_id', 'match_index']] = test_data[['id', 'image_id', 'match_index']].astype('str')    
    
    train_data.match_index = "train_" + train_data.match_index
    valid_data.match_index = "valid_" + valid_data.match_index    
    test_data.match_index = "test_" + test_data.match_index  
    
    encoder_version = choose_encoder_version.replace('-', '').replace('/', '')

    X_image_embeddings = np.load(output_path + data_name_X + '_' + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy")
    X_image_ids = np.load(output_path + data_name_X + "_image_ids_" + encoder_version +".npy")
    X_image_embeddings = pd.DataFrame(X_image_embeddings, index=X_image_ids).T
    X_image_embeddings.columns = X_image_embeddings.columns.astype('str')

    X_text_embeddings = np.load(output_path + data_name_X + '_' + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy")
    X_text_ids = np.load(output_path + data_name_X + "_text_ids_" + encoder_version +".npy")

    X_text_embeddings = pd.DataFrame(X_text_embeddings, index=X_text_ids).T
    X_text_embeddings.columns = X_text_embeddings.columns.astype('str')  
    X_text_embeddings = X_text_embeddings.loc[:, ~X_text_embeddings.columns.duplicated()]

    all_ids = np.concatenate([train_data.id, valid_data.id, test_data.id])
    all_image_ids = np.concatenate([train_data.image_id, valid_data.image_id, test_data.image_id])
    keep_ids = np.unique(np.concatenate([all_ids, all_image_ids]))

    image_embeddings, text_embeddings = load_features(data_path, data_name, encoder, encoder_version, keep_ids)
    
    image_cols = X_image_embeddings.columns.tolist()
    text_cols = X_text_embeddings.columns.tolist()
    
    valid_ranked_X_img, valid_ranked_X_txt = rank_X_items(valid_data, 
                                                          image_cols, 
                                                          text_cols,
                                                          X_image_embeddings, 
                                                          X_text_embeddings,
                                                          image_embeddings,
                                                          text_embeddings
                                                         )
    valid_data = pd.concat([valid_data, valid_ranked_X_img], axis=1)
    valid_data = pd.concat([valid_data, valid_ranked_X_txt], axis=1)

    valid_data.to_csv(data_path + 'news_clippings/merged_balanced_valid_ranked_' + encoder.lower() + "_" + encoder_version + '.csv')
    
    test_ranked_X_img, test_ranked_X_txt = rank_X_items(test_data, 
                                                        image_cols, 
                                                        text_cols, 
                                                        X_image_embeddings, 
                                                        X_text_embeddings,
                                                        image_embeddings,
                                                        text_embeddings                                                       
                                                       )
    test_data = pd.concat([test_data, test_ranked_X_img], axis=1)
    test_data = pd.concat([test_data, test_ranked_X_txt], axis=1)
    test_data.to_csv(data_path + 'news_clippings/merged_balanced_test_ranked_' + encoder.lower() + "_" + encoder_version + '.csv')
    
    train_ranked_X_img, train_ranked_X_txt = rank_X_items(train_data, 
                                                          image_cols, 
                                                          text_cols, 
                                                          X_image_embeddings, 
                                                          X_text_embeddings,
                                                          image_embeddings,
                                                          text_embeddings
                                                         )
    train_data = pd.concat([train_data, train_ranked_X_img], axis=1)
    train_data = pd.concat([train_data, train_ranked_X_txt], axis=1)
    train_data.to_csv(data_path + 'news_clippings/merged_balanced_train_ranked_' + encoder.lower() + "_" + encoder_version + '.csv')  

    calculate_most_similar(train_data, "train", image_embeddings, text_embeddings, output_path, data_name, encoder, encoder_version,K=12)
    calculate_most_similar(valid_data, "valid", image_embeddings, text_embeddings, output_path, data_name, encoder, encoder_version,K=12)
    calculate_most_similar(test_data, "test", image_embeddings, text_embeddings, output_path, data_name, encoder, encoder_version,K=12)      

    train_data[["most_similar_id", "most_similar_image_id"]] = pd.read_csv(output_path + data_name + "_most_similar_items_" + "train" + "_" + encoder.replace("-", "").replace("/", "").lower() + "_" + encoder_version.replace("-", "").replace("/", "").lower()
    + ".csv", index_col=0)[["most_similar_id", "most_similar_image_id"]]

    valid_data[["most_similar_id", "most_similar_image_id"]] = pd.read_csv(output_path + data_name + "_most_similar_items_" + "valid" + "_" + encoder.replace("-", "").replace("/", "").lower() + "_" + encoder_version.replace("-", "").replace("/", "").lower()
    + ".csv", index_col=0)[["most_similar_id", "most_similar_image_id"]]

    test_data[["most_similar_id", "most_similar_image_id"]] = pd.read_csv(output_path + data_name + "_most_similar_items_" + "test" + "_" + encoder.replace("-", "").replace("/", "").lower() + "_" + encoder_version.replace("-", "").replace("/", "").lower()
    + ".csv", index_col=0)[["most_similar_id", "most_similar_image_id"]]

    prepare_negative_evidence(train_data, "train", output_path, data_name, encoder, encoder_version)
    prepare_negative_evidence(valid_data, "valid", output_path, data_name, encoder, encoder_version)
    prepare_negative_evidence(test_data, "test", output_path, data_name, encoder, encoder_version)
    
def re_rank_verite(data_path, data_name, output_path, encoder='CLIP', choose_encoder_version='ViT-B/32'):
    
    data = pd.read_csv(data_path + 'VERITE_with_evidence.csv', index_col=0)
    data['match_index'] = data.index.astype(str).tolist()            
    data["id"] = data.index.astype(str).tolist()
    data["image_id"] = data.index.astype(str).tolist()   
    encoder_version = choose_encoder_version.replace('-', '').replace('/', '')
    
    verite_text_embeddings = np.load(data_path + "VERITE_" + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy").astype('float32')
    verite_image_embeddings = np.load(data_path + "VERITE_" + encoder.lower() +"_image_embeddings_" + encoder_version + ".npy").astype('float32')

    verite_image_embeddings = pd.DataFrame(verite_image_embeddings, index=[str(x) for x in range(1001)]).T
    verite_text_embeddings = pd.DataFrame(verite_text_embeddings, index=[str(x) for x in range(1001)]).T
    
    X_image_embeddings = np.load(output_path + data_name + '_external_' + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy")
    X_image_ids = np.load(output_path + data_name + "_external_image_ids_" + encoder_version +".npy")
    X_image_embeddings = pd.DataFrame(X_image_embeddings, index=X_image_ids).T
    X_image_embeddings.columns = X_image_embeddings.columns.astype('str')

    X_text_embeddings = np.load(output_path + data_name + '_external_' + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy")
    X_text_ids = np.load(output_path + data_name + "_external_text_ids_" + encoder_version +".npy")
    X_text_embeddings = pd.DataFrame(X_text_embeddings, index=X_text_ids).T
    X_text_embeddings.columns = X_text_embeddings.columns.astype('str')
    
    data = str_to_list(data, list_columns=['captions', 'images_paths'])
    
    image_cols = X_image_embeddings.columns.tolist()
    text_cols = X_text_embeddings.columns.tolist()

    ranked_X_img, ranked_X_txt = rank_X_items(data, 
                                              image_cols, 
                                              text_cols, 
                                              X_image_embeddings, 
                                              X_text_embeddings, 
                                              verite_image_embeddings, 
                                              verite_text_embeddings)

    ranked_data = pd.concat([data, ranked_X_img], axis=1)
    ranked_data = pd.concat([ranked_data, ranked_X_txt], axis=1)  

    ranked_data.to_csv(data_path + "VERITE_ranked_evidence_" + encoder.lower() + "_" + encoder_version +  ".csv")