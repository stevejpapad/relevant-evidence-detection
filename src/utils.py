import os
import json
import time
import torch 
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from ast import literal_eval
from torch.utils.data import DataLoader
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def ensure_directories_exist(*directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        else:
            print(f"Directory '{directory}' already exists.")

def load_training_dataset(data_path, data_name):        
    print("Data path: ", data_path)
    print("Load: ", data_name)
        
    if data_name == 'news_clippings_balanced':

        train_data = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/train.json"))
        valid_data = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/val.json"))
        test_data = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/test.json"))

        train_data = pd.DataFrame(train_data["annotations"])
        valid_data = pd.DataFrame(valid_data["annotations"])
        test_data = pd.DataFrame(test_data["annotations"])
                                   
    else:
        raise ValueError("data_name does not match any available dataset")
                
    train_data.id = train_data.id.astype('str')
    valid_data.id = valid_data.id.astype('str')
    test_data.id = test_data.id.astype('str')
    
    train_data.image_id = train_data.image_id.astype('str')
    valid_data.image_id = valid_data.image_id.astype('str')
    test_data.image_id = test_data.image_id.astype('str')            
            
    return train_data, valid_data, test_data

def load_visual_news(data_path):
    
    vn_data = json.load(open(data_path + 'VisualNews/origin/data.json'))
    vn_data = pd.DataFrame(vn_data)
    vn_data['image_id'] = vn_data['id']
    
    return vn_data

def load_features(data_path, data_name, encoder, encoder_version, filter_ids=[None]):
    
    print("Load features")
    
    encoder_version = encoder_version.replace("-", "").replace("/", "")
    
    ocr_embeddings = None
    
    if data_name == 'news_clippings_balanced':
        image_embeddings = np.load(
            data_path + "news_clippings/" + data_name + "_" + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy"
        ).astype("float32")

        text_embeddings = np.load(
            data_path + "news_clippings/" + data_name + "_" + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy"
        ).astype("float32")

        item_ids = np.load(data_path + "news_clippings/" + data_name + "_item_ids_" + encoder_version + ".npy")
        
        
    image_embeddings = pd.DataFrame(image_embeddings, index=item_ids).T
    text_embeddings = pd.DataFrame(text_embeddings, index=item_ids).T
    
    image_embeddings.columns = image_embeddings.columns.astype('str')
    text_embeddings.columns = text_embeddings.columns.astype('str')  
    
    if len(filter_ids) > 1:
        image_embeddings = image_embeddings[filter_ids]
        text_embeddings = text_embeddings[filter_ids]
        
    return image_embeddings, text_embeddings


def load_negative_evidence(input_data, dataset_name, encoder, encoder_version, split_name, evidence_path, use_evidence, use_evidence_neg, filter_items=True):

    out_negative_images_path = evidence_path + dataset_name + "_negative_images_" + split_name + "_" + encoder.replace("-", "").replace("/", "").lower() + "_" + encoder_version.replace("-", "").replace("/", "").lower() + ".json"
    out_negative_texts_path = evidence_path + dataset_name + "_negative_texts_" + split_name + "_" + encoder.replace("-", "").replace("/", "").lower() + "_" + encoder_version.replace("-", "").replace("/", "").lower() + ".json"

    with open(out_negative_images_path, 'r') as file:
        negative_images = json.load(file)
        
    with open(out_negative_texts_path, 'r') as file:
        negative_texts = json.load(file)     
        
    input_data["negative_images"] = negative_images
    input_data["negative_texts"] = negative_texts
    
    if filter_items:
        input_data = filter_out_items_without_enough_evidence(input_data, use_evidence, use_evidence_neg)

        input_data['negative_images'] = input_data['negative_images'].apply(lambda x: remove_empty_lists(x))    
        input_data['negative_texts'] = input_data['negative_texts'].apply(lambda x: remove_empty_lists(x))

    return input_data

class DatasetIterator_negative_Evidence(torch.utils.data.Dataset):
    
    def __init__(
        self,
        input_data,
        visual_features,
        textual_features,
        X_visual_features,
        X_textual_features,
        use_evidence,
        use_evidence_neg,
        random_permute,
        fuse_evidence=[False]
    ):
        self.input_data = input_data
        self.visual_features = visual_features
        self.textual_features = textual_features
        self.X_visual_features = X_visual_features
        self.X_textual_features = X_textual_features
        self.use_evidence = use_evidence
        self.use_evidence_neg = use_evidence_neg
        self.random_permute = random_permute
        self.fuse_evidence = fuse_evidence
        
    def __len__(self):
        return self.input_data.shape[0]
        
    def __getitem__(self, idx):
        current = self.input_data.iloc[idx]
        
        img = self.visual_features[current.image_id].values
        txt = self.textual_features[current.id].values
        
        label = float(current.falsified)
                
        if self.use_evidence == 0:
            return img, txt, label, np.nan, np.nan
        
        X_img = self.X_visual_features[current.img_ranked_items[:self.use_evidence]].T.values
        X_txt = self.X_textual_features[current.txt_ranked_items[:self.use_evidence]].T.values
                
        if self.use_evidence_neg > 0:
            
            total_items = 2*self.use_evidence + 2*self.use_evidence_neg
            available_X_img_neg = len(current.negative_images[:self.use_evidence_neg])
            available_X_txt_neg = len(current.negative_texts[:self.use_evidence_neg])
            
            negatives_to_add = total_items - available_X_img_neg - available_X_txt_neg - X_txt.shape[0] - X_img.shape[0]
            
            how_many_neg_imgs = 0
            how_many_neg_txts = 0
            
            extra_random_txts = []
            extra_random_imgs = [] 

            if negatives_to_add > 0:
                max_images = max(0,len(current.negative_images) - self.use_evidence_neg)
                max_texts = max(0,len(current.negative_texts) - self.use_evidence_neg)

                if max_texts == 0 and max_images>=negatives_to_add:
                    how_many_neg_imgs = negatives_to_add

                elif max_images == 0 and max_texts>=negatives_to_add:
                    how_many_neg_txts = negatives_to_add

                else:    
                    if max_images >= negatives_to_add and max_texts >= negatives_to_add:
                        how_many_neg_imgs = random.randint(0, negatives_to_add)
                        how_many_neg_txts = negatives_to_add - how_many_neg_imgs

                    else:
                        if max_images < negatives_to_add:
                            how_many_neg_imgs = max_images
                            how_many_neg_txts = min(max_texts, negatives_to_add - how_many_neg_imgs)

                        elif max_texts < negatives_to_add:
                            how_many_neg_txts = max_texts
                            how_many_neg_imgs = min(max_images, negatives_to_add - how_many_neg_txts)
                
                if how_many_neg_imgs + how_many_neg_txts < negatives_to_add:              
                    for _ in range(negatives_to_add - how_many_neg_imgs - how_many_neg_txts):
                        rand_modality = random.randint(0, 1)
                        
                        current_split = current.match_index.split("_")[0]
                        avoid_id_pattern = current.match_index + "_" 
                        
                        if rand_modality == 0:
                            avoid_columns = [col for col in self.X_visual_features.columns if col.startswith(avoid_id_pattern)]
                            valid_columns = [col for col in self.X_visual_features.columns if col not in avoid_columns and col.startswith(current_split)]
                            selected_item = random.choice(valid_columns)
                            extra_random_imgs.append(selected_item)
                            
                        else:
                            avoid_columns = [col for col in self.X_textual_features.columns if col.startswith(avoid_id_pattern)]
                            valid_columns = [col for col in self.X_textual_features.columns if col not in avoid_columns and col.startswith(current_split)]
                            selected_item = random.choice(valid_columns)                      
                            extra_random_txts.append(selected_item)
                                                                        
                                
            X_img_neg_items = current.negative_images[:self.use_evidence_neg + how_many_neg_imgs] + extra_random_imgs            
            X_img_neg = self.X_visual_features[X_img_neg_items].T.values
        
            X_txt_neg_items = current.negative_texts[:self.use_evidence_neg + how_many_neg_txts] + extra_random_txts            
            X_txt_neg = self.X_textual_features[X_txt_neg_items].T.values
                               
            X_all = np.concatenate([X_img, X_txt, X_img_neg, X_txt_neg])
            pos_labels = np.ones(X_img.shape[0] + X_txt.shape[0])
            neg_labels = np.zeros(X_img_neg.shape[0] + X_txt_neg.shape[0])
            X_all_labels = np.concatenate([pos_labels, neg_labels])

            if self.random_permute:
                random_indices = np.random.permutation(X_all.shape[0])
            else:
                # Used for comparable evaluation
                random_indices = [x for x in range(X_all.shape[0]-1, -1, -1)]
                    
            X_all = X_all[random_indices]
            X_all_labels = X_all_labels[random_indices]
        
        else:
                        
            if X_img.shape[0] < self.use_evidence:            
                pad_zeros = np.zeros((self.use_evidence - X_img.shape[0], self.X_visual_features.shape[0]))
                X_img = np.vstack([X_img, pad_zeros])

            if X_txt.shape[0] < self.use_evidence:
                pad_zeros = np.zeros((self.use_evidence - X_txt.shape[0], self.X_textual_features.shape[0]))
                X_txt = np.vstack([X_txt, pad_zeros])          
                
            X_all = np.concatenate([X_img, X_txt])
            X_all_labels = np.ones(X_img.shape[0] + X_txt.shape[0])
            
        return img, txt, label, X_all.astype("float32"), X_all_labels.astype("float32")
    
def prepare_dataloader_negative_Evidence(image_embeddings, text_embeddings, X_image_embeddings, X_text_embeddings, input_data, batch_size, use_evidence, use_evidence_neg, fuse_evidence, num_workers, shuffle, random_permute):
    
    dg = DatasetIterator_negative_Evidence(
        input_data,
        visual_features=image_embeddings,
        textual_features=text_embeddings,
        X_visual_features=X_image_embeddings,
        X_textual_features=X_text_embeddings,
        use_evidence=use_evidence,
        use_evidence_neg = use_evidence_neg,
        random_permute=random_permute,
        fuse_evidence=fuse_evidence
    )

    dataloader = DataLoader(
        dg,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return dataloader
    

def modality_fusion(fusion_method, mod_a, mod_b):
    
    x = [None]
                    
    if "concat_1" in fusion_method:    
        
        if mod_a.dim() == 2:
            mod_a = mod_a.unsqueeze(1)
            
        if mod_b.dim() == 2:
            mod_b = mod_b.unsqueeze(1)
        
        x = torch.cat([mod_a, mod_b], axis=1)
                    
    if 'add' in fusion_method:
        
        added = torch.add(mod_a, mod_b)
        x = torch.cat([x, added], axis=1) if x.dim() > 1 else added
        
    if 'mul' in fusion_method:
        
        mult = torch.mul(mod_a, mod_b)
        x = torch.cat([x, mult], axis=1) if x.dim() > 1 else mult
        
    if 'sub' in fusion_method:
        
        sub = torch.sub(mod_a, mod_b)
        x = torch.cat([x, sub], axis=1) if x.dim() > 1 else sub
                                          
    return x

def prepare_input(fusion_method, fuse_evidence, use_evidence, images, texts, X_images, X_texts, X_all=None):
    
    if fusion_method:
        x = modality_fusion(fusion_method, 
                            mod_a=images, 
                            mod_b=texts)

    if use_evidence:
        
        if X_all != None:
            
            x = torch.cat([x, X_all], axis=1)
            return x

        if len(fuse_evidence) > 1:

            img2Ximg = modality_fusion(fuse_evidence, 
                                       images if images.dim() > 2 else images.unsqueeze(1), 
                                       X_images)
            txt2Xtxt = modality_fusion(fuse_evidence, 
                                       texts if texts.dim() > 2  else texts.unsqueeze(1), 
                                       X_texts)    

            if "concat_1" in fuse_evidence:
                img2Ximg = img2Ximg[:, 1:, :]
                txt2Xtxt = txt2Xtxt[:, 1:, :]    

        else:
            img2Ximg = X_images 
            txt2Xtxt = X_texts

        if fusion_method:
            x = torch.cat([x, img2Ximg, txt2Xtxt], axis=1)            

        else:
            x = modality_fusion(fuse_evidence, 
                                mod_a=X_images, 
                                mod_b=X_texts)
            
    return x


def check_C(C, pos):
    
    if C == 0:
        return np.zeros(pos.shape[0])    
    else: 
        return np.ones(pos.shape[0])
        
        
def sensitivity_per_class(y_true, y_pred, C):
    
    pos = np.where(y_true == C)[0]
    y_true = y_true[pos]
    y_pred = y_pred[pos]
    
    if C == 2:
        y_true = np.ones(y_true.shape[0]).reshape(-1, 1)
    
    return round((y_pred == y_true).sum() / y_true.shape[0], 4)

def accuracy_CvC(y_true, y_pred, Ca, Cb):
    pos_a = np.where(y_true == Ca)[0]
    pos_b = np.where(y_true == Cb)[0]

    y_pred_a = y_pred[pos_a].flatten()
    y_pred_b = y_pred[pos_b].flatten()   
    
    y_true_a = check_C(Ca, pos_a)
    y_true_b = check_C(Cb, pos_b)
    
    y_pred_avb = np.concatenate([y_pred_a, y_pred_b])
    y_true_avb = np.concatenate([y_true_a, y_true_b])
    
    return round(metrics.accuracy_score(y_true_avb, y_pred_avb), 4)

def eval_verite(model, verite_data_generator, fusion_method, use_evidence, fuse_evidence, device, zero_pad=False, label_map={'true': 0, 'miscaptioned': 1, 'out-of-context': 2}, cur_epoch=-3):
    
    print("\nEvaluation on VERITE")
    model.eval()

    y_true = []
    y_pred = []

    y_pred_X_labels = []
    y_true_X_labels = []

    total_tokens = total_tokens = len(fusion_method) + 4 * use_evidence * len(fuse_evidence) + (1 if "concat_1" in fusion_method else 0)

    with torch.no_grad():

        for i, data in enumerate(verite_data_generator, 0):

            images = torch.tensor(data[0]).to(device, non_blocking=True).unsqueeze(0)
            texts = torch.tensor(data[1]).to(device, non_blocking=True).unsqueeze(0)
            labels = torch.tensor(data[2]).to(device, non_blocking=True)
            X_all = torch.tensor(data[3]).to(device, non_blocking=True).unsqueeze(0)            
            x = prepare_input(fusion_method, fuse_evidence, use_evidence, images, texts, None, None, X_all)

            if x.shape[1] < total_tokens and zero_pad:
                pad_zeros = torch.zeros((x.shape[0], total_tokens - x.shape[1], x.shape[-1])).to(device)
                x = torch.concat([x, pad_zeros], axis=1)

            predictions = model(x, inference=True)

            y_pred.append(predictions[0].item())
            y_true.append(labels.item())

    y_true = np.array(y_true)    
    y_pred = np.array(y_pred)
    y_pred = 1/(1 + np.exp(-y_pred))
    y_pred = y_pred.round()

    verite_results = {}

    verite_results['true'] = sensitivity_per_class(y_true, y_pred, 0)
    verite_results['miscaptioned'] = sensitivity_per_class(y_true, y_pred, 1)
    verite_results['out_of_context'] = sensitivity_per_class(y_true, y_pred, 2)

    verite_results['true_v_miscaptioned'] = accuracy_CvC(y_true, y_pred, 0, 1)
    verite_results['true_v_ooc'] = accuracy_CvC(y_true, y_pred, 0, 2)
    verite_results['miscaptioned_v_ooc'] = accuracy_CvC(y_true, y_pred, 1, 2)

    y_true_all = y_true.copy()
    y_true_all[np.where(y_true_all == 2)[0]] = 1

    verite_results['accuracy'] = round(metrics.accuracy_score(y_true_all, y_pred), 4)
    verite_results['bal_accuracy'] = round(metrics.balanced_accuracy_score(y_true_all, y_pred), 4)

    print(verite_results)
    return verite_results


def load_verite(data_path, encoder, encoder_version, label_map={'true': 0, 'miscaptioned': 1, 'out-of-context': 2}):

    encoder_version = encoder_version.replace('-', '').replace('/', '')
    
    verite_test = pd.read_csv(data_path + 'VERITE.csv', index_col=0)
    verite_test = verite_test.reset_index().rename({'index': 'id', 'label': 'falsified'}, axis=1)
    verite_test['image_id'] = verite_test['id']

    verite_text_embeddings = np.load(data_path + "VERITE_" + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy").astype('float32')
    verite_image_embeddings = np.load(data_path + "VERITE_" + encoder.lower() +"_image_embeddings_" + encoder_version + ".npy").astype('float32')

    verite_image_embeddings = pd.DataFrame(verite_image_embeddings, index=verite_test.id.values).T
    verite_text_embeddings = pd.DataFrame(verite_text_embeddings, index=verite_test.id.values).T

    verite_test.falsified.replace(label_map, inplace=True)

    return verite_test, verite_image_embeddings, verite_text_embeddings


def train_step(model, input_dataloader, encoder, fusion_method, use_evidence, fuse_evidence, current_epoch, optimizer, criterion, criterion_mlb, device, batches_per_epoch):
    epoch_start_time = time.time()

    running_loss = 0.0
    running_loss_binary = 0.0
    running_loss_mlb = 0.0
    
    model.train()
    
    for i, data in enumerate(input_dataloader, 0):

        images = data[0].to(device, non_blocking=True)
        texts = data[1].to(device, non_blocking=True)
        labels = data[2].to(device, non_blocking=True)
        X_all = data[3].to(device, non_blocking=True)
        X_labels = data[4].to(device, non_blocking=True)
        
        x = prepare_input(fusion_method, fuse_evidence, use_evidence, images, texts, None, None, X_all)
        optimizer.zero_grad()
        
        outputs = model(x, False, X_labels)
        y_binary = outputs[0]
        y_relevance = outputs[1]
        
        loss_binary = criterion(
            y_binary, labels.unsqueeze(1)
        )
        
        if model.model_version == "baseline":
            loss = loss_binary            

        else:
            loss_mlb = criterion_mlb(
                y_relevance, X_labels
            )            
            loss = loss_mlb + loss_binary            
            running_loss_mlb += loss_mlb.item()   
                    
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_binary += loss_binary.item()


        print(
            f"[Epoch:{current_epoch + 1}, Batch:{i + 1:5d}/{batches_per_epoch}]. Passed time: {round((time.time() - epoch_start_time) / 60, 1)} minutes. loss: {running_loss / (i+1):.3f}. Binary loss: {round(running_loss_binary / (i+1), 3)}. Relevance loss: {round(running_loss_mlb / (i+1), 3)}",
            end="\r",
        )          
        

def eval_step(model, input_dataloader, encoder, fusion_method, use_evidence, fuse_evidence, current_epoch, device, calculate_mlb=True, return_results=True):
    
    if current_epoch >= 0:
        print("\nEvaluation:", end=" -> ")
    elif current_epoch == -1:
        print("\nFinal evaluation on the VALIDATION set", end=" -> ")
    else:
        print("\nFinal evaluation on the TESTING set", end=" -> ")
        
    model.eval()

    y_true = []
    y_pred = []
    
    y_pred_X_labels = []
    y_true_X_labels = []

    with torch.no_grad():
        
        for i, data in enumerate(input_dataloader, 0):

            images = data[0].to(device, non_blocking=True)
            texts = data[1].to(device, non_blocking=True)
            labels = data[2].to(device, non_blocking=True)
            X_all = data[3].to(device, non_blocking=True)            
            X_labels = data[4].to(device, non_blocking=True)            

            x = prepare_input(fusion_method, fuse_evidence, use_evidence, images, texts, None, None, X_all)
            predictions = model(x, False, X_labels)
            
            y_pred.extend(predictions[0].cpu().detach().numpy())
            y_true.extend(labels.cpu().detach().numpy())
            
            if model.model_version != "baseline":
                y_pred_X_labels.extend(predictions[1].cpu().detach().numpy())
                y_true_X_labels.extend(X_labels.cpu().detach().numpy())

    y_pred = np.vstack(y_pred)
    y_pred = 1/(1 + np.exp(-y_pred))
    y_true = np.array(y_true).reshape(-1,1)
    
    if not return_results:
        return y_true, y_pred
    
    auc = metrics.roc_auc_score(y_true, y_pred)
    y_pred = np.round(y_pred)        
    acc = metrics.accuracy_score(y_true, y_pred)    
    prec = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred) 
    f1 = metrics.f1_score(y_true, y_pred)
    cm = metrics.confusion_matrix(y_true, y_pred, normalize="true").diagonal()

    if model.model_version != "baseline":
        y_pred_X_labels = np.stack(y_pred_X_labels)
        y_pred_X_labels = 1/(1 + np.exp(-y_pred_X_labels))
        y_pred_X_labels = y_pred_X_labels.round()
        y_true_X_labels = np.stack(y_true_X_labels)
        hl = metrics.hamming_loss(y_true_X_labels, y_pred_X_labels)
        exact_match = metrics.accuracy_score(y_true_X_labels, y_pred_X_labels)

        macro_f1_mlb = metrics.f1_score(y_true_X_labels, y_pred_X_labels, average='macro')
        micro_ap_mlb = metrics.average_precision_score(y_true_X_labels, y_pred_X_labels, average='micro')        
    else:
        hl = 0.0
        exact_match = 0.0
        macro_f1_mlb = 0.0
        micro_ap_mlb = 0.0        

    results = {
        "epoch": current_epoch,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        'Pristine': round(cm[0], 4),
        'Falsified': round(cm[1], 4),
        'hamming_loss': round(hl, 4),
        'exact_match': round(exact_match, 4),
        'macro_f1_mlb': round(macro_f1_mlb, 4),
        'micro_average_precision': round(micro_ap_mlb, 4)

    }
    print(results)
    
    return results

def topsis(xM, wV=None):
    m, n = xM.shape

    if wV is None:
        wV = np.ones((1, n)) / n
    else:
        wV = wV / np.sum(wV)

    normal = np.sqrt(np.sum(xM**2, axis=0))

    rM = xM / normal
    tM = rM * wV
    twV = np.max(tM, axis=0)
    tbV = np.min(tM, axis=0)
    dwV = np.sqrt(np.sum((tM - twV) ** 2, axis=1))
    dbV = np.sqrt(np.sum((tM - tbV) ** 2, axis=1))
    swV = dwV / (dwV + dbV)

    arg_sw = np.argsort(swV)[::-1]

    r_sw = swV[arg_sw]

    return np.argsort(swV)[::-1]

def choose_best_model(input_df, metrics, epsilon=1e-6):

    X0 = input_df.copy()
    X0 = X0.reset_index(drop=True)
    X1 = X0[metrics]
    X1 = X1.reset_index(drop=True)
    
    # Stop if the scores are identical in all consecutive epochs
    X1[:-1] = X1[:-1] + epsilon

    if "Accuracy" in metrics:
        X1["Accuracy"] = 1 - X1["Accuracy"]    

    if "Precision" in metrics:
        X1["Precision"] = 1 - X1["Precision"]    

    if "Recall" in metrics:
        X1["Recall"] = 1 - X1["Recall"]          
        
    if "AUC" in metrics:
        X1["AUC"] = 1 - X1["AUC"]
        
    if "F1" in metrics:
        X1["F1"] = 1 - X1["F1"]

    if "Pristine" in metrics:
        X1["Pristine"] = 1 - X1["Pristine"]
        
    if "Falsified" in metrics:
        X1["Falsified"] = 1 - X1["Falsified"]
        
    if "exact_match" in metrics:
        X1["exact_match"] = 1 - X1["exact_match"]
        
    if "true_v_ooc" in metrics:
        X1["true_v_ooc"] = 1 - X1["true_v_ooc"]
        
    X_np = X1.to_numpy()
    best_results = topsis(X_np)
    top_K_results = best_results[:1]
    return X0.iloc[top_K_results]

def early_stop(has_not_improved_for, model, optimizer, history, current_epoch, PATH, metrics_list):

    best_index = choose_best_model(
        pd.DataFrame(history), metrics=metrics_list
    ).index[0]
        
    if not os.path.isdir(PATH.split('/')[0]):
        os.mkdir(PATH.split('/')[0])

    if current_epoch == best_index:
        
        print("Checkpoint!\n")
        torch.save(
            {
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            PATH,
        )

        has_not_improved_for = 0
    else:
        
        print("DID NOT CHECKPOINT!\n")
        has_not_improved_for += 1
            
    return has_not_improved_for

def save_results_csv(output_folder_, output_file_, model_performance_):
    print("Save Results ", end=" ... ")
    exp_results_pd = pd.DataFrame(pd.Series(model_performance_)).transpose()
    if not os.path.isfile(output_folder_ + "/" + output_file_ + ".csv"):
        exp_results_pd.to_csv(
            output_folder_ + "/" + output_file_ + ".csv",
            header=True,
            index=False,
            columns=list(model_performance_.keys()),
        )
    else:
        exp_results_pd.to_csv(
            output_folder_ + "/" + output_file_ + ".csv",
            mode="a",
            header=False,
            index=False,
            columns=list(model_performance_.keys()),
        )
    print("Done\n")


def str_to_list(df, list_columns):
  
    df[list_columns] = df[list_columns].fillna('[]')

    for column in list_columns:
        df[column] = df[column].apply(literal_eval)
        
    return df

def filter_out_items_without_enough_evidence(input_data, use_evidence, use_evidence_neg):

    input_data["len_negative_images"] = input_data['negative_images'].apply(lambda x: len(x))
    input_data["len_negative_texts"] = input_data['negative_texts'].apply(lambda x: len(x))

    input_data["len_X_texts"] = input_data['txt_ranked_items'].apply(lambda x: len(x))
    input_data["len_X_images"] = input_data['img_ranked_items'].apply(lambda x: len(x))

    input_data['total'] = input_data['len_negative_images'] + input_data['len_negative_texts'] + input_data['len_X_texts'] + input_data['len_X_images']
    
    return input_data[input_data.total >= 2 * (use_evidence + use_evidence_neg)]

def remove_empty_lists(lst):
    return [item for item in lst if item]

def load_ranked_evidence(encoder, choose_encoder_version, data_path, data_name, data_name_X):

    encoder_version = choose_encoder_version.replace("-", "").replace("/", "")
    train_data_ranked = pd.read_csv(data_path + 'news_clippings/merged_balanced_train_ranked_' + encoder.lower() + "_" + encoder_version + '.csv', index_col=0)
    valid_data_ranked = pd.read_csv(data_path + 'news_clippings/merged_balanced_valid_ranked_' + encoder.lower() + "_" + encoder_version + '.csv', index_col=0)
    test_data_ranked = pd.read_csv(data_path + 'news_clippings/merged_balanced_test_ranked_' + encoder.lower() + "_" + encoder_version + '.csv', index_col=0)

    list_cols = ['entities', 'q_detected_labels', 'captions', 'titles', 'images_paths', 'images_labels', 'img_ranked_items', 'img_sim_scores','txt_ranked_items', 'txt_sim_scores']  
    train_data_ranked = str_to_list(train_data_ranked, list_cols)
    valid_data_ranked = str_to_list(valid_data_ranked, list_cols)
    test_data_ranked = str_to_list(test_data_ranked, list_cols)
    
    train_data_ranked[['id', 'image_id']] = train_data_ranked[['id', 'image_id']].astype('str')
    valid_data_ranked[['id', 'image_id']] = valid_data_ranked[['id', 'image_id']].astype('str')
    test_data_ranked[['id', 'image_id']] = test_data_ranked[['id', 'image_id']].astype('str') 
                    
    return train_data_ranked, valid_data_ranked, test_data_ranked

def load_evidence_features(encoder, choose_encoder_version, evidence_path, data_name, data_name_X):
    
    encoder_version = choose_encoder_version.replace('-', '').replace('/', '')

    X_image_embeddings = np.load(evidence_path + data_name_X + '_' + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy")
    X_image_ids = np.load(evidence_path + data_name_X + "_image_ids_" + encoder_version +".npy")
    X_image_embeddings = pd.DataFrame(X_image_embeddings, index=X_image_ids).T
    X_image_embeddings.columns = X_image_embeddings.columns.astype('str')

    X_text_embeddings = np.load(evidence_path + data_name_X + '_' + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy")
    X_text_ids = np.load(evidence_path + data_name_X + "_text_ids_" + encoder_version +".npy")

    X_text_embeddings = pd.DataFrame(X_text_embeddings, index=X_text_ids).T
    X_text_embeddings.columns = X_text_embeddings.columns.astype('str')  
    X_text_embeddings = X_text_embeddings.loc[:, ~X_text_embeddings.columns.duplicated()]
    
    return X_image_embeddings, X_text_embeddings



def load_ranked_verite(encoder, choose_encoder_version, data_path, label_map={'true': 0, 'miscaptioned': 1, 'out-of-context': 2}):
    
    encoder_version = choose_encoder_version.replace('-', '').replace('/', '')
    
    verite_test = pd.read_csv(data_path + "VERITE_ranked_evidence_" + encoder.lower() + "_" + encoder_version +  ".csv", index_col=0)

    verite_test = str_to_list(verite_test, ['captions', 'images_paths', 'img_ranked_items', 'txt_ranked_items'])
    
    verite_test = verite_test.reset_index().rename({'index': 'id', 'label': 'falsified'}, axis=1)
    verite_test['image_id'] = verite_test['id']

    verite_text_embeddings = np.load(data_path + "VERITE_" + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy").astype('float32')
    verite_image_embeddings = np.load(data_path + "VERITE_" + encoder.lower() +"_image_embeddings_" + encoder_version + ".npy").astype('float32')
    verite_image_embeddings = pd.DataFrame(verite_image_embeddings, index=verite_test.id.values).T
    verite_text_embeddings = pd.DataFrame(verite_text_embeddings, index=verite_test.id.values).T

    verite_test.falsified.replace(label_map, inplace=True)

    data_name = 'VERITE'
    X_verite_image_embeddings = np.load(data_path + data_name + '_external_' + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy")
    X_verite_image_ids = np.load(data_path + data_name + "_external_image_ids_" + encoder_version +".npy")
    X_verite_image_embeddings = pd.DataFrame(X_verite_image_embeddings, index=X_verite_image_ids).T
    X_verite_image_embeddings.columns = X_verite_image_embeddings.columns.astype('str')

    X_verite_text_embeddings = np.load(data_path + data_name + '_external_' + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy")
    X_verite_text_ids = np.load(data_path + data_name + "_external_text_ids_" + encoder_version +".npy")
    X_verite_text_embeddings = pd.DataFrame(X_verite_text_embeddings, index=X_verite_text_ids).T
    X_verite_text_embeddings.columns = X_verite_text_embeddings.columns.astype('str')
    
    return verite_test, verite_image_embeddings, verite_text_embeddings, X_verite_image_embeddings, X_verite_text_embeddings

