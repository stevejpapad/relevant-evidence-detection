import os
import itertools
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from models import RED_DOT
from utils import (
    early_stop,
    train_step,
    eval_step,
    set_seed,
    eval_verite,
    save_results_csv,
    load_features,
    load_ranked_evidence,
    load_evidence_features,
    load_ranked_verite,
    load_negative_evidence,
    DatasetIterator_negative_Evidence,
    prepare_dataloader_negative_Evidence
)

def run_experiment(RED_DOT_version,
                   use_evidence,
                   use_evidence_neg,
                   k_fold,
                   dataset_name = 'news_clippings_balanced', 
                   dataset_name_X = 'news_clippings_balanced_external_info',
                   data_path = '../data/',
                   verite_path = '../data/VERITE/',
                   evidence_path='../data/news_clippings/',
                   encoder = 'CLIP',
                   encoder_version = 'ViT-B/32',
                   choose_gpu = 0,
                   epochs=100,
                   seed_options = [0],
                   lr_options = [1e-4],
                   batch_size_options = [512],
                   choose_fusion_method = [["concat_1", "add", "sub", "mul"]],
                   tf_layers_options = [4, 6],
                   tf_head_options = [2, 8], 
                   tf_dim_options = [128, 2048]
                  ):
    
    if RED_DOT_version not in ["baseline", "single_stage", "single_stage_guided", "dual_stage", "dual_stage_guided", "dual_stage_two_transformers"]:
        
        raise Exception("Choose one of the available models: baseline, single_stage, single_stage_guided, dual_stage, dual_stage_guided, dual_stage_two_transformers")
        
    if RED_DOT_version in ["single_stage", "single_stage_guided", "dual_stage", "dual_stage_guided", "dual_stage_two_transformers"] and use_evidence_neg <= 0:
         raise Exception("RED-DOT methods, with the exception of Baseline, must have >0 negative evidence! ")

    if RED_DOT_version == "baseline" and use_evidence_neg > 0:
         raise Exception("RED-DOT-Baseline, must have 0 negative evidence!")
        
    zero_pad = False
    if "dual_stage" in RED_DOT_version:
        zero_pad = True

    init_model_name = '_RED_DOT_' + str(use_evidence) + '_' + RED_DOT_version 
    results_filename = "results"    
    token_level = False
    fuse_evidence_options = [["concat_1"]] if use_evidence else [[False]]
    num_workers=8
    early_stop_epochs = 10

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda:" + str(choose_gpu) if torch.cuda.is_available() else "cpu")
    print(device)

    print("Load VERITE")
    verite_test, verite_image_embeddings, verite_text_embeddings, X_verite_image_embeddings, X_verite_text_embeddings = load_ranked_verite(encoder, encoder_version, verite_path)

    print("Load", dataset_name)
    train_data, valid_data, test_data = load_ranked_evidence(encoder, encoder_version, data_path, dataset_name, dataset_name_X)

    print("Load external evidence")
    X_image_embeddings, X_text_embeddings = load_evidence_features(encoder, encoder_version, evidence_path, dataset_name, dataset_name_X)

    all_ids = np.concatenate([train_data.id, valid_data.id, test_data.id])
    all_image_ids = np.concatenate([train_data.image_id, valid_data.image_id, test_data.image_id])
    keep_ids = np.unique(np.concatenate([all_ids, all_image_ids]))

    image_embeddings, text_embeddings = load_features(data_path, dataset_name, encoder, encoder_version, keep_ids)

    train_data = load_negative_evidence(train_data, dataset_name, encoder, encoder_version, "train", evidence_path, use_evidence, use_evidence_neg)
    valid_data = load_negative_evidence(valid_data, dataset_name, encoder, encoder_version, "valid", evidence_path, use_evidence, use_evidence_neg)
    test_data = load_negative_evidence(test_data, dataset_name, encoder, encoder_version, "test", evidence_path, use_evidence, use_evidence_neg)    

    grid = itertools.product(choose_fusion_method, batch_size_options, lr_options, tf_layers_options, tf_head_options, tf_dim_options, fuse_evidence_options, seed_options)        

    experiment = 0
    for params in grid:

        fusion_method, batch_size, lr, tf_layers, tf_head, tf_dim, fuse_evidence, seed = params
        set_seed(seed)
        valid_data_list = []
        final_verite_list = []
        
        if k_fold > 1: 
            kf = KFold(n_splits=k_fold, shuffle=True)

            for fold, (valid_index, test_index) in enumerate(kf.split(verite_test)):    
                valid_data = verite_test.iloc[valid_index]
                final_test_data = verite_test.iloc[test_index]  

                valid_data_list.append(valid_data)
                final_verite_list.append(final_test_data)

        else: 
            valid_data_list = [valid_data]
            final_verite_list = [verite_test]


        for fold in range(k_fold):

            valid_data_fold = valid_data_list[fold]
            verite_data_fold = final_verite_list[fold]

            model_name = dataset_name + '_multimodal_' + str(seed) + init_model_name
            torch.manual_seed(seed)

            print("*****", seed, dataset_name, encoder, encoder_version, model_name, "*****")

            experiment += 1

            parameters = {
                "LEARNING_RATE": lr,
                "EPOCHS": epochs, 
                "BATCH_SIZE": batch_size,
                "TF_LAYERS": tf_layers,
                "TF_HEAD": tf_head,
                "TF_DIM": tf_dim,
                "NUM_WORKERS": 8,
                "USE_FEATURES": ["images", "texts"],
                "EARLY_STOP_EPOCHS": early_stop_epochs,
                "CHOOSE_DATASET": dataset_name,
                "ENCODER": encoder,
                "ENCODER_VERSION": encoder_version,
                "SEED": seed,
                "FUSION_METHOD": fusion_method, 
                "NETWORK_VERSION": RED_DOT_version,
                "TOKEN_LEVEL": token_level,
                "USE_EVIDENCE": use_evidence,
                "USE_NEG_EVIDENCE": use_evidence_neg,
                "FUSE_EVIDENCE": fuse_evidence,
                "k_fold": k_fold,
                "current_fold": fold                 
            }

            train_dataloader = prepare_dataloader_negative_Evidence(
                image_embeddings,
                text_embeddings,
                X_image_embeddings, 
                X_text_embeddings,
                train_data,
                parameters["BATCH_SIZE"],
                parameters["USE_EVIDENCE"],  
                parameters["USE_NEG_EVIDENCE"],   
                fuse_evidence,
                parameters["NUM_WORKERS"],
                True,
                True
            )

            if k_fold == 1:

                valid_dataloader = prepare_dataloader_negative_Evidence(
                    image_embeddings,
                    text_embeddings,
                    X_image_embeddings, 
                    X_text_embeddings,
                    valid_data_fold,           
                    parameters["BATCH_SIZE"],
                    parameters["USE_EVIDENCE"],     
                    parameters["USE_NEG_EVIDENCE"], 
                    fuse_evidence,
                    parameters["NUM_WORKERS"],
                    False,
                    False,
                )

            else:  
                valid_dataloader = DatasetIterator_negative_Evidence(
                    valid_data_fold,
                    visual_features=verite_image_embeddings,
                    textual_features=verite_text_embeddings,
                    X_visual_features=X_verite_image_embeddings,
                    X_textual_features=X_verite_text_embeddings,
                    use_evidence=use_evidence,
                    use_evidence_neg = 0,
                    random_permute=False,
                    fuse_evidence=fuse_evidence
                )

            test_dataloader = prepare_dataloader_negative_Evidence(
                image_embeddings,
                text_embeddings,
                X_image_embeddings, 
                X_text_embeddings,
                test_data,          
                parameters["BATCH_SIZE"],
                parameters["USE_EVIDENCE"],     
                parameters["USE_NEG_EVIDENCE"],   
                fuse_evidence,
                parameters["NUM_WORKERS"],
                False,
                False,
            )

            verite_data_generator = DatasetIterator_negative_Evidence(
                verite_data_fold,
                visual_features=verite_image_embeddings,
                textual_features=verite_text_embeddings,
                X_visual_features=X_verite_image_embeddings,
                X_textual_features=X_verite_text_embeddings,
                use_evidence=use_evidence,
                use_evidence_neg = 0,
                random_permute=False,
                fuse_evidence=fuse_evidence
            )

            print("!!!!!!!!!!!!!!!!!!!", experiment, "!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!", parameters, "!!!!!!!!!!!!!!!!!!!")

            if parameters["ENCODER_VERSION"] == 'ViT-B/32':
                emb_dim_ = 512

            elif parameters["ENCODER_VERSION"] == 'ViT-L/14':    
                emb_dim_ = 768

            parameters["EMB_SIZE"] = emb_dim_             

            model = RED_DOT(
                tf_layers=parameters["TF_LAYERS"],
                tf_head=parameters["TF_HEAD"],
                tf_dim=parameters["TF_DIM"],
                emb_dim=parameters["EMB_SIZE"],
                skip_tokens=len(fusion_method) if "concat_1" not in fusion_method else len(fusion_method) + 1,
                use_evidence=parameters["USE_EVIDENCE"],
                use_neg_evidence=parameters["USE_NEG_EVIDENCE"],
                model_version = RED_DOT_version,
                device=device,
                fuse_evidence=fuse_evidence,
            )

            model.to(device)
            criterion = nn.BCEWithLogitsLoss()
            criterion_mlb = nn.BCEWithLogitsLoss()

            optimizer = torch.optim.Adam(
                model.parameters(), lr=parameters["LEARNING_RATE"]
            )

            batches_per_epoch = train_dataloader.__len__()

            history = []
            has_not_improved_for = 0

            PATH = "checkpoints_pt/model_" + model_name + ".pt"  

            for epoch in range(parameters["EPOCHS"]):

                train_step(
                    model,
                    train_dataloader,
                    encoder, 
                    fusion_method,
                    use_evidence,
                    fuse_evidence,
                    epoch,
                    optimizer,
                    criterion,
                    criterion_mlb,
                    device,
                    batches_per_epoch
                )

                if k_fold > 1:
                    results = eval_verite(model, 
                                          valid_dataloader, 
                                          fusion_method, 
                                          use_evidence, 
                                          fuse_evidence, 
                                          device,
                                          zero_pad=zero_pad
                                          )
                else:
                    results = eval_step(model, 
                                        valid_dataloader, 
                                        encoder, 
                                        fusion_method,
                                        use_evidence,
                                        fuse_evidence,
                                        epoch, 
                                        device
                                    )

                history.append(results)

                has_not_improved_for = early_stop(
                    has_not_improved_for,
                    model,
                    optimizer,
                    history,
                    epoch,
                    PATH,
                    metrics_list=["true_v_ooc"] if k_fold > 1 else ["Accuracy", "exact_match"] if use_evidence_neg > 0 else ["Accuracy"],
                )

                if has_not_improved_for >= parameters["EARLY_STOP_EPOCHS"]:

                    EARLY_STOP_EPOCHS = parameters["EARLY_STOP_EPOCHS"]
                    print(
                        f"Performance has not improved for {EARLY_STOP_EPOCHS} epochs. Stop training at epoch {epoch}!"
                    )
                    break

            print("Finished Training. Loading the best model from checkpoints.")

            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]

            if k_fold == 1:
                res_val = eval_step(model, 
                                    valid_dataloader, 
                                    encoder, 
                                    fusion_method,
                                    use_evidence,
                                    fuse_evidence,
                                    -1, 
                                    device, 
                                    )                

            else:
                res_val = eval_verite(model, 
                                      valid_dataloader, 
                                      fusion_method, 
                                      use_evidence, 
                                      fuse_evidence, 
                                      device,
                                      zero_pad=zero_pad
                                      )

            res_test = eval_step(model, 
                                 test_dataloader, 
                                 encoder, 
                                 fusion_method,
                                 use_evidence,
                                 fuse_evidence,
                                 -2, 
                                 device, 
                                 )

            res_verite = eval_verite(model, 
                                     verite_data_generator, 
                                     fusion_method, 
                                     use_evidence, 
                                     fuse_evidence, 
                                     device,
                                     zero_pad=zero_pad
                                     )

            res_val = {
            "valid_" + str(key.lower()): val for key, val in res_val.items()
            }

            res_test = {
            "test_" + str(key.lower()): val for key, val in res_test.items()
            }

            res_verite = {
            "verite_" + str(key.lower()): val for key, val in res_verite.items()
            }

            all_results = {**res_test, **res_val}
            all_results = {**parameters, **all_results}
            all_results = {**all_results, **res_verite}

            all_results["path"] = PATH
            all_results["history"] = history

            if not os.path.isdir("results"):
                os.mkdir("results")

            save_results_csv(
                "results/",
                results_filename + "_cvood" if k_fold > 1 else "_idv",            
                all_results,
            )
