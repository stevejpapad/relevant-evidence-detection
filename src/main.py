from utils import ensure_directories_exist
from experiment import run_experiment
from extract_features import extract_encoder_features
from prepare_VERITE import collect_evidence, download_images, save_verite_file 
from prepare_evidence import load_merge_evidence_data, extract_features_for_evidence, re_rank_evidence, re_rank_verite

DATA_PATH = 'data/'
VERITE_PATH = 'data/VERITE/'
EVIDENCE_PATH = 'data/news_clippings/'
SOURCE_EVIDENCE_PATH = 'data/news_clippings/queries_dataset' 
DATA_NAME = 'news_clippings_balanced'
DATA_NAME_X = 'news_clippings_balanced_external_info'

directories_to_check = [DATA_PATH, VERITE_PATH, EVIDENCE_PATH, SOURCE_EVIDENCE_PATH, DATA_PATH+'VisualNews/origin/']
ensure_directories_exist(*directories_to_check)

### The collected evidence are provided. But if you need to re-collect them, uncomment the following:
### It will require the installation of Google API with 'pip install google-api-python-client'
# collect_evidence(data_path = '/VERITE', API_KEY = "YOUR_KEY", CUSTOM_SEARCH_ENGINE_ID = "YOUR_ID")

### Ιf you need to download the VERITE images, uncomment the following: 
# download_images(data_path = '/VERITE')

### The 'VERITE_with_evidence.csv' file is provided. But if you need to re-create it, uncomment the following: 
# save_verite_file(data_path = '/VERITE')

### Extract CLIP features for NewsCLIPings
extract_encoder_features(data_path=DATA_PATH, images_path=DATA_PATH+'VisualNews/origin/', 
                         data_name=DATA_NAME, output_path=EVIDENCE_PATH)
load_merge_evidence_data(SOURCE_EVIDENCE_PATH, data_path, DATA_NAME)

### Extract CLIP features for VERITE
extract_encoder_features(data_path=VERITE_PATH, images_path=VERITE_PATH, data_name='VERITE', output_path=VERITE_PATH)
   
### Extract CLIP features for the external evidence of NewsCLIPings
extract_features_for_evidence(data_path = DATA_PATH, output_path= EVIDENCE_PATH, data_name_X=DATA_NAME_X)

### Evidence re-ranking module for NewsCLIPings
re_rank_evidence(data_path=DATA_PATH, data_name=DATA_NAME, data_name_X=DATA_NAME_X, output_path=EVIDENCE_PATH)

### CLIP features for VERITE are provided. If you want to re-extract them, uncomment the following: 
# extract_features_for_evidence(data_path = VERITE_PATH, output_path=VERITE_PATH, data_name_X="VERITE_external")

### Ranked evidence for VERITE are provided. If you want to re-calculate them, uncomment the following: 
# re_rank_verite(data_path=VERITE_PATH, data_name="VERITE", output_path=DATA_NAME)

## Run experiments for the RED-DOT-Baseline which does not leverage irrelevant evidence
## Results are shown in Tables 1 and 2 
for k_fold in [1, 3]:
    for num_evidence in [0, 1, 2, 4]:
        run_experiment(RED_DOT_version = "baseline", 
                       data_path=DATA_PATH,
                       evidence_path=EVIDENCE_PATH,
                       verite_path=VERITE_PATH,
                       use_evidence=num_evidence, 
                       use_evidence_neg=0, 
                       k_fold=k_fold, 
                       choose_fusion_method = [["concat_1", "add", "sub", "mul"], 
                                               ["concat_1", "add", "sub"],
                                               ["concat_1", "add", "mul"],
                                               ["concat_1", "sub", "mul"],
                                               ["concat_1"]
                                              ])

### Run experiments for the RED-DOT variants that leverage irrelevant evidence    
### Results are shown in Tables 1, 2 and 3.
for RED_DOT_version in ["single_stage", "single_stage_guided", "dual_stage", "dual_stage_guided", "dual_stage_two_transformers"]:
    for k_fold in [1, 3]:
        for num_evidence in [1, 2, 4]:
            run_experiment(RED_DOT_version = RED_DOT_version, 
                           data_path=DATA_PATH,
                           evidence_path=EVIDENCE_PATH,
                           verite_path=VERITE_PATH,
                           use_evidence=num_evidence, 
                           use_evidence_neg=num_evidence, 
                           k_fold=k_fold, 
                           choose_fusion_method = [["concat_1", "add", "sub", "mul"], 
                                                   ["concat_1", "add", "sub"],
                                                   ["concat_1", "add", "mul"],
                                                   ["concat_1", "sub", "mul"],
                                                   ["concat_1"]
                                                  ])