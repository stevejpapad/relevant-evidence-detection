import io
import os
import json
import requests
from PIL import Image
import pandas as pd
from tqdm import tqdm
from io import BytesIO
from utils_evidence import (fetch_evidence_split, 
                            load_merge_evidence_w_data, 
                            str_to_list, 
                            process_string, 
                            idx_captions, 
                            idx_images)

# Function to perform inverse image search
def inverse_image_search(image_url, service, CUSTOM_SEARCH_ENGINE_ID):
    search_parameters = {
        "searchType": "image",
        "q": image_url,
        "cx":CUSTOM_SEARCH_ENGINE_ID
    }
    response = service.cse().list(**search_parameters).execute()
    return response.get("items", [])


# Function to perform caption-based image search
def caption_image_search(caption, service, CUSTOM_SEARCH_ENGINE_ID):
    search_parameters = {
        "q": caption,
        "cx": CUSTOM_SEARCH_ENGINE_ID
    }
    response = service.cse().list(**search_parameters).execute()
    return response.get("items", [])

def save_image(url, name, data_path):
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            width, height = img.size

            if not img.mode == 'RGB':
                img = img.convert('RGB')
                
            img.save(data_path + "external_evidence/images/" + name + ".jpg")            
            return True
            
    except Exception as e: 
        
        print(e, "!!!", url)
        return False

def save_images(list_urls, idx, prefix, data_path):
    
    count_img = 0
    for img_url in list_urls:
        
        print(count_img, end="\r")
        
        img_name = prefix + '_image_' + str(idx) + '_' + str(count_img) 
        count_img += 1
        status = save_image(img_url, img_name, data_path)

        
def get_image_paths(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']  # Add more extensions if needed
    image_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths


def filter_images(image_paths, x):
    false_prefix = "false_image_" + str(x) + "_"
    true_prefix = "true_image_" + str(x) + "_"

    false_images = [path for path in image_paths if path.split("/")[-1].startswith(false_prefix)]
    true_images = [path for path in image_paths if path.split("/")[-1].startswith(true_prefix)]

    return false_images, true_images

def natural_sort_key(path):
    parts = path.split("/")
    filename = parts[-1]
    filename_parts = filename.split("_")
    number1 = ''.join(filter(str.isdigit, filename_parts[2]))
    number2 = ''.join(filter(str.isdigit, filename_parts[3]))
    return int(number1), int(number2)

def unique_keep_order(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]


def collect_evidence(data_path = '/VERITE', API_KEY = "YOUR_KEY", CUSTOM_SEARCH_ENGINE_ID = "YOUR_ID"):
    
    from googleapiclient.discovery import build
    
    # Authenticate and create the Google Custom Search service
    service = build("customsearch", "v1", developerKey=API_KEY)

    empty_result = {}

    for k in ['kind', 'title', 'htmlTitle', 'link', 'displayLink', 'snippet', 'htmlSnippet', 'mime', 'fileFormat', 'image']:
        empty_result[k] = None

    if not os.path.isdir(data_path + '/external_evidence'):
        os.makedirs(data_path + '/external_evidence')

    data = pd.read_csv(data_path + 'VERITE_articles.csv', index_col=0)    

    for idx in range(0, data.shape[0]):       

        print(idx, end='\r')

        current_data = data.iloc[idx]

        current_url_t = current_data.true_url
        current_url_f = current_data.false_url

        current_caption_t = current_data.true_caption
        current_caption_f = current_data.false_caption     

        # Perform inverse image search
        inverse_results_t = inverse_image_search(current_url_t, service, CUSTOM_SEARCH_ENGINE_ID)

        if current_url_f:
            inverse_results_f = inverse_image_search(current_url_f, service, CUSTOM_SEARCH_ENGINE_ID) 
        else:
            inverse_results_f = [empty_result]

        caption_results_t = caption_image_search(current_caption_t, service, CUSTOM_SEARCH_ENGINE_ID)
        caption_results_f = caption_image_search(current_caption_f, service, CUSTOM_SEARCH_ENGINE_ID)


        with open(data_path + "external_evidence/" + "true_inverse_" + str(idx) + ".json", "w") as file:
            json.dump(inverse_results_t, file)

        with open(data_path + "external_evidence/" + "false_inverse_" + str(idx) + ".json", "w") as file:
            json.dump(inverse_results_f, file)

        with open(data_path + "external_evidence/" + "true_direct_" + str(idx) + ".json", "w") as file:
            json.dump(caption_results_t, file)    

        with open(data_path + "external_evidence/" + "false_direct_" + str(idx) + ".json", "w") as file:
            json.dump(caption_results_f, file)    
        
def unique_keep_order(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

def keep_images_urls(input_dict):

    thumbnails = []
    images = []

    for i in range(len(input_dict)):
        
        if 'pagemap' in input_dict[i]:
            
            x = input_dict[i]['pagemap']
            if 'cse_thumbnail' in x.keys():
                
                y = x['cse_thumbnail']
                for j in range(len(y)):
                    thumbnails.append(y[j]['src'])

            if 'imageobject' in x.keys():

                y = x['imageobject']

                for j in range(len(y)):

                    if 'contenturl' in y[j]:
                        images.append(y[j]['contenturl'])        

                    if 'url' in y[j]:
                        images.append(y[j]['url'])   

                    if 'image' in y[j]:
                        images.append(y[j]['image'])   

            if 'metatags' in x.keys():

                for k in range(len(x['metatags'])):

                    z = x['metatags'][k]

                    if 'og:image' in z.keys():    
                        images.append(z['og:image'])

                    if 'image' in z.keys():    
                        images.append(z['image'])

                    if 'twitter:image' in z.keys():    
                        images.append(z['twitter:image'])
            
    return unique_keep_order(thumbnails), unique_keep_order(images)
        
def download_images(data_path = '../data/VERITE/'):  
    
    data = pd.read_csv(data_path + 'VERITE_articles.csv', index_col=0)
    
    if not os.path.isdir(data_path + 'external_evidence/images'):
        os.makedirs(data_path + 'external_evidence/images')


    for (idx) in tqdm(range(0, data.shape[0]), total=data.shape[0]):

        with open(data_path + "external_evidence/" + "true_direct_" + str(idx) + ".json", "r") as file:
            direct_results_t = json.load(file)    

        with open(data_path + "external_evidence/" + "false_direct_" + str(idx) + ".json", "r") as file:
            direct_results_f = json.load(file)  

        t_thumbnails, t_direct_images = keep_images_urls(direct_results_t)
        f_thumbnails, f_direct_images = keep_images_urls(direct_results_f)

        save_images(t_direct_images, idx, 'true', data_path)
        save_images(f_direct_images, idx, 'false', data_path)

        
def save_verite_file(data_path = '/VERITE'):
        
    data = pd.read_csv(data_path + 'VERITE_articles.csv', index_col=0)
    
    images_paths = get_image_paths(data_path + "external_evidence/images/")
    images_paths = sorted(images_paths, key=natural_sort_key)

    unpack_data = []

    for (i, row) in tqdm(data.iterrows(), total=data.shape[0]):

        idx = row.id

        true_caption = row.true_caption
        false_caption = row.false_caption 
        true_img_path = 'images/true_' + str(idx) + '.jpg'

        f_direct_images, t_direct_images = filter_images(images_paths, idx)    

        with open(data_path + "external_evidence/" + "true_inverse_" + str(idx) + ".json", "r") as file:
            inverse_results_t = json.load(file)

        with open(data_path + "external_evidence/" + "false_inverse_" + str(idx) + ".json", "r") as file:
            inverse_results_f = json.load(file)

        inverse_t = unique_keep_order([inverse_results_t[i]['title'] for i in range(len(inverse_results_t))])
        inverse_f = unique_keep_order([inverse_results_f[i]['title'] for i in range(len(inverse_results_f))])      

        unpack_data.append({
            'caption': true_caption,
            'image_path': true_img_path,
            'captions': inverse_t,
            'len_text_info': len(inverse_t),        
            'images_paths': t_direct_images,
            'num_images': len(t_direct_images),
            'label': 'true'
        })

        unpack_data.append({
            'caption': false_caption,
            'image_path': true_img_path,
            'captions': inverse_t,
            'len_text_info': len(inverse_t),                
            'images_paths': f_direct_images,        
            'num_images': len(f_direct_images),        
            'label': 'miscaptioned'
        })  

        if pd.notna(row.false_url) and pd.notna(row.query):

            false_img_path = 'images/false_' + str(idx) + '.jpg' 

            if not os.path.isfile(data_path + false_img_path):
                print("!!!!!!!!!!!!!!!!!", false_img_path)

            unpack_data.append({
                'caption': true_caption,
                'image_path': false_img_path,
                'captions': inverse_f,
                'len_text_info': len(inverse_f),                    
                'images_paths': t_direct_images,              
                'num_images': len(t_direct_images),            
                'label': 'out-of-context'
            })              


    verite_df = pd.DataFrame(unpack_data)
    verite_df.to_csv(data_path + 'VERITE_with_evidence.csv')