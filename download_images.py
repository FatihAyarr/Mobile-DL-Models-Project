# It is used to download photos from AWS S3. This script creates a new folder in the directory where the script is located and downloads it automatically.
# Lines with description: 11,12,14,15,39

import boto3
import os
from tqdm import tqdm
import pandas as pd
import time


KEY_ID = '' #aws access key id
ACCESS_KEY = '' #aws secret access key 

csv_Base_path = "../" #csv path
csv = "data2.csv" #csv name

def download_images(fortune_list):
    csv_f_name = csv.split('.csv')[0]
    path = csv_Base_path+csv_f_name
    if os.path.exists(path):
        os.popen('rm -r ' + path)
        time.sleep(1)

    os.makedirs(path)
    print(path,os.path.exists(path))


    s3 = boto3.resource('s3',
                        region_name = 'eu-west-1',
                        aws_access_key_id = KEY_ID,
                        aws_secret_access_key = ACCESS_KEY)
    
    f_count = 0
    pbar = tqdm(range(len(fortune_list)))
    for fortune_index in pbar:
        for i in range(3):
            desc = "Photo Count: {}, ID:{}, {}"
            error = "No error"
            if f_count >= 10: #how many photos to download
                break
            filename = ''.join([fortune_list[fortune_index], "_", str(i), ".jpg"])
            save_filename = os.path.join(path, filename)
            try:
                key = ".../" + filename
                s3.Bucket('...').download_file(key, save_filename)
                if os.stat(save_filename).st_size == 0:
                    os.popen('rm -rf '+ save_filename)
                    #print('Deleted 0 byte file', filename)
                    error = 'Deleted 0 byte file'
                    desc = desc.format(f_count,filename,error)
                    pbar.set_description(desc)
                else:
                    f_count += 1
                    desc = desc.format(f_count,filename,error)
                    pbar.set_description(desc)

            except:
                #print(filename, " - File not found")
                error = "File not found"
                desc = desc.format(f_count,filename,error)
                pbar.set_description(desc)
                continue
            
            
            

    print('download ' + ' : image_count: ' +
          str(len(fortune_list)))
    
    # remove duplicates
    print('zip -r '+path+'.zip '+path)
    os.system('zip -r '+path+'.zip '+path)

    return


df = pd.read_csv(csv_Base_path+csv)
rng = df.shape[0]

fortune_list = []
for i in tqdm(range(0, rng - 1)):
    fortune_list.append(df.loc[i][0])
    
download_images(fortune_list)






