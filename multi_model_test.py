# this script is used when multiple models are to be tested at the same time. or when a model is to be tested for 2 classes (cup, red) together.
# lines with description: 24, 30, 37, 130, 160, 173, 176, 218, 220

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import boto3
import time
from tqdm import tqdm
from numba import cuda 
from skimage.transform import resize
from skimage.io import imread
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,2"



model_list = ['new_mobil.tflite', 'new_mobil.tflite', 
              'new_mobil_optimized.tflite', 'new_mobil_optimized.tflite'
             ] # each of the models to be tested must be added to the list 2 times consecutively 

sayac = 1

for j in tqdm(model_list):
    if sayac%2 == 1:
        base_path = "../FatihNewModel/Datasets/TFLite_56-44/test/class1" # path of the class1 folder to test
        data = []
        for root, dirs, files in os.walk(base_path, topdown=False):
            for name in files:
                data.append(os.path.join(root, name))
                
    else:
        base_path = "../FatihNewModel/Datasets/TFLite_56-44/test/class2" # path of the class2 folder to test
        data = []
        for root, dirs, files in os.walk(base_path, topdown=False):
            for name in files:
                data.append(os.path.join(root, name))


    data_size = len(data)
    print(data_size)



    df = pd.DataFrame(data, columns =['Names'])
    df.to_csv('names.csv')

    '''
    def read_tensor_from_image_file(file_name, input_height=224, input_width=224,
                    input_mean=0, input_std=255):
      input_name = "file_reader"
      output_name = "normalized"
      file_reader = tf.io.read_file(file_name, input_name)

      if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels = 3,
                                           name='png_reader')
      elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
      elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
      else:
        image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                            name='jpeg_reader')
      resized = tf.image.resize(image_reader, [input_height, input_width])

      return resized
    '''

    def chunks(lst, n):
        result = []
        for i in range(0, len(lst), n):
            result.append(lst[i:i + n])
        return result

    if sayac%2 == 1:
        def calculateAccuracy(total, i, current): 
            i_ = i
            if current != "class1" :
                i_ -= 1

            if i_ < 0:
                i_ = 0


            acc = i_ / total
            if acc > 1:
                acc = 1

            if acc < 0:
                acc = 0

            return acc, i_
        
    else:
        def calculateAccuracy(total, i, current): 
            i_ = i
            if current != "class2" :
                i_ -= 1

            if i_ < 0:
                i_ = 0


            acc = i_ / total
            if acc > 1:
                acc = 1

            if acc < 0:
                acc = 0

            return acc, i_




    def printLabel(f,r):
        if f>r :
            return "class1"
        elif r>f:
            return "class2"



    TFLITE_MODEL = '/FatihNewModel/ML/converted_models/' + j # enter the path of the folder where the model to be tested is located, do not enter the model name                                                                    
    tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    print("== Input details ==")
    print("shape:", input_details[0]['shape'])
    print("type:", input_details[0]['dtype'])
    print("\n== Output details ==")
    print("shape:", output_details[0]['shape'])
    print("type:", output_details[0]['dtype'])


    result = {
                'link': [],
                'class2': [],
                'class1': [],
                'label': [],
                'id': []
    }
    k = data_size
        
    ii = 0
    pbar = tqdm(chunks(data[:k],1))
    for dataChunk in pbar:
        start = time.time()

        chunkLen = len(dataChunk)

        tflite_interpreter.resize_tensor_input(input_details[0]['index'], (chunkLen, 224, 224, 3)) # input values are entered whatever they are
        tflite_interpreter.resize_tensor_input(output_details[0]['index'], (chunkLen, 2))
        tflite_interpreter.allocate_tensors()

        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()

        photo_matrix = []
        data_id = ""
        for i in dataChunk:
            data_id = i.split(base_path)[1].split("/")[1].split(".jpg")[0]
            #img_resized = read_tensor_from_image_file(i)
            img = imread(i)
            img_resized = resize(img,(224,224,3)) # be resized according to input size
            photo_matrix.append(img_resized)
        val_image_batch = np.stack(photo_matrix,axis=0)
        val_image_batch = val_image_batch.astype('float32') # changed if the model receives a different input, e.g. 'uint8'
        #print(type(val_image_batch[0][0]))

        # Set batch of images into input tensor
        tflite_interpreter.set_tensor(input_details[0]['index'], val_image_batch)
        # Run inference
        #print("Prediction started!")
        tflite_interpreter.invoke()
        # Get prediction results
        tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
        #print("Prediction results shape:", tflite_model_predictions.shape)

        # >> Prediction results shape: (36, 2)

        # Convert prediction results to Pandas dataframe, for better visualization
        tflite_pred_dataframe = pd.DataFrame(tflite_model_predictions)
        tflite_pred_dataframe.columns = ['class1','class2']


        f = tflite_pred_dataframe['class1'][0]
        r = tflite_pred_dataframe['class2'][0]

        result['link'].append("https://xxxx.s3.amazonaws.com/yyyyy/" + data_id + ".jpg")
        result['id'].append(data_id)
        result['class1'].append(f)
        result['class2'].append(r)


        currentlbl = printLabel(f,r)
        result['label'].append(currentlbl)
        
        acc, i_ = calculateAccuracy(data_size,k,currentlbl)
        k = i_
        #print(acc, flush=True)
        pbar.set_description("Accuracy {:.4%} : Current Image ID: {}, label: {}, f: {:.3%}, r:{:.3%}  ----".format(acc,data_id,currentlbl,f,r))
        end = time.time()
        ii += 1
        #print((end-start)/60)

    tf_result = pd.DataFrame(result)  
    #tf_result.to_csv('class1_ds_test.csv')
    if sayac%2 == 1:
        tf_result.to_excel('/FatihNewModel/TFLite_Model_Maker/Results/' + j + '_class1_results_model.xlsx') # excel name
    else:
        tf_result.to_excel('/FatihNewModel/TFLite_Model_Maker/Results/' + j + '_class2_results_model.xlsx') # excel name
    sayac += 1






