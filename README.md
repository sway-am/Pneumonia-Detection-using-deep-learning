# Pneumonia-Detection-using-deep-learning

### Description
<pre>
1. Detected Pneumonia from images of Chest X-rays, utilizing the kaggle data set(1.15 GB) to train the model.
2. Used "ImageDataGenerator" from keras.preprocessing.image for Data argumentation
3. Implemented a custom CNN architecture producing a test accuracy of 92.47 % and test loss of 0.39.
4. Implemented Resnet50, VGG 19 and InceptionV3 using transfer learning and compared the results to that of the custom model.
</pre>

#### Dataset
<pre>
Dataset Name      : Chest X-Ray Images (Pneumonia)
Dataset Link      : <a href=https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia>Chest X-Ray Images (Pneumonia) Dataset (Kaggle)</a>
                  : <a href=https://data.mendeley.com/datasets/rscbjbr9sj/2>Chest X-Ray Images (Pneumonia) Dataset (Original Dataset)</a>
Reference Papers   : <a href=https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5>Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning</a>
                   (Daniel S. Kermany, Michael Goldbaum, Wenjia Cai, M. Anthony Lewis, Huimin Xia, Kang Zhang)
                   https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
</pre>


<pre>
<b>Dataset Details</b>
Dataset Name            : Chest X-Ray Images (Pneumonia)
Number of Class         : 2
Number/Size of Images   : Total      : 5856 (1.15 Gigabyte (GB))
                          Training   : 5216 (1.07 Gigabyte (GB))
                          Validation : 320  (42.8 Megabyte (MB))
                          Testing    : 320  (35.4 Megabyte (MB))
</pre>

###Accuracy comparision :

![meta-chart](https://github.com/sway-am/Pneumonia-Detection-using-deep-learning/assets/118014263/cf0a986e-5857-4968-aed6-c36d9f021faa)

#### <b>Model Parameters</b>
<pre>
<b>Custom Deep Convolutional Neural Network : </b>
<b>Training Parameters : </b>
  
>> Batch Size               : 32
>> Number of Epochs         : 111
>> Learning Rate            : ReduceLROnPlateau(monitor='val_accuracy', patience = 3,verbose=1, factor=0.5, min_lr=0.0001)
>> Training Time            : ~5 Hours
>> Optimizer                :'Adam'

Model: "sequential_10"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_51 (Conv2D)          (None, 120, 120, 32)      320       
                                                                 
 batch_normalization_21 (Bat  (None, 120, 120, 32)     128       
 chNormalization)                                                
                                                                 
 max_pooling2d_30 (MaxPoolin  (None, 60, 60, 32)       0         
 g2D)                                                            
                                                                 
 conv2d_52 (Conv2D)          (None, 60, 60, 32)        9248      
                                                                 
 batch_normalization_22 (Bat  (None, 60, 60, 32)       128       
 chNormalization)                                                
                                                                 
 max_pooling2d_31 (MaxPoolin  (None, 30, 30, 32)       0         
 g2D)                                                            
                                                                 
 conv2d_53 (Conv2D)          (None, 30, 30, 64)        18496     
                                                                 
 batch_normalization_23 (Bat  (None, 30, 30, 64)       256       
 chNormalization)                                                
                                                                 
 max_pooling2d_32 (MaxPoolin  (None, 15, 15, 64)       0         
 g2D)                                                            
                                                                 
 conv2d_54 (Conv2D)          (None, 15, 15, 128)       73856     
                                                                 
 batch_normalization_24 (Bat  (None, 15, 15, 128)      512       
 chNormalization)                                                
                                                                 
 max_pooling2d_33 (MaxPoolin  (None, 8, 8, 128)        0         
 g2D)                                                            
                                                                 
 conv2d_55 (Conv2D)          (None, 8, 8, 256)         295168    
                                                                 
 batch_normalization_25 (Bat  (None, 8, 8, 256)        1024      
 chNormalization)                                                
                                                                 
 max_pooling2d_34 (MaxPoolin  (None, 4, 4, 256)        0         
 g2D)                                                            
                                                                 
 flatten_10 (Flatten)        (None, 4096)              0         
                                                                 
 dense_30 (Dense)            (None, 128)               524416    
                                                                 
 dropout_20 (Dropout)        (None, 128)               0         
                                                                 
 dense_31 (Dense)            (None, 64)                8256      
                                                                 
 dropout_21 (Dropout)        (None, 64)                0         
                                                                 
 dense_32 (Dense)            (None, 1)                 65        
                                                                 
=================================================================
Total params: 931,873
Trainable params: 930,849
Non-trainable params: 1,024
_________________________________________________________________


<b>ResNet 50 : </b>
<b>Training Parameters : </b>
  
>> Batch Size               : 32
>> Number of Epochs         : 30
>> Learning Rate            : tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=8)
>> Total params             : 931,873
>> Trainable params         : 930,849
>> Non-trainable params     : 1,024

<b>VGG19: </b>
<b>Training Parameters : </b>
  
>> Batch Size               : 32
>> Number of Epochs         : 30
>> Learning Rate            : tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=8)
>> Total params             : 20090177 (76.64 MB)
>> Trainable params         : 65793 (257.00 KB)
>> Non-trainable params     : 20024384 (76.39 MB)

<b>InceptionV3: </b>
<b>Training Parameters : </b>
  
>> Batch Size               : 32
>> Number of Epochs         : 30
>> Learning Rate            : tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=8)
>> Total params             : 22065185 (84.17 MB)
>>Trainable params          : 262401 (1.00 MB)
>>Non-trainable params      : 21802784 (83.17 MB)
  
</pre>

#### Confusion Matrix and model Evaluations of Custom Model
![Confusion Mt_custom_kagg](https://github.com/sway-am/Pneumonia-Detection-using-deep-learning/assets/118014263/8a0d3d48-fc6a-489b-ab66-d82982c1842c)
![Train and val loss_custom_kagg](https://github.com/sway-am/Pneumonia-Detection-using-deep-learning/assets/118014263/9fb5c220-8614-4d53-a3c2-e5c45bd30993)
![train and val loss 2_custom_kagg](https://github.com/sway-am/Pneumonia-Detection-using-deep-learning/assets/118014263/b8cf9f50-80a9-4c84-9f69-38006173a244)


#### Tools / Libraries
<pre>
Languages               : Python v 3.10.6
Tools/IDE               : Kaggle cloud editor, Google Collabatory
Libraries               : Pandas, seaborn, matplotlib, Keras, TensorFlow, Inception, ImageNet
</pre>

#### Dates
<pre>
Current Version         : v1.0.0.0
Last Update             : 24.10.2023
</pre>

#### Other Reference
<pre>
  1. <href> https://github.com/anjanatiha/Pneumonia-Detection-from-Chest-X-Ray-Images-with-Deep-Learning/tree/master </href>
  2. <href>https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/code?datasetId=17810&searchQuery=resnet+&language=Python&tagIds=16580</href>
  3. <href> https://www.kaggle.com/code/danushkumarv/pneumonia-detection-resnet </href>
</pre>
