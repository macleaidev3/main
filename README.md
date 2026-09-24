# This is the branch where I am storing my GlaucoSense related work:


## 1> "14 Sept 2026" :

The folder "14 Sept 2026" is the work completed main folder. The work's serial number is **"_PRJ-0021/12_"** in the iid platform. Inside this folder we have 2 more folders named :- "Evaluation of Fundus Images" and "Real_ESRGAN".

- Folder "Evaluation of Fundus Images" :- Inside this folder we have a python script named "Evaluating_ESRGAN_process_script.py" which is being used to evaluated the quality of the images between ORIGINAL IMAGES and ESRGAN IMAGES. In this script we have used a hybrid no-reference image-quality assessment (IQA) approach technique which is  based on Sharpness (Laplacian variance), Contrast (standard deviation), and Entropy, combined into a weighted relative quality score.
  
  This script saves a .csv file inside the "ESRGAN_Result" folder in which we have saved the metric of the evaluation between the ORIGINAL IMAGES and the ESRGAN IMAGES. In the .csv file the column name "Quality Change (%)" is the column which is being treated as the final score percentage of the quality improvement between the images. This column shows how much the quality of the images has improved after processing the original images with the ESRGAN model.
  
  Moreover, the script also generates a new folder by the name "Red Flag" inside which we have separately stored those images whose quality did not improve much. If the percentage values in the column "Quality Change (%)" is <=60% (less than equal to 60%) that ESRGAN image gets deleted from the "ESRGAN_Result" folder and is stored in the "Red Flag" folder. 


- Folder "Real_ESRGAN" :- Inside this folder we are using the model "RealESRGAN_x2plus.pth" and the script "Real_ESRGAN.py" to use the model to improve the quality of the ORIGINAL FUNDUS IMAGES. We have download the model Real_ESRGAN from the GitHub source "https://github.com/xinntao/Real-ESRGAN" and its doc page "https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md" where we got the "RealESRGAN_x2plus.pth" model. Earlier we were using the "RealESRGAN_x4plus.pth" model but it was giving an issue as the model was converting the images to paint like images. Moreover, this code also generates a new folder by the name "ESRGAN_Result" where it stores the newly ESRGAN processed images.



## 2> "24 Sept 2026" :

The folder “24 Sept 2026” is the work completed main folder. The work’s serial number is **_“PRJ-0021/11”_** in the iid platform. Inside this folder we have 2 more folders named:- “Eye-Disease-Classification-Using-Deep-Learning-main” and “glaucoma-detector-master “ and a pdf file named “Performance of the model not integrated in GlaucoSense.pdf “.

- Folder “Eye-Disease-Classification-Using-Deep-Learning-main” :- I have been researching for repositories whose pre-trained models could  be suitable for our scenario of GlaucoSense, and found a model in this “https://github.com/navidnayyem/Eye-Disease-Classification-Using-Deep-Learning/tree/main” repository. The provided model’s prediction is closest to our scenario.
       
  The used model is based on a Convolutional Neural Network (CNN) architecture and is a multi-class fundus image classification model which classifies across 4 eye diseases namely- “Cataract”, “Diabetic Retinopathy”, “Glaucoma” and “Normal”. As per the author, this model was trained on a total of 4,217 images out of which 1038 images were used for “Cataract”, 1098 images were used for “Diabetic Retinopathy”, 1007 images were used for “Glaucoma” and 1074 images were used for “Normal”.

- Folder “glaucoma-detector-master ” :- I have been researching for repositories whose pre-trained models could  be suitable for our scenario of GlaucoSense, and found a second model in this “https://github.com/golden-panther/glaucoma-detector ” repository. The provided model’s prediction is also closest to our scenario.

  The used model is based on Convolutional Neural Network (CNN) architecture and is a multi-class fundus image classification model which classifies across - “Glaucoma” and “Normal”. This model was trained on a total of 2,230 images as per the author.

- File “Performance of the model not integrated in GlaucoSense.pdf“ :-  This is the report pdf file in which i have written the report of both the models performance. I have compared both the models output with the current GlaucoSense AI results based on the same images. Even though both the models prediction result about “Glaucoma” is different from that of GlaucoSense, but I think the model of “glaucoma-detector-master ” is much closer to our scenario.



## 3> _**"Do not use these code"**_ :

We will delete the folder "_Do not use these code_"  along with the all the non useable folder except for the above 2 folders after discussing with Sir. Till then in this branch only use the above mentioned folders for the work. 
