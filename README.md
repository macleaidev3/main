# This is the branch where I am storing my GlaucoSense related work:


## 1> "14 Sept 2026" :

The folder "14 Sept 2026" is the work completed main folder. Inside this folder we have 2 more folders named :- "Evaluation of Fundus Images" and "Real_ESRGAN".

- Folder "Evaluation of Fundus Images" :- Inside this folder we have a python script named "Evaluating_ESRGAN_process_script.py" which is being used to evaluated the quality of the images between ORIGINAL IMAGES and ESRGAN IMAGES. In this script we have used a hybrid no-reference image-quality assessment (IQA) approach technique which is  based on Sharpness (Laplacian variance), Contrast (standard deviation), and Entropy, combined into a weighted relative quality score.
  
  This script saves a .csv file inside the "ESRGAN_Result" folder in which we have saved the metric of the evaluation between the ORIGINAL IMAGES and the ESRGAN IMAGES. In the .csv file the column name "Quality Change (%)" is the column which is being treated as the final score percentage of the quality improvement between the images. This column shows how much the quality of the images has improved after processing the original images with the ESRGAN model.
  
  Moreover, the script also generates a new folder by the name "Red Flag" inside which we have separately stored those images whose quality did not improve much. If the percentage values in the column "Quality Change (%)" is <=60% (less than equal to 60%) that ESRGAN image gets deleted from the "ESRGAN_Result" folder and is stored in the "Red Flag" folder. 


- Folder "Real_ESRGAN" :- Inside this folder we are using the model "RealESRGAN_x2plus.pth" and the script "Real_ESRGAN.py" to use the model to improve the quality of the ORIGINAL FUNDUS IMAGES. We have download the model Real_ESRGAN from the GitHub source "https://github.com/xinntao/Real-ESRGAN" and its doc page "https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md" where we got the "RealESRGAN_x2plus.pth" model. Earlier we were using the "RealESRGAN_x4plus.pth" model but it was giving an issue as the model was converting the images to paint like images. Moreover, this code also generates a new folder by the name "ESRGAN_Result" where it stores the newly ESRGAN processed images.
