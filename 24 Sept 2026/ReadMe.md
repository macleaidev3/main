- **Folder “Eye-Disease-Classification-Using-Deep-Learning-main”** :- I have been researching for repositories whose pre-trained models could  be suitable for our scenario of GlaucoSense, and found a model in this “https://github.com/navidnayyem/Eye-Disease-Classification-Using-Deep-Learning/tree/main” repository. The provided model’s prediction is closest to our scenario.

  The used model is based on a Convolutional Neural Network (CNN) architecture and is a multi-class fundus image classification model which classifies across 4 eye diseases namely- “_Cataract_”, “_Diabetic Retinopathy_”, “_Glaucoma_” and “_Normal_”. As per the author, this model was trained on a total of 4,217 images out of which 1038 images were used for “Cataract”, 1098 images were used for “Diabetic Retinopathy”, 1007 images were used for “Glaucoma” and 1074 images were used for “Normal”.

  Inside this folder is the file name “_Glaucoma_Inference.py_ “ which is the inference script that has been used to make the pre-trained model predict on our ESRGAN processed fundus images. Further details of what happens after running the script is given below:-

      • In the script if you upload a black and white image, the script will firstly remove the black borders from the particular image. This will only work if the image is black and white.
  
      • The script will then resize the uploaded image’s resolution to 224 x 224 as the model can only predict if the resolution of the image is in this size. The images which we are uploading is of resolution 3912 x 3868.
  
      • It will then perform colour conversion by converting the black and white fundus images to RGB. It is because the model was trained on RGB images.
  
      • It will then perform normalisation by dividing the pixel values with 255, so that the pixel comes within the range between 0 to 1 so that the Neural Network can process it easily.
  
      • This script gives us the probability score of all the mentioned 4 eye diseases and the category which has the highest probability, the fundus image is categorised under that. 
      
  Moreover, there is another file by the name “NASNetMobile_best.keras “ which is the model which we are using.
