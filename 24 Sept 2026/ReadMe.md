- **Folder “Eye-Disease-Classification-Using-Deep-Learning-main”** :- I have been researching for repositories whose pre-trained models could  be suitable for our scenario of GlaucoSense, and found a model in this “https://github.com/navidnayyem/Eye-Disease-Classification-Using-Deep-Learning/tree/main” repository. The provided model’s prediction is closest to our scenario.

  The used model is based on a Convolutional Neural Network (CNN) architecture and is a multi-class fundus image classification model which classifies across 4 eye diseases namely- “_Cataract_”, “_Diabetic Retinopathy_”, “_Glaucoma_” and “_Normal_”. As per the author, this model was trained on a total of 4,217 images out of which 1038 images were used for “Cataract”, 1098 images were used for “Diabetic Retinopathy”, 1007 images were used for “Glaucoma” and 1074 images were used for “Normal”.

  Inside this folder is the file name “_Glaucoma_Inference.py_ “ which is the inference script that has been used to make the pre-trained model predict on our ESRGAN processed fundus images. Further details of what happens after running the script is given below:-

      • In the script if you upload a black and white image, the script will firstly remove the black borders from the particular image. This will only work if the image is black and white.
  
      • The script will then resize the uploaded image’s resolution to 224 x 224 as the model can only predict if the resolution of the image is in this size. The images which we are uploading is of resolution 3912 x 3868.
  
      • It will then perform colour conversion by converting the black and white fundus images to RGB. It is because the model was trained on RGB images.
  
      • It will then perform normalisation by dividing the pixel values with 255, so that the pixel comes within the range between 0 to 1 so that the Neural Network can process it easily.
  
      • This script gives us the probability score of all the mentioned 4 eye diseases and the category which has the highest probability, the fundus image is categorised under that. 
      
  Moreover, there is another file by the name “_NASNetMobile_best.keras_ “ which is the model which we are using.
  

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


- **Folder “glaucoma-detector-master ”** :- I have been researching for repositories whose pre-trained models could  be suitable for our scenario of GlaucoSense, and found a second model in this “https://github.com/golden-panther/glaucoma-detector ” repository. The provided model’s prediction is also closest to our scenario.

  The used model is based on Convolutional Neural Network (CNN) architecture and is a multi-class fundus image classification model which classifies across - “_Glaucoma_” and “_Normal_”. This model was trained on a total of 2,230 images as per the author.

  Inside this folder is the file name “_Glaucoma_inference_script.py_” which is the inference script that has been used to make the pre-trained model predict on our ESRGAN processed fundus images. Further details of what happens after running the script is given below:-

      • In the script if you upload a black and white image, the script will firstly convert the image to RGB as the model was trained on RGB images. 
      • The script will then detect the optic disc in the uploaded fundus image for cropping the particular area. It is because the model was trained on detecting the diseases with the cropped optic disc portion only and since we have full fundus images we need to crop out the optic disc portion and then ask the model to predict on it.
      • After detecting the optic disc area a pop-up window will pop in the screen where you can adjust the cropping portion of the optic disc area of the uploaded fundus image. Once your satisfied with the cropping portion press ENTER, the adjusted portion will be selected which will be given to the model.
      • The script will then resize the cropped portion of the image to 100 x 100 resolution, as the model can only predict in this resolution. The images which we are uploading is of resolution 3912 x 3868.
      • In the terminal, you will get and option of “yes” and “no”. If your satisfied with the above steps then write “y” or “yes” in the terminal so that the image is then moved forward for the model’s processing for the classification, else write “n” or “no”
      • If written “y” or “yes”, the script will then perform normalisation by dividing the pixel values with 255, so that the pixel comes within the range between 0 to 1 so that the Neural Network can process it easily.
      • This script gives us the probability score of the mentioned 2 eye diseases and the category which has the highest probability, the fundus image is categorised under that disease. 
      
Moreover, there is another file by the name “_my_model2.h5_  “ which is the model which we are using.


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


- **File “Performance of the model not integrated in GlaucoSense.pdf “** :-  This is the report pdf file in which i have written the report of both the models performance. I have compared both the models output with the current GlaucoSense AI results based on the same images.

  Even though both the models prediction result about “Glaucoma” is different from that of GlaucoSense, but I think the model of “glaucoma-detector-master ” is much closer to our scenario.


_**Note**_ :- I was not able to validate the results of both the models as I could not find a file where the images actual classification are mentioned. So I am not assured if the models classifications are correct or not.
