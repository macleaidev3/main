Folder contains the 3D visualizations of validation data for 5 test cases (at input and output). Here, we are basically checking the predicted "Corrosion Rate" in the Input and Output sections of the equipment. The predicted "Corrosion Rate" has been taken from the model's predicted data on 5 unseen data sets. 

In this folder use the files stored on the folder name "27_Aug_2026". All the files in the folder "27_Aug_2026" are completed.

Inside the folders, we have stored the folders of 7 sections I have worked on. Inside those folder we have the python scripts to generate the 3D models with the corrosion rate. Each of the files details are given below:

1> **_Load_STL_file_Script-1.py_**  :- We have used STL files to generate our 3D visualizations. So in this script we are loading the STL file to check the if the equipement is correct or not.

2> **_Red_strip_Script_2.py_**  :- We have used this script to mark the RED STRIP in the INLETs and OUTLETs of the equipment's. This script only marks the red strip, nothing else.

3> **_Converting_STL_to_CSV_script_3.py_** :- We have used this script to extract the X, Y, Z coordinates of the equipment's from the whole STL file. 

4> **_Coordinates_extract_Red_Strip_Script_4.py_** :- We have used this script to extract the coordinates of X, Y, Z of the places where the RED STRIP has been marked in the equipment. These coordinates will be saved in an output file name like "Nozzle_1_Red_Strip.csv", "Nozzle_2_Red_Strip.csv", etc. 

5> **_Nozzle_csv_convert_Script_5.py_** :- We have used this script to convert the extracted X, Y, Z values for point number 4 to the values that matches our model's predicted csv files values. These conversion is saved in the output file name like "COnverted_Nozzle_1_Red_Strip.csv", "COnverted_Nozzle_2_Red_Strip.csv", etc.

6> **_Combine_datasets_Script_6.py_** :- 
