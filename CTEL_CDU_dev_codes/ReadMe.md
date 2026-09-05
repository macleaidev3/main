In this folder consist of the date wise development process of my contribution to the Sentinel Application. Please use only the file from the folder name "5_Sept_2026_full_and_final_all_complete/CTEL_CDU-ml_integration". The file in the folder "5_Sept_2026_full_and_final_all_complete/CTEL_CDU-ml_integration" is the complete implementation of the required task.

I have been working to build a process in the Sentinel App. where in the "Flag" column in the "Corrosion Probe" section a flag message is displayed in the red telling that in the uploaded dataset the date whose prediction is required is missing. We solved this problem by taking the previous 30 days data to fill in the gap of the date where values are missing. This technique has only be applied to the IP21 and Lab Report section of the app. 

The following are the scenarios which can occur:

# **IP21 Scenarios**:

>> _Situation-1_: For the date whose “Cr/Thickness” we need to predict all the data are available for the date in the uploaded dataset. So prediction will happen quickly.

>> _Situation-2:_ For the date whose “Cr/Thickness” we need to predict, the date is present but only with few hours information (instead of 24 hours information). In this situation the recorded hours could be present in a consecutive way or might not be in a consecutive way. So in this case we do not have the full 24 hours data for the certain date for which the integrated model in the Sentinel app will not be able to predict and will show pending status.

>> _Situation-3:_ For the date whose “Cr/Thickness” we need to predict, the whole date along with its values is missing from the uploaded dataset. So in this case we do not have the 24 hours data for the certain date for which the integrated model in the Sentinel app will not be able to predict and will show pending status.

**Lap Report Scenarios:**

>> _Situation-1:_ For the date whose “Cr/Thickness” we need to predict, all the data are available for the date in the uploaded dataset. So prediction will happen quickly.

>> _Situation-2:_ For the date whose “Cr/Thickness” we need to predict, the date is present but one column value is missing.  So in this case we do not have the full data for the certain date for which the integrated model in the Sentinel app will not be able to predict and will show pending status.

>> _Situation-3:_ For the date whose “Cr/Thickness” we need to predict, the whole date along with its values is missing from the uploaded dataset. So in this case we do not the data for the certain date for which the integrated model in the Sentinel app will not be able to predict and will show pending status.


Therefore, to tackle these scenarios, what we have does is that we take the average of the previous 30 days to fill in these gaps and then the data is passed to the model for prediction. So, if by chance while calculating the average from the previous 30 days we found out one the date’s data is missing which is required to be filled to get the missing data of the certain date, we will fill up the date’s values first by taking the average of its previous 30 days. In this way we will fill up the missing gap of the target date whose prediction for “Cr/Thickness” we want to see.
This technique will be used for both the IP21 and Lab Report section and all the missing data calculation for the target date and the other date will get written in the database.

For Example:- I want to predict the “Cr/Thickness” for the date 5/9/2025, this is my target date. But i do not have this date’s data in the uploaded dataset. So SENTINEL will start calculating the average from the previous 30 days to fill in the required column values for the model to predict. While calculating for the previous 30 days the SENTINEL found out the data for the date 30/8/2025 is also missing from the dataset, so SENTINEL will first calculate the average for 30/8/2025 to fill in the values of this particular date and then it will again start calculating the average for 5/9/2025 where it will include the new details of the 30/8/2025.

After filling up the gaps the data will be given to the model for prediction and in the “Corrosion Probe” page we have a column name “Flag” and in this column in red colour a flag message will be displayed showing which sections data was missing along with the date and that the SENTINEL has used the 30 days averaging technique to fill up the required gap for the prediction.
 
We can then refresh the IP21 and Lap Report pages to check the average values. The values will be written in the datasets loaded in that page.

_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

In this folder "5_Sept_2026_full_and_final_all_complete/CTEL_CDU-ml_integration" the files which I have worked on are :

1> src -> ut_ml -> ut_thickness_contributor.py

2> src -> ut_ml -> ut_contributore_database.py

3> src -> ut_ml -> ut_ip21_recovery.py

4> src -> utils -> missing_data_handler.py

5> src -> utils -> year_month_table_combined -> light_column_frozen_table.py

6> src -> utils -> year_month_table_combined -> column_frozen_table.py

7> src -> server_manager -> operation_manager.py

8> src-> utils -> lab_report_recovery.py

9> src-> utils -> core_utility_functions.py

10> src -> application_started -> ml_job.py



