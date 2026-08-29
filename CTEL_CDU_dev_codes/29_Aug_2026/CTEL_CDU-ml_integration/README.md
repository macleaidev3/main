# Sentinel

**Sentinel** is a software tool designed for **corrosion rate calculation** and **thickness reduction analysis** for the **Overhead Distillation Column pipelines and instrument**.  
It integrates process flow visualization with corrosion loop diagrams and supports detailed reporting.

This readme defines the software which is deployed in Kochi Refinery on date 15th July, 2026

---

## Features

- **Main Overview**  
  - Representation of Process diagram of the refinery.
  - There are *"i"* buttons corresponding to equipment, pipeline and corrosion probes
  - Clicking on the *"i"* button for equipment and pipelines will show the corresponding static 3D visualization.
  - Clicking on the *"i"* button corresponding to the corrosion probes will open the menu options to see the thickness table and graph of a selected corrosion probe.

- **General Crude**  
  - This is the table that maintains the details of all the crudes that are used in the refinery.

- **Lab reports**  
  - These are the tables that maintain the record of daily lab analysis

- **Crude Blend**  
  - Create the crude blend to be processed.

- **Cr/Thickness**  
  - Interface to manually select instrument, pipeline and Corrosion Probes and perform thickness prediction or corrosion rate prediction.
  - In the software version deployed in Kochi refinery, only the thickness prediction of the corrosion probes is implemented

- **Export Report**  
  - Interface to export reports as excel files in the local system.

- **Database Management**  
  - Two separate database are maintained: **1. Sentinel Internal Database** and **KR Database**
  - The Sentinel internal database is independent of client database connection(i.e, Kochi refinery database). It is used to maintain the data for the software utilities.
  - The KR database is in the client database server and it has to be set up to be connected with client IP address, User and password with SQL server authentication.

- **Data synchroniztion**  
  - Data from the KR database table is synchronized into the Sentinel Internal database.
  - These data include IP21 data and LIMS data.
  - Thus, Sentinal is updated with the synchronized IP21 data and LIMS data from the refinery database.

- **License**  
  - Proper License is maintained till 31/12/2027 

---

## Installation & Usage

Operating system requirement: **Windows 10 and above**

Python version: **Python 3.12.3**

---

Follow these steps to set up and run Sentinel:

### 1. Clone the Repository (branch: `KRDeployed15July2026`)
```bash
git clone -b KRDeployed15July2026 https://github.com/MACLEAI/CTEL_CDU.git
```

```bash
cd <repository-folder>
```

### 2. Create a Virtual Environment (Python 3.12.3)
```bash
python -m venv vEnv
```

### 3. Activate the Virtual Environment

```bash
.\vEnv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```
### 5. Set up database connectivity

Database set up: [Refer database set up manual](https://docs.google.com/document/d/1zLKk6UWWw4fdhGkhF5qkRhstUaL-VUClga7xKdYNNhA/edit?tab=t.0#heading=h.d9vgk67qwkya)  


### 7. Download the ML model

Download the ml models from [ML Model](https://drive.google.com/file/d/1esllj_O_T_wsfHQnEyTzt-Y4DdQUSjD0/view?usp=sharing)

Extract it and cut at paste it inside them ml_module folder


### 8. Run the Application
```bash
python main.py
```

User manual Link

[User Manual](https://docs.google.com/document/d/1H9cNYoOV8JXJSF1rQ_16qVgIOfLpSDguAXwAvbIP9kY/edit?tab=t.j0d9nxjowmfp#heading=h.t1lde97or015)

### Notes

Ensure you have Python 3.12.3 installed.

Recommended to use a dedicated virtual environment to avoid dependency conflicts.

Corrosion visualization (2D/3D) features are still in development.(Static data visualization is integrated for pipelines and equipment)
