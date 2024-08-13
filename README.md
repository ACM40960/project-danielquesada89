# <span style="color:#5fa8d3;">Insurance auto claim damage and severity detection for cars</span>

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue) ![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange) ![Platform](https://img.shields.io/badge/platform-aws%20sagemaker%20%7C%20jupyter%20%7C%20streamlit-lightgrey) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)

## Table of Contents

<p>
<span style="color:#fb8500;">1.</span> <a href="#about">About</a><br>
<span style="color:#fb8500;">2.</span> <a href="#example">Example</a><br>
<span style="color:#fb8500;">3.</span> <a href="#workflow">Workflow</a><br>
<span style="color:#fb8500;">4.</span> <a href="#structure">Project structure</a><br>
</p>

## <span id="about" style="color:#5fa8d3;">About</span>

This project enhances the **car insurance claims** process by leveraging machine learning to create an end-to-end solution that detects, localizes, and estimates the severity of **car damages** using deep learning techniques. It includes a **YOLOv8 model** retrained with data from the [Vehide Dataset](https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection/data) to accurately identify and locate various car damages, a **cost estimation model** obtained from simulated data using **Monte Carlo** methods, and a user-friendly **AutoClaim web application** where clients can upload photos of their damaged vehicles, receive repair cost estimates, and be directed to an appropriate workshop for repairs.


## <span id="example" style="color:#5fa8d3;">Example</span>

Below is an example of how the application and the outputs would appear when a client of the insurance company uploads a car damage.

<p align="center">
  <img src="images_readme/example_usage.jpg" alt="Description of Image" width="800px" height="300px"/>
</p>




## <span id="workflow" style="color:#5fa8d3;"> Workflow</span>

<p align="center">
  <img src="./images_readme/diagram.jpg" alt="Description of Image" width="800px" height="600px"/>
</p>




## <span id="structure" style="color:#5fa8d3;">Project Structure</span>
The project is organized into the following main directories:

- **/API/**: contains the *python* scripts and images used to construct the Autoclaim application using Streamlit.

- **/Data/**: Contains the datasets and the COCO format JSON used in the project.
  - **/train/**: folder with the *train* and *validation* folders. It also saves the augmented photos when *[3_data_augmentation.ipynb](./Notebooks/3_data_augmentation.ipynb)* is run.
  - **/test/**: folder with the test photos.
  - **/Yoloimages/**: folder, which is generated when *[4_yolo_code.ipynb](./Notebooks/4_yolo_code.ipynb)* is executed, with the datasets already prepared for the YOLO model. Each folder will have an *images* folder containing the images and a *labels* folder containing the *.txt* files with the annotations for each damage.
    - **/train/**: A directory containing the training set, including both original and augmented data.
    - **/val/**: A directory containing the validation set, used to tune the model's hyperparameters and evaluate performance during training.
    - **/test/**: A directory containing the test set, used to assess the model's generalization capability on unseen data.


- **/Models/**: folder containing the cost model and the car detection models.

- **/Notebooks/**: Jupyter notebooks used for the project.
  - *0_linux_requirements.ipynb*: Jupyter notebook for installing the requirements defined for Sagemaker.
  - *1_data_processing.ipynb*: Jupyter notebook with the data processing.
  - *2_Claim_costs.ipynb*: Jupyter notebook that explains how the auto claim data is generated and the model obtained for predicting the car damage reparation.
  - *4_yolo_code.ipynb*: Code with all the preparation of the data for the YOLO model, the migration of the data to Amazon S3, and the code for hyperparameter tuning (explaining the election of the parameters) and training in Amazon Sagemaker.
  - *98_model_predictions.ipynb*: Jupyter notebook to check the predictions of the code estimation model.
  - *99_check_photos.ipynb*: Jupyter notebook for plotting the photos and their corresponding damage polygons.
  - *data.yaml*, *data_test.yaml*, *data_train.yaml*, *data_tune.yaml*: YAML files for indicating the validation and train files for the YOLO models depending on the context.
  
- **/Sagemaker/**: folder that contains the two python programs with the code for defining the models to train in Amazon Sagemaker.

- **/src/**: Folder that contains the *mymodule.py* script with all the functions used in the project, each one containing an explanation of their function, outputs, and inputs.

- **/Others/**: Folder with different files that have been useful during the project (git steps for every time that we opened Sagemaker, some relevant links, tips, etc.).

- **/images_readme/**: Folder with the images for generating this README.

- *requirements.txt*: Contains the file that lists all the dependencies needed to run the project locally.
- *requirements_linux.txt*: Contains the file that lists all the dependencies needed to run the project in Amazon Sagemaker.
- *config.yaml*: yaml file with the definition of the variables that are used over the project.


## Usage of Each Folder

## Join pkl 