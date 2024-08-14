# <span style="color:#5fa8d3;">Insurance auto claim damage and severity detection for cars</span>

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue) ![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange) ![Platform](https://img.shields.io/badge/platform-aws%20sagemaker%20%7C%20jupyter%20%7C%20streamlit-lightgrey) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)

## Table of Contents

<p>
<span style="color:#fb8500;">1.</span> <a href="#about">About</a><br>
<span style="color:#fb8500;">2.</span> <a href="#example">Example</a><br>
<span style="color:#fb8500;">3.</span> <a href="#workflow">Workflow</a><br>
<span style="color:#fb8500;">4.</span> <a href="#structure">Project structure</a><br>
<span style="color:#fb8500;">5.</span> <a href="#depolyment">Deployment of the project</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#fb8500;">5.1.</span> <a href="#local">Local</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#fb8500;">5.2.</span> <a href="#amazonsagemaker">Amazon SageMaker</a><br>
</p>

## <span id="about" style="color:#5fa8d3;">1. About</span>

This project enhances the **car insurance claims** process by leveraging machine learning to create an end-to-end solution that detects, localizes, and estimates the severity of **car damages** using deep learning techniques. It includes a **YOLOv8 model** retrained with data from the [Vehide Dataset](https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection/data) to accurately identify and locate various car damages, a **cost estimation model** obtained from simulated data using **Monte Carlo** methods, and a user-friendly **AutoClaim web application** where clients can upload photos of their damaged vehicles, receive repair cost estimates, and be directed to an appropriate workshop for repairs.


## <span id="example" style="color:#5fa8d3;">2. Example</span>

Below is an example of how the application and the outputs would appear when a client of the insurance company uploads a car damage.

<p align="center">
  <img src="images_readme/example_usage.jpg" alt="Description of Image" width="800px" height="300px"/>
</p>




## <span id="workflow" style="color:#5fa8d3;">3. Workflow</span>
<p align="center">
  <img src="./images_readme/diagram.jpg" alt="Description of Image" width="800px" height="600px"/>
</p>


## <span id="structure" style="color:#5fa8d3;">4. Project Structure</span>
The project is organized into the following main directories:

- [**/API/**](./API/): Contains the *Python* scripts and images used to construct the Autoclaim application using Streamlit.
  - [**/Icons/**](./API/Icons/): Folder for the images used in the construction of the application.
  - [**/pages/**](./API/pages/): Folder containing the Python scripts for each of the secondary pages of the API.
    - [*1_Policy Details*](./API/pages/1_Policy%20Details.py): Python script for displaying the details of the client.
    - [*2_File Claim.py*](./API/pages/2_File%20Claim.py): Python script for opening a claim.
    - [*3_Upload Damage.py*](./API/pages/3_Upload%20Damage.py): Python script for uploading images of the damage.
    - [*4_Final View.py*](./API/pages/4_Final%20View.py): Python script for displaying the prediction of the damage and the assignment of the nearest workshop.
  - [**/uploads/**](API/uploads/): Folder where the uploaded and predicted images are saved.
  - [*Homepage.py*](./API/Homepage.py): Python script that serves as the main program for running the web application.
  - [*requirements.txt*](./API/requirements.txt): Text file with the requirements needed for converting the local Streamlit application to a cloud Streamlit application.

- [**/Data/**](./Data/): Contains the datasets and the COCO format JSON files used in the project (the JSON files ending with *updated* are the ones used throughout the entire project).
  - [**/train/**](./Data/train/): Folder with the *train* and *validation* datasets. It also saves the augmented photos when *[3_data_augmentation.ipynb](./Notebooks/3_data_augmentation.ipynb)* is run.
  - [**/test/**](./Data/test/): Folder with the test photos.
  - [**/Yoloimages/**](./Data/Yoloimages/): Folder, generated when *[4_yolo_code.ipynb](./Notebooks/4_yolo_code.ipynb)* is executed, containing the datasets prepared for the YOLO model. Each folder contains an *images* folder with the images and a *labels* folder with the *.txt* files containing annotations for each damage.
    - [**/train/**](./Data/Yoloimages/train): Directory containing the training set, including both original and augmented data.
    - [**/val/**](./Data/Yoloimages/val): Directory containing the validation set, used to tune the model's hyperparameters and evaluate performance during training.
    - [**/test/**](./Data/Yoloimages/test): Directory containing the test set, used to assess the model's generalization capability on unseen data.

- [**/Models/**](./Models/): Folder containing the cost model [*cost_model.pkl*](./Models/cost_model.pkl) and the car detection models (DIRECTORIES OF MODELS TO BE ADDED).

- [**/Notebooks/**](./Notebooks/): Jupyter notebooks used for the project.
  - *[0_linux_requirements.ipynb](./Notebooks/0_linux_requirements.ipynb)*: Jupyter notebook for installing the requirements defined for SageMaker.
  - [*1_data_processing.ipynb*](./Notebooks/1_data_processing.ipynb): Jupyter notebook for data processing.
  - [*2_Claim_costs.ipynb*](./Notebooks/2_Claim_costs.ipynb): Jupyter notebook explaining how the auto claim data is generated and the model obtained for predicting car damage repair costs.
  - [*3_data_augmentation.ipynb*](./Notebooks/3_data_augmentation.ipynb): Jupyter notebook applying the data augmentation to the data train and merging the new data with the train data.
  - [*4_yolo_code.ipynb*](./Notebooks/4_yolo_code.ipynb): Code for preparing the data for the YOLO model, migrating the data to Amazon S3, and tuning the hyperparameters and training in Amazon SageMaker.
  - [*98_model_predictions.ipynb*](./Notebooks/98_model_predictions.ipynb): Jupyter notebook for checking the predictions of the cost estimation model.
  - [*99_check_photos.ipynb*](./Notebooks/99_check_photos.ipynb): Jupyter notebook for plotting the photos and their corresponding damage polygons.
  - [*data.yaml*](./Notebooks/data.yaml), [*data_test.yaml*](./Notebooks/data_test.yaml), [*data_train_final.yaml*](./Notebooks/data_train_final.yaml), [*data_tune_metrics.yaml*](./Notebooks/data_tune_metrics.yaml), [*data_train_final_metrics.yaml*](./Notebooks/data_train_final_metrics.yaml): YAML files for specifying the validation and training datasets for the YOLO models depending on the context (for training or validating the data).

- [**/Sagemaker/**](./Sagemaker/): Folder containing the two Python programs ([train_tune.py](./Sagemaker/train_tune.py) and [train_final.py](./Sagemaker/train_final.py)) with the code for defining the models to train in Amazon SageMaker.

- [**/src/**](./src/): Folder containing the [*mymodule.py*](./src/mymodule.py) script with all the functions used in the project, each containing an explanation of their purpose, inputs, and outputs.

- [**/Others/**](./Others/): Folder with various files that were useful during the project (e.g., Git steps for every time SageMaker was opened, relevant links, tips, etc.).

- [**/images_readme/**](./images_readme/): Folder with the images used to generate this README.

- [*requirements.txt*](./requirements.txt): File listing all the dependencies needed to run the project locally.
- [*requirements_linux.txt*](./requirements_linux.txt): File listing all the dependencies needed to run the project in Amazon SageMaker.
- [*config.yaml*](./config.yaml): YAML file defining the variables used throughout the project.


##  <span id="deployment" style="color:#5fa8d3;">5. Deployment of the project</span>
Our work is divided into two main parts: one that can be executed locally and another that, due to the high computational power required by YOLOv8, has been migrated to Amazon SageMaker. In the local environment, tasks such as data preparation and processing, model evaluation, and small-scale experiments can be performed. However, training and hyperparameter optimization, which demand significant computational resources, are carried out on Amazon SageMaker. Below, we explain in detail the activities that can be performed in each environment.



###  <span id="local" style="color:#5fa8d3;">5.2 Local</span>
The parts of the project that can be computed locally are:
- [*1_data_processing.ipynb*](./Notebooks/1_data_processing.ipynb)
- [*2_Claim_costs.ipynb*](./Notebooks/2_Claim_costs.ipynb)
- [*3_data_augmentation.ipynb*](./Notebooks/3_data_augmentation.ipynb
- The application of [**/API/**](./API/)

This part of the project have been executed with [Python 3.11]() and [Python  3.12.4](https://www.python.org/downloads/release/python-3124/). 

It is a good practice to create an evironment for working in projects like this one. After creating the enviroment and once it is activate it, it has to be run in the terminal inside the main folder of the project (in the case that it is a python enviroment):

````
pip install requirements.txt
````

<pre>
<code>
def hello_world():
    print("Hello, World!")
</code>
</pre>

### Local 

### Sagemaker

## Join pkl 