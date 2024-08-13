# <span style="color:#5fa8d3;">Insurance auto claim damage and severity detection for cars</span>

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue) ![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange) ![Platform](https://img.shields.io/badge/platform-aws%20sagemaker%20%7C%20jupyter%20%7C%20streamlit-lightgrey) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)

## Table of Contents

<p>
<span style="color:#fb8500;">1.</span> <a href="#about">About</a><br>
<span style="color:#fb8500;">2.</span> <a href="#example">Example</a><br>
</p>

## <span id="about" style="color:#5fa8d3;">About</span>

This project aims to improve the process of car insurance claims using machine learning by building an end-to-end solution for detecting, localizing, and estimating the severity of **car damages** using **deep learning** techniques. The project is designed to streamline the auto insurance claim process by offering a comprehensive set of tools, including:

<ul>
    <li style="list-style-type: none;">
        <span style="color: #fb8500;">&#8226; </span>
        <strong>Neural Network for Damage Detection and Localization</strong>: a <strong>YOLOv8</strong> model that is retrained with the data obtained from <a href="https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection/data">Vehide Dataset</a> to accurately identify and localize different types of car damages from images.
    </li>
    <li style="list-style-type: none;">
        <span style="color: #fb8500;">&#8226; </span>
        <strong>Cost Estimation Model</strong>: a model that estimates <strong>repair costs</strong> based on the type of damage, the car model, and location-specific factors (including various counties in Ireland) using <strong>synthesized data</strong> created by us.
    </li>
    <li style="list-style-type: none;">
        <span style="color: #fb8500;">&#8226; </span>
        <strong>Design of an AutoClaim Web Application</strong>: A user-friendly application where clients can submit photos of their damaged vehicles, receive a repair cost estimate, and be assigned a workshop for the repair.
    </li>
</ul>

## <span id="example" style="color:#5fa8d3;">Example</span>

Below is an example of how the application and the outputs would appear when a client of the insurance company uploads a car damage.

<p align="center">
  <img src="images_readme/exampleof usage.jpg" alt="Description of Image" width="800px" height="300px"/>
</p>
