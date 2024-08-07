
import pandas as pd # to manipulate datasets
import numpy as np # to perform numerical operations
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for plotting nicer
from scipy.stats import gamma
import cv2
import os
import json # to import jsons
from IPython.display import clear_output, display
import datetime
import random
import shutil
import ipywidgets as widgets
from PIL import Image  # Import the Image module from Pillow
from ultralytics.data.converter import convert_coco
import yaml
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import subprocess

################################################
########## 1 Data Processing Functions #########
################################################

### Function calculate_bounding_box
def calculate_bounding_box(row):
    """
    Obtain the rectangles to indicate the damages of the car.

    Args:
        row (pd.Series): series of the damage of the image.

    Returns:
        rectangle (pd.Series): series of the rectangle of the damage.
    """
    min_x = min(row["all_x"])
    max_x = max(row["all_x"])
    min_y = min(row["all_y"])
    max_y = max(row["all_y"])

    x = min_x
    y = min_y
    width = max_x - min_x
    height = max_y - min_y

    rectangle = pd.Series([x, y, width, height], index=["x_rect", "y_rect", "width_rect", "height_rect"])
    return rectangle


def dataframe_to_coco_format(df, categories):
    """
   Convert DataFrame to COCO format.

    Args:
        df (pd.DataFrame): DataFrame with the data.
        categories (list): list of categories.

    Returns:
        dict: dictionary in COCO format.
    """
    coco_format = {
        "info": {
            "description": "Vehide",
            "url": "https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data",
            "version": "0.1.0",
            "year": "2024",
            "contributor": "Daniel_Teresa",
            "date_created": "datetime.datetime.now().isoformat()"
        },
        "licenses": [
            {
                "id": 1,
                "name": "",
                "url": ""
            }
        ],
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "car_damage"} for i, name in enumerate(categories)
        ],
        "images": [],
        "annotations": []
    }

    annotation_id = 1
    for id, image_name in enumerate(df["name"].unique()):
        df_image = df[df["name"] == image_name]
        row = df_image.iloc[0]
        image_info = {
            "id": id + 1,
            "file_name": row["name"],
            "width": row["width"],
            "height": row["height"],
            "date_captured": datetime.datetime.now().isoformat(),
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        }
        coco_format["images"].append(image_info)

        for idx, row in df_image.iterrows():
            bbox = [row["x_rect"], row["y_rect"], row["width_rect"], row["height_rect"]]
            area = row["width_rect"] * row["height_rect"]
            segmentation = []
            for x, y in zip(row["all_x"], row["all_y"]):
                segmentation.append(x)
                segmentation.append(y)
            annotation = {
                "id": annotation_id,
                "image_id": id + 1,
                "category_id": categories.index(row["class"]) + 1,
                "iscrowd": 0,
                "area": area,
                "bbox": bbox,
                "segmentation": [segmentation],
                "width": row["width"],  # WIDTH CHECKEAR
                "height": row["height"]
            }
            coco_format["annotations"].append(annotation)
            annotation_id += 1

    return coco_format

### Function extract_coco_data
def extract_coco_data(coco_data, class_name):
    """
    Extracts image names and polygons for a specified class from a COCO format JSON file.

    Parameters:
    coco_data (dict): dictionary with the coco format for the images.
    class_name (str): The name of the class to filter annotations.

    Returns:
    dict: A dictionary where keys are image names and values are lists of polygons.
    """

    # Extract categories to find the category id for the specified class
    category_id = None
    for category in coco_data.get("categories", []):
        if category["name"] == class_name:
            category_id = category["id"]
            break

    # Raise an error if the class name is not found
    if category_id is None:
        raise ValueError( f"Class \"{class_name}\" not found in categories.")

    # Create a dictionary to map image ids to image names
    image_id_to_name = {image["id"]: image["file_name"] for image in coco_data.get("images", [])}

    # Create a dictionary to store the results
    results = {}

    # Extract annotations for the specified class
    for annotation in coco_data.get("annotations", []):
        if annotation["category_id"] == category_id:
            image_id = annotation["image_id"]
            image_name = image_id_to_name.get(image_id, "Unknown")
            polygons = annotation.get("segmentation", [])

            if image_name not in results:
                results[image_name] = []

            results[image_name].extend(polygons)

    return results


### Function plot_photo_df
def plot_photo_df(image_path, image_name, data_class):
    """
    Plots the specified image with polygons and bounding boxes using the provided data dictionary.

    Parameters:
    image_path (str): Path to the directory containing the images.
    image_name (str): Name of the image file to be plotted.
    data_class (dict): Dictionary containing polygon and bounding box data.
    """
    # Construct the full image path
    full_image_path = os.path.join(image_path, image_name)

    # Read the image
    image = cv2.imread(full_image_path)

    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Check if the image name exists in the data dictionary
    if image_name not in data_class:
        print(f"Image \"{image_name}\" not found in data_class.")
        return

    # Get the polygons from the data dictionary
    polygons = data_class[image_name]

    # Iterate through each polygon and plot it
    for polygon in polygons:
        x_coords = []
        y_coords = []

        # Separate x and y coordinates
        for i in range(0, len(polygon), 2):
            x_coords.append(polygon[i])
            y_coords.append(polygon[i + 1])

        polygon_points = np.array(list(zip(x_coords, y_coords)), np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))
        # Draw polygons
        cv2.polylines(image_rgb, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()


### Function classify_damage_and_update_json
def classify_damage_and_update_json(input_path, json_name, class_name, output_json_path,image_path):
    """
    Classifies damage severity for a specified class in a COCO format JSON file and updates the JSON.

    Parameters:
    input_path (str): Path to the directory containing the input COCO format JSON file.
    json_name (str): Name of the input COCO format JSON file.
    class_name (str): The name of the class to classify damages.
    output_json_path (str): Path to save the updated COCO format JSON file.
    image_path (str): Path to the directory containing the images referenced in the COCO format JSON file.

    Example usage:
        json_name = "annotations.json"
        class_name = "damage"
        output_json_path = "./archive/image/prueba_salida.json"

        classify_damage_and_update_json(train_path_original, json_name, class_name, output_json_path,image_path)
    """
    # Read the COCO format JSON file
    with open(os.path.join(input_path, json_name), "r") as f:
        coco_data = json.load(f)

    # Extract categories to find the category id for the specified class
    category_id = None
    for category in coco_data.get("categories", []):
        if category["name"] == class_name:
            category_id = category["id"]
            break

    # Raise an error if the class name is not found
    if category_id is None:
        raise ValueError(f"Class \"{class_name}\" not found in categories.")

    # Create a dictionary to map image ids to image names
    image_id_to_name = {image["id"]: image["file_name"] for image in coco_data.get("images", [])}

    # Create a new category for each type of damage
    new_categories = []
    for damage_type in ["minor", "medium", "severe"]:
        new_categories.append({
            "supercategory": class_name,
            "id": len(coco_data["categories"]) + len(new_categories) + 1,
            "name": f"{class_name}_{damage_type}"
        })
    # Append new categories to the existing categories in coco_data
    coco_data["categories"].extend(new_categories)

    # Get the new category ids
    new_category_ids = {cat["name"]: cat["id"] for cat in coco_data["categories"] if class_name in cat["name"]}

    # Initialize counters
    total_annotations = len(
        [annotation for annotation in coco_data["annotations"] if annotation["category_id"] == category_id])

    annotations = [ann for ann in coco_data["annotations"] if ann["category_id"] == category_id]

    dropdown = widgets.Dropdown(
        options=[("Select classification", 0), ("Minor", 1), ("Medium", 2), ("Severe", 3)],
        description="Damage:",
        disabled=False,
    )

    next_button = widgets.Button(description="Next")
    output = widgets.Output()

    processed_annotations = 0

    def on_next_button_clicked(b):
        nonlocal processed_annotations
        damage_class = dropdown.value

        if damage_class not in [1, 2, 3]:
            print("Please select a valid damage classification before proceeding.")
            return

        # Map user input to new category name
        if damage_class == 1:
            new_category_name = f"{class_name}_minor"
        elif damage_class == 2:
            new_category_name = f"{class_name}_medium"
        elif damage_class == 3:
            new_category_name = f"{class_name}_severe"

        # Update the annotation with the new category id
        annotation = annotations[processed_annotations]
        annotation["category_id"] = new_category_ids[new_category_name]

        processed_annotations += 1

        # Check if there are more annotations to process
        if processed_annotations < total_annotations:
            display_next_image()
        else:
            save_and_exit()

    def display_next_image():
        clear_output(wait=True)
        annotation = annotations[processed_annotations]
        image_id = annotation["image_id"]
        image_name = image_id_to_name.get(image_id, "Unknown")
        polygons = annotation.get("segmentation", [])

        # Read the image
        full_image_path = os.path.join(image_path, image_name)
        image = cv2.imread(full_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw polygons
        for polygon in polygons:
            x_coords = []
            y_coords = []

            # Separate x and y coordinates
            for i in range(0, len(polygon), 2):
                x_coords.append(polygon[i])
                y_coords.append(polygon[i + 1])

            polygon_points = np.array(list(zip(x_coords, y_coords)), np.int32)
            polygon_points = polygon_points.reshape((-1, 1, 2))

            cv2.polylines(image_rgb, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Display the image
        plt.figure(figsize=(7, 7))
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.title(f"{processed_annotations + 1} of {total_annotations} annotations")
        plt.show()

        # Reset dropdown for next image
        dropdown.value = 0

        # Display the dropdown and buttons
        display(dropdown, next_button, output)

    def save_and_exit():
        # Remove the old category from coco_data["categories"]
        coco_data["categories"] = [cat for cat in coco_data["categories"] if cat["id"] != category_id]

        # Save the updated JSON file with indentation
        with open(output_json_path, "w") as f:
            json.dump(coco_data, f, indent=4)

        print(f"Updated JSON saved to {output_json_path}")

    # Assign the button click event handlers
    next_button.on_click(on_next_button_clicked)

    # Start processing the first image
    if annotations and processed_annotations < total_annotations:
        display_next_image()
    else:
        print("No annotations found for the specified class.")





### Function count_category_occurrences
def count_category_occurrences(coco_json_path):
    """
    Count the number of times of each oft he categories of a Coco format appears in the annotations part

    Args:
        coco_jso _path (str): path of the annotations file.


    Outputs:
        category_counts (dict) : dictionary with the name of the categories and the number of times that appears each one.
    """
    # Leer el archivo JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Crear un diccionario para mapear ids de categorías a nombres
    category_id_to_name = {category['id']: category['name'] for category in coco_data['categories']}

    # Inicializar un diccionario para contar las apariciones de cada categoría
    category_counts = {category['name']: 0 for category in coco_data['categories']}

    # Contar las apariciones de cada categoría en las anotaciones
    for annotation in coco_data['annotations']:
        category_id = annotation['category_id']
        category_name = category_id_to_name[category_id]
        category_counts[category_name] += 1

    return category_counts


################################################
############ 2 Claim Costs Functions ###########
################################################

def truncated_gamma(shape, scale, lower, upper, size):
    """
    Generates samples from a truncated gamma distribution within specified bounds.

    Parameters:
    shape (float): The shape parameter of the gamma distribution.
    scale (float): The scale parameter of the gamma distribution.
    lower (float): The lower bound for truncation.
    upper (float): The upper bound for truncation.
    size (int): The number of samples to generate.

    Returns:
    np.ndarray: An array of samples from the truncated gamma distribution.
    """
    samples = []
    while len(samples) < size:
        sample = gamma(shape, scale=scale).rvs(size)
        sample = sample[(sample >= lower) & (sample <= upper)]
        samples.extend(sample)
    return np.array(samples[:size])


def exponential_dist(lmbda, size=1,ini=0.5):
    """
    Simulates an exponential distribution with a specified lambda value and initial offset.

    Parameters:
    lmbda (float): The rate parameter (lambda) of the exponential distribution.
    size (int): The number of samples to generate. Default is 1.
    ini (float): An initial offset added to each sample. Default is 0.5.

    Returns:
    np.ndarray: An array of samples from the exponential distribution with the initial offset applied.
    """
    return ini+np.random.exponential(1/lmbda, size)


def simulate_data(n=5000, damage_type="glass_crack", low_trunc=100, high_trunc=400,
                  brands={"Toyota": 1, "Volkswagen": 1.25},
                  models={"Yaris": 0.75, "Corolla": 1, "C-HR": 1.20, "Polo": 0.73, "Golf": 1, "Tiguan": 1.45},
                  model_brand={"Yaris": "Toyota", "Corolla": "Toyota", "C-HR": "Toyota", "Polo": "Volkswagen",
                               "Golf": "Volkswagen", "Tiguan": "Volkswagen"},
                  age_range={"Newer": 0.8, "Middle": 1, "Old": 0.5},
                  model_effect=1,
                  age_effect=1):
    """
    Simulates repair cost data for various car brands and models, considering different vehicle age ranges.

    Parameters:
    n (int): The number of samples to generate for each combination of brand, model, and age range.
    damage_type (str): The type of damage being simulated.
    low_trunc (float): The lower bound for truncating the gamma distribution.
    high_trunc (float): The upper bound for truncating the gamma distribution.
    brands (dict): A dictionary with car brands as keys and brand-specific multipliers as values.
    models (dict): A dictionary with car models as keys and model-specific multipliers as values.
    model_brand (dict): A dictionary mapping each model to its corresponding brand.
    age_range (dict): A dictionary with age ranges as keys and their specific multipliers as values.
    model_effect (float): A multiplier applied to the model-specific cost.
    age_effect (float): A multiplier applied to the age-specific cost.

    Returns:
    DataFrame: A DataFrame containing the simulated data with columns for brand, model, vehicle age, age range, damage type, and cost.
    """

    gamma_shape_base = 1.5
    gamma_scale_base = 100

    df_combined = pd.DataFrame()

    for brand, brand_multiplier in brands.items():
        for model, model_multiplier in models.items():
            if model_brand[model] != brand:
                continue

            # Lower age case
            veh_age_lower = np.random.randint(0, 8, size=n)  # 0 to 7
            cost_lower = truncated_gamma(
                gamma_shape_base,
                gamma_scale_base * age_range["Newer"] * model_multiplier * brand_multiplier * model_effect,
                low_trunc * model_multiplier * model_effect * brand_multiplier * age_range["Newer"] * age_effect,
                high_trunc * model_multiplier * model_effect * brand_multiplier * age_range["Newer"] * age_effect,
                n)

            # Base case
            veh_age_base = np.random.randint(7, 11, size=n)  # 7 to 10
            cost_base = truncated_gamma(
                gamma_shape_base,
                gamma_scale_base * age_range["Middle"] * model_multiplier * brand_multiplier * model_effect,
                low_trunc * model_multiplier * model_effect * brand_multiplier * age_range["Middle"],
                high_trunc * model_multiplier * model_effect * brand_multiplier * age_range["Middle"],
                n)

            # Higher age case
            veh_age_higher = np.random.randint(10, 16, size=n)  # 10 to 15
            cost_higher = truncated_gamma(
                gamma_shape_base,
                gamma_scale_base * age_range["Old"] * model_multiplier * brand_multiplier * model_effect,
                low_trunc * model_multiplier * model_effect * brand_multiplier * age_range["Old"] * age_effect,
                high_trunc * model_multiplier * model_effect * brand_multiplier * age_range["Old"] * age_effect,
                n)

            # Create DataFrames
            df_lower = pd.DataFrame({
                "brand": [brand] * n,
                "model": [model] * n,
                "veh_age": veh_age_lower,
                "veh_age_range": "Newer",
                "damage_type": [damage_type] * n,
                "cost": cost_lower
            })

            df_base = pd.DataFrame({
                "brand": [brand] * n,
                "model": [model] * n,
                "veh_age": veh_age_base,
                "veh_age_range": "Middle",
                "damage_type": [damage_type] * n,
                "cost": cost_base
            })

            df_higher = pd.DataFrame({
                "brand": [brand] * n,
                "model": [model] * n,
                "veh_age": veh_age_higher,
                "veh_age_range": "Old",
                "damage_type": [damage_type] * n,
                "cost": cost_higher
            })

            # Combine DataFrames for the current model
            df_model_combined = pd.concat([df_lower, df_base, df_higher], ignore_index=True)
            df_combined = pd.concat([df_combined, df_model_combined], ignore_index=True)

    # Shuffle the combined DataFrame
    df_combined = df_combined.sample(frac=1).reset_index(drop=True)

    return df_combined


# we create a table with the Irish counties real population proportions and their salary
Ireland_av_salary = 41736

data = [
        ('Dublin', 28.7, 46138),
        ('Cork', 11.4, 42766),
        ('Galway', 5.3, 40974),
        ('Limerick', 4.5, 40721),
        ('Kildare', 4.2, 45179),
        ('Tipperary', 3.6, 37961),
        ('Donegal', 3.5, 33526),
        ('Meath', 3.4, 43563),
        ('Kerry', 3.4, 36309),
        ('Mayo', 3, 37750),
        ('Wexford', 3, 36471),
        ('Wicklow', 2.9, 43140),
        ('Clare', 2.6, 40663),
        ('Louth', 2.6, 38686),
        ('Waterford', 2.6, 39439),
        ('Kilkenny', 2.1, 40024),
        ('Westmeath', 1.8, 40079),
        ('Offaly', 1.6, 38238),
        ('Laois', 1.5, 40307),
        ('Sligo', 1.5, 39444),
        ('Cavan', 1.4, 37236),
        ('Roscommon', 1.4, 39060),
        ('Monaghan', 1.3, 34718),
        ('Carlow', 1.2, 37809),
        ('Longford', 0.8, 36180),
        ('Leitrim', 0.7, 37715)
]

# we calculate the factors of the counties
county_info = pd.DataFrame(data, columns=['County', 'prob_dist','salary'])
county_info["factor"] = np.round(county_info.salary/Ireland_av_salary,1)
county_info.drop('salary', axis=1, inplace=True)

def simulate_labour(n=90000, lmbda=2, ini=0.5,
                    workshop_qlty_factor={"Low": 0.7,"Medium": 1,"High": 1.7},
                    workshop_qlty_dist={"Low": 0.3,"Medium": 0.5,"High": 0.2},
                    county_info=county_info,
                    av_lab_ph=60):
    """
    Simulates labour costs for a given number of simulations and lambda value.

    Parameters:
    n (int): The number of simulations to run. Default is 90,000.
    lmbda (float): The rate parameter (lambda) of the exponential distribution for hours worked. Default is 2.
    ini (float): An initial offset added to each sample of hours worked. As such, the minimum is half an hour. Default is 0.5.

    Returns:
    DataFrame: A DataFrame containing the simulated data with columns for average labour cost per hour,
               workshop factors, workshop quality, county factors, counties, labour costs, number of hours worked and total labour costs.
    """

    # Simulate the number of hours worked
    num_hours = exponential_dist(lmbda, size=n, ini=ini)

    # Simulate the workshop quality factor
    workshop_quality = np.random.choice(list(workshop_qlty_factor.keys()), size=n, p=list(workshop_qlty_dist.values()))
    workshop_factors = np.array([workshop_qlty_factor[quality] for quality in workshop_quality])

    # Simulate the county factor
    counties = np.random.choice(county_info['County'], size=n, p=county_info['prob_dist']/100)
    county_factors = county_info.set_index('County').loc[counties, 'factor'].values

    # Calculate the labour cost
    labour_costs = av_lab_ph * workshop_factors * county_factors
    total_labour_costs = labour_costs * num_hours

    # Create a DataFrame with all the traces
    df_labour = pd.DataFrame({
        'av_labour_per_hour': av_lab_ph,
        'workshop_factors': workshop_factors,
        'workshop_quality': workshop_quality,
        'county_factors': county_factors,
        'counties': counties,
        'labour_costs': labour_costs,
        'num_hours': num_hours,
        'total_labour_costs': total_labour_costs
    })

    return df_labour


def cost_analysis(df_combined, models,
                  model_brand={"Yaris": "Toyota", "Corolla": "Toyota", "C-HR": "Toyota", "Polo": "Volkswagen",
                               "Golf": "Volkswagen", "Tiguan": "Volkswagen"},
                  age=["Newer", "Middle", "Old"],
                  model_palette={"Yaris": "blue", "Corolla": "green", "C-HR": "red", "Polo": "blue",
                                 "Golf": "green", "Tiguan": "red"}):
    """
    Analyzes and visualizes cost data across different vehicle models, brands, and age ranges.

    This function performs the following tasks:
    1. Plots density (including KDE - Kernel Density Estimate) and histogram of costs by vehicle age range for each model, for both Toyota and Volkswagen brands.
    2. Prints summary statistics of costs, grouped by vehicle age range, model, and brand.

    Parameters:
    df_combined (DataFrame): The combined DataFrame containing cost data for various car brands, models, and vehicle age ranges.
    models (dict): A dictionary with car models as keys and model-specific multipliers as values.
    model_brand (dict): A dictionary mapping each model to its corresponding brand. Default maps some Toyota and Volkswagen models.
    age (list): A list of age ranges to consider. Default is ["Newer", "Middle", "Old"].
    model_palette (dict): A dictionary mapping each model to its color for plotting. Default assigns colors to some Toyota and Volkswagen models.

    Returns:
    None
    """

    # Set the dark theme
    sns.set_theme(style="dark")

    # Define a custom color palette
    palette = model_palette

    # Plot density of cost with histogram by veh_age_range for each model
    ordered_models = sorted(models.items(), key=lambda item: item[1])

    for age_type in age:
        fig, axes = plt.subplots(1, 2, figsize=(19, 5), sharey=True)

        for i, brand in enumerate(["Toyota", "Volkswagen"]):
            ax = axes[i]
            brand_models = [model for model in models if model_brand[model] == brand]

            # Plot the histogram
            sns.histplot(
                df_combined[(df_combined['veh_age_range'] == age_type) & (df_combined['brand'] == brand)],
                x="cost", hue="model", bins=50, edgecolor='black', palette=palette, ax=ax,
                stat="density", common_norm=False
            )

            # Add KDE plots with thicker lines
            for model in brand_models:
                sns.kdeplot(
                    df_combined[(df_combined['veh_age_range'] == age_type) & (df_combined['brand'] == brand) & (
                                df_combined['model'] == model)]['cost'],
                    ax=ax, linewidth=3, label=model, color=palette[model]
                )

            ax.set_title(f'Density Plot and Histogram of Cost by Vehicle Age Range for {brand} - {age_type}')
            ax.set_xlabel('Cost')
            ax.set_ylabel('Density')

        plt.tight_layout()
        plt.show()

    # Summary statistics of cost by veh_age_range, model, and brand
    summary_stats = df_combined.groupby(['brand', 'model', 'veh_age_range'])['cost'].describe().drop(
        columns='count').round(0).astype(int)

    # Reorder the index based on the desired order
    summary_stats = summary_stats.reindex(
        [("Toyota", model, age_type) for model in model_brand if model_brand[model] == "Toyota" for age_type in age] +
        [("Volkswagen", model, age_type) for model in model_brand if model_brand[model] == "Volkswagen" for age_type in
         age])

    # Print the formatted and ordered summary statistics
    print("Summary statistics by brand, model, and vehicle age range:")
    print(summary_stats)


def labour_analysis(df_labour, lmbda, ini=0.5,workshop_qlty_dist={"Low": 0.3,"Medium": 0.5,"High": 0.2}):
    """
    Analyzes and plots various aspects of simulated labour cost data.

    This function performs several tasks:
    1. Prints summary statistics for the number of hours worked, labour costs, and total labour costs.
    2. Plots the observed distribution of hours worked against the theoretical shifted exponential distribution.
    3. Plots the distribution of labour costs per hour.
    4. Plots the distribution of total labour costs.
    5. Compares the real and simulated distribution of counties.
    6. Compares the real and simulated distribution of workshop quality.

    Parameters:
    df_labour (DataFrame): The DataFrame containing simulated labour data. It should include columns for:
                           - 'num_hours': Number of hours worked
                           - 'labour_costs': Cost per hour of labour
                           - 'total_labour_costs': Total labour costs
                           - 'counties': County where the work was performed
                           - 'workshop_quality': Quality of the workshop
    lmbda (float): The rate parameter (lambda) of the exponential distribution used to simulate the number of hours worked.
    ini (float): An initial offset added to each sample of hours worked. Default is 0.5.

    Returns:
    None
    """

    num_hours = np.array(df_labour["num_hours"])
    labour_costs = np.array(df_labour["labour_costs"])
    total_labour_costs = np.array(df_labour["total_labour_costs"])
    counties = np.array(df_labour["counties"])
    workshop_quality = np.array(df_labour["workshop_quality"])

    # print all of the statistics
    print(df_labour[["num_hours", "labour_costs", "total_labour_costs"]].describe().round(2).T)

    # Plot 1: Observed hours distribution vs theoretical exponential distribution
    plt.figure(figsize=(12, 3))
    sns.histplot(num_hours, bins=50, kde=True, stat='density', color='blue', label='Observed')
    # Theoretical distribution
    x = np.linspace(0.5, np.max(num_hours), 1000)
    y = lmbda * np.exp(-lmbda * (x - ini)) * (x >= ini)
    plt.plot(x, y, 'r-', lw=2, label='Theoretical')
    plt.title('Observed Hours Distribution vs Theoretical Exponential Distribution')
    plt.xlabel('Hours')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Plot 2: Cost per hour distribution
    plt.figure(figsize=(12, 3))
    bins = np.linspace(0, np.max(labour_costs), 51)
    sns.histplot(labour_costs, bins=bins, kde=True, color='green')
    plt.title('Labour Cost per Hour Distribution')
    plt.xlabel('Cost per Hour')
    plt.ylabel('Frequency')

    # Manually setting the ticks and labels
    bin_edges = np.linspace(0, np.max(labour_costs), 31)  # fewer labels for clarity
    plt.xticks(bin_edges, labels=[f"{edge:.0f}" for edge in bin_edges])

    plt.show()

    # Plot 3: Total labour costs
    plt.figure(figsize=(12, 3))
    bins = np.linspace(0, np.max(total_labour_costs), 51)
    sns.histplot(total_labour_costs, bins=bins, kde=True, color='green')
    plt.title('Total Labour Costs Distribution')
    plt.xlabel('Total Labour Cost')
    plt.ylabel('Frequency')

    # Manually setting the ticks and labels
    bin_edges = np.linspace(0, np.max(total_labour_costs), 31)  # fewer labels for clarity
    plt.xticks(bin_edges, labels=[f"{edge:.0f}" for edge in bin_edges])

    plt.show()

    # Plot 4: County distribution comparison
    simulated_county_dist = pd.Series(counties).value_counts(normalize=True).sort_index() * 100
    county_df = pd.DataFrame({
        'County': county_info['County'],
        'Real Distribution': county_info['prob_dist'],
        'Simulated Distribution': simulated_county_dist.reindex(county_info['County']).values
    })
    county_df = county_df.melt(id_vars='County', var_name='Distribution Type', value_name='Percentage')
    plt.figure(figsize=(12, 3))
    sns.barplot(x='County', y='Percentage', hue='Distribution Type', data=county_df)
    plt.xticks(rotation=90)
    plt.title('County Distribution: Real vs Simulated')
    plt.xlabel('County')
    plt.ylabel('Percentage')
    plt.legend()
    plt.show()

    # Plot 5: Workshop quality distribution comparison
    simulated_workshop_dist = pd.Series(workshop_quality).value_counts(normalize=True).sort_index() * 100
    workshop_df = pd.DataFrame({
        'Workshop Quality': list(workshop_qlty_dist.keys()),
        'Real Distribution': np.array(list(workshop_qlty_dist.values())) * 100,
        'Simulated Distribution': simulated_workshop_dist.reindex(list(workshop_qlty_dist.keys())).values
    })
    workshop_df = workshop_df.melt(id_vars='Workshop Quality', var_name='Distribution Type', value_name='Percentage')
    plt.figure(figsize=(12, 3))
    sns.barplot(x='Workshop Quality', y='Percentage', hue='Distribution Type', data=workshop_df)
    plt.title('Workshop Quality Distribution: Real vs Simulated')
    plt.xlabel('Workshop Quality')
    plt.ylabel('Percentage')
    plt.legend()
    plt.show()


def conc_total_cost(df_repair, df_labour):
    """
    Concatenate repair and labour DataFrames, rename and calculate cost columns,
    drop unnecessary columns, and reorder variables.

    Parameters:
    df_repair (DataFrame): DataFrame containing repair costs and other details.
    df_labour (DataFrame): DataFrame containing labour costs and other details.

    Returns:
    DataFrame: Concatenated DataFrame with calculated total costs and reordered columns.
    """
    # Select the relevant columns from df_labour
    labour_columns = df_labour[["workshop_quality", "counties", "total_labour_costs"]]

    # Concatenate the DataFrames
    df_total = pd.concat([df_repair, labour_columns], axis=1)

    # Rename the column 'cost' to 'total_repair_costs'
    df_total = df_total.rename(columns={"cost": "repair_cost",
                                        "total_labour_costs": "labour_cost"})

    # Create the 'total_costs' column
    df_total["total_cost"] = df_total["repair_cost"].round(2) + df_total["labour_cost"].round(2)

    # Drop the 'veh_age' column
    df_total = df_total.drop(columns=["veh_age"])

    # Reorder variables
    desired_order = ["brand", "model", "veh_age_range", "workshop_quality", "counties", "damage_type", "repair_cost",
                     "labour_cost", "total_cost"]
    df_total = df_total[desired_order]
    df_total[["repair_cost", "labour_cost"]] = df_total[["repair_cost", "labour_cost"]].round(2)

    return df_total


def analysis_total_cost(df):
    """
    Analyzes and plots various aspects of total cost data, including distributions and categorical comparisons.

    This function performs the following analyses:
    1. Plots the distribution of total costs using a histogram and KDE (Kernel Density Estimate).
    2. Plots the distribution of repair costs and labour costs in separate subplots.
    3. For each categorical variable (brand, model, vehicle age range, workshop quality, counties):
       - Plots a boxplot of total cost by the categorical variable.
       - Plots a stacked bar plot of average repair cost and labour cost by the categorical variable.
       - Annotates the bar plots with the average total cost on top of the bars and the percentage contribution of repair and labour costs inside the bars.

    Parameters:
    df (DataFrame): The DataFrame containing total cost data, with columns for total cost, repair cost, labour cost,
                    and various categorical variables such as brand, model, vehicle age range, workshop quality, and counties.

    Returns:
    None
    """

    # Plot distribution of the cost variables
    plt.figure(figsize=(18, 4))
    bins = np.linspace(0, np.max(df["total_cost"]), 51)
    sns.histplot(df["total_cost"], bins=bins, kde=True, color='blue')
    plt.title('Total Cost Distribution')
    plt.xlabel('Total Cost')
    plt.ylabel('Frequency')

    plt.show()

    plt.figure(figsize=(18, 4))

    # Distribution of repair_cost
    plt.subplot(1, 2, 1)
    bins = np.linspace(0, np.max(df["repair_cost"]), 51)
    sns.histplot(df["repair_cost"], bins=bins, kde=True, color='orange')
    plt.title('Repair Cost Distribution')
    plt.xlabel('Repair Cost')
    plt.ylabel('Frequency')

    # Distribution of labour_cost
    plt.subplot(1, 2, 2)
    bins = np.linspace(0, np.max(df["labour_cost"]), 51)
    sns.histplot(df["labour_cost"], bins=bins, kde=True, color='green')
    plt.title('Labour Cost Distribution')
    plt.xlabel('Labour Cost')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Categorical variables
    categorical_vars = ["brand", "model", "veh_age_range", "workshop_quality", "counties"]

    for var in categorical_vars:
        plt.figure(figsize=(18, 6))

        # Boxplot of total_cost by categorical variable
        plt.subplot(1, 2, 1)
        sns.boxplot(x=var, y="total_cost", data=df)
        plt.title(f'Total Cost by {var}')
        plt.yticks(fontsize=18)
        plt.xticks(rotation=90, fontsize=18, fontweight='bold')

        # Bar plot of average total_cost by categorical variable with repair_cost and labour_cost stacked
        plt.subplot(1, 2, 2)
        avg_cost = df.groupby(var)[["repair_cost", "labour_cost"]].mean().reset_index()
        avg_cost["total_cost"] = avg_cost["repair_cost"] + avg_cost["labour_cost"]

        # Plot stacked bars
        ax = avg_cost.set_index(var).sort_values(by="total_cost", ascending=False)[["repair_cost", "labour_cost"]].plot(
            kind='bar', stacked=True, color=['orange', 'green'], ax=plt.gca())

        plt.title(f'Average Total Cost by {var}')
        plt.xlabel(var)
        plt.ylabel('Cost')
        plt.yticks(fontsize=18)
        plt.xticks(rotation=90, fontsize=18, fontweight='bold')
        plt.legend(title='Cost Type')

        plt.tight_layout()
        plt.show()

##################################################
########## 3 Data Augmentation Functions #########
##################################################

def merge_coco_annotations(file1, file2, output_file):
    """
    Merge two coco json files ensuring unique ids for images and annotations.

    Args:
        file1 (str): Path of one of the annotations (reference file).
        file2 (str): Path of the other annotations.
        output_file (str): Path of the merged annotations.
    """
    # open json files
    with open(file1, "r") as f:
        data1 = json.load(f)
    with open(file2, "r") as f:
        data2 = json.load(f)

    combined = {
        "info": data1["info"],
        "licenses": data1["licenses"],
        "images": [],
        "annotations": [],
        "categories": data1["categories"]
    }

    # Get the current max IDs to avoid collisions
    max_image_id = max(image["id"] for image in data1["images"]) if data1["images"] else 0
    max_annotation_id = max(ann["id"] for ann in data1["annotations"]) if data1["annotations"] else 0

    # Add data1 to combined
    combined["images"].extend(data1["images"])
    combined["annotations"].extend(data1["annotations"])

    # Update ids for data2
    def update_ids(data, max_image_id, max_annotation_id):
        image_id_map = {}
        for image in data["images"]:
            new_image_id = max_image_id + 1
            image_id_map[image["id"]] = new_image_id
            image["id"] = new_image_id
            max_image_id += 1

        for annotation in data["annotations"]:
            annotation["id"] = max_annotation_id + 1
            annotation["image_id"] = image_id_map[annotation["image_id"]]
            max_annotation_id += 1

    # Update the indexes of the second json and add to the combined json
    update_ids(data2, max_image_id, max_annotation_id)

    combined["images"].extend(data2["images"])
    combined["annotations"].extend(data2["annotations"])
    print(combined["images"])
    print(len(data1["images"]))
    print(len(data2["images"]))
    print(combined["annotations"])
    print(len(data1["annotations"]))
    print(len(data2["annotations"]))

    # Save the merged json
    with open(output_file, "w") as f:
        json.dump(combined, f, indent=4)



def copy_or_move_coco_images(coco_annotation_file, source_images_folder, destination_images_folder, operation="copy"):
    """
    Copy or move images specified in a COCO annotation file from a source folder to a destination folder.

    This function reads a COCO (Common Objects in Context) annotation file to get the list of image file names
    and either copies or moves those images from the source folder to the destination folder based on the operation
    parameter. If the destination folder does not exist, it will be created.

    Parameters:
    coco_annotation_file (str): Path to the COCO annotation JSON file.
    source_images_folder (str): Path to the source folder containing the images.
    destination_images_folder (str): Path to the destination folder where images will be copied or moved.
    operation (str): Operation to perform: "copy" or "move". Default is "copy".

    Returns:
    None

    Example:
    copy_or_move_coco_images(
        coco_annotation_file='annotations/instances_train2017.json',
        source_images_folder='images/train2017',
        destination_images_folder='images/processed_train2017',
        operation='move'
    )

    Notes:
    - The function checks if each image exists in the source folder before copying or moving it.
    - If an image is not found in the source folder, a message is printed.
    - The function uses `shutil.copy` to copy the images and `shutil.move` to move the images.
    """
    # Create destination folder if it does not exist
    os.makedirs(destination_images_folder, exist_ok=True)

    # Load the COCO annotation file
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Create a dictionary to store image_id to file_name mapping
    image_id_to_filename = {image['id']: image['file_name'] for image in coco_data['images']}

    # Process each image
    for image_id, filename in image_id_to_filename.items():
        source_image_path = os.path.join(source_images_folder, filename)
        destination_image_path = os.path.join(destination_images_folder, filename)

        # Copy or move image to destination folder
        if os.path.exists(source_image_path):
            if operation == "copy":
                shutil.copy(source_image_path, destination_image_path)
            elif operation == "move":
                shutil.move(source_image_path, destination_image_path)
            else:
                raise ValueError("Operation must be 'copy' or 'move'.")
        else:
            print(f"Image {filename} not found in source folder.")


def image_coco_plot(path, annotation_json, image_name):
    """
    Display a photo and its polygons from a coco format json

    Args:
        path (str): path of the folder with the images and the json with the annotations.
        annotations_json (str): path  of the file of the annotations.
        image_name (str): name of the photo to display.
    """

    # Load the annotations JSON file
    annotations_path = annotation_json
    with open(annotations_path, "r") as file:
        annotations = json.load(file)

    # Extract the information for the specific image
    image_filename = image_name
    image_info = None
    for image in annotations["images"]:
        if image["file_name"] == image_filename:
            image_info = image
            break

    if image_info is None:
        raise ValueError(f"Image {image_filename} not found in annotations.")

    image_id = image_info["id"]  # take the id for the annotations part

    # Extract annotations for the image
    image_annotations = [ann for ann in annotations["annotations"] if ann["image_id"] == image_id]

    # Load the image
    image_path = os.path.join(path,image_filename)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw the polygons and rectangles
    for ann in image_annotations:
        if "segmentation" in ann:
            x_coords = []
            y_coords = []

            # Separar las coordenadas en x e y
            for i in range(0, len(ann["segmentation"][0]), 2):
                x_coords.append(ann["segmentation"][0][i])
                y_coords.append(ann["segmentation"][0][i + 1])

            polygon_points = np.array(list(zip(x_coords, y_coords)), np.int32)
            polygon_points = polygon_points.reshape((-1, 1, 2))
            # Dibujar el polígono
            cv2.polylines(image_rgb, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw the bounding box (if any)
        if "bbox" in ann:
            bbox = ann["bbox"]
            # rectangle coordiantes
            x_rect = int(ann["bbox"][0])
            y_rect = int(ann["bbox"][1])
            width = int(ann["bbox"][2])
            height = int(ann["bbox"][3])

            # Dibujar el rectángulo
            cv2.rectangle(image_rgb, (x_rect, y_rect), (x_rect + width, y_rect + height), (255, 0, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis("off")  # Ocultar los ejes
    plt.show()



################################################
##########    4 Yolo Code Functions    #########
################################################

def validate_coco_dataset(coco_annotation_path, images_folder_path):
    """
    Validates a COCO dataset to check if the dimensions, IDs, and naming conventions are correct.

    Parameters:
    coco_annotation_path (str): Path to the COCO annotations JSON file.
    images_folder_path (str): Path to the folder containing images.

    Returns:
    bool: True if the dataset is valid, False otherwise.
    """
    # Load the COCO annotations
    with open(coco_annotation_path, 'r') as file:
        coco_data = json.load(file)

    # Create a set to store unique image IDs
    image_ids = set()

    # Validate images
    for image_info in coco_data['images']:
        image_id = image_info['id']
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']

        # Check if image ID is unique
        if image_id in image_ids:
            print(f"Duplicate image ID found: {image_id}")
            return False
        image_ids.add(image_id)

        # Check if the image file exists
        image_path = os.path.join(os.path.join(images_folder_path), file_name)
        if not os.path.exists(image_path):
            print(f"Image file not found: {file_name}")
            return False

        # Check if the image dimensions match
        with Image.open(image_path) as img:
            if img.width != width or img.height != height:
                print(f"Image dimensions do not match for {file_name}: "
                      f"expected ({width}, {height}), got ({img.width}, {img.height})")
                return False

    # Validate annotations
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in image_ids:
            print(f"Annotation references non-existent image ID: {image_id}")
            return False
    return True


def clamp(value, min_value, max_value):
    """
    Ensures a value is within a specified range.

    Args:
        value (float): The value to adjust.
        min_value (float): The minimum allowable value.
        max_value (float): The maximum allowable value.

    Returns:
        float: The adjusted value within the specified range.
    """
    return max(min(value, max_value), min_value)

def adjust_polygon(polygon, image_width, image_height):
    """
    Adjusts the points of a polygon to ensure they stay within the image boundaries.

    Args:
        polygon (list): List of polygon coordinates in the format [x1, y1, x2, y2, ..., xn, yn].
        image_width (int): The width of the image.
        image_height (int): The height of the image.

    Returns:
        list: The adjusted polygon coordinates.
    """
    adjusted_polygon = []
    for i in range(0, len(polygon), 2):
        x = clamp(polygon[i], 0, image_width - 1)
        y = clamp(polygon[i+1], 0, image_height - 1)
        adjusted_polygon.extend([x, y])
    return adjusted_polygon

def process_coco_annotations(coco_annotations_path, output_path):
    """
    Processes COCO annotations to adjust polygons to stay within image boundaries.

    Args:
        coco_annotations_path (str): The file path to the input COCO annotations JSON file.
        output_path (str): The file path to save the adjusted COCO annotations JSON file.
    """
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    for annotation in coco_data['annotations']:
        segmentation = annotation['segmentation']
        for i, polygon in enumerate(segmentation):
            # Assuming the segmentation is in the format [x1, y1, x2, y2, ..., xn, yn]
            if len(polygon) % 2 != 0:
                print(f"Warning: Polygon with odd number of coordinates found in annotation ID {annotation['id']}")
                continue
            
            image_id = annotation['image_id']
            image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
            image_width = image_info['width']
            image_height = image_info['height']
            
            segmentation[i] = adjust_polygon(polygon, image_width, image_height)
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

def copy_json_if_not_exists(source, destination):
    """
    Copies a JSON file from the source to the destination if it doesn't already exist at the destination.

    Args:
    source: the file path of the source JSON file
    destination: the directory path where the JSON file should be copied to
    """
    # Get the file name from the source path
    file_name = os.path.basename(source)
    
    # Create the full path for the file in the destination directory
    destination_full_path = os.path.join(destination, file_name)
    
    # Check if the file already exists in the destination directory
    if not os.path.exists(destination_full_path):
        # Copy the file if it does not exist
        shutil.copy2(source, destination_full_path)

def move_and_rename_folder(source_folder, destination_folder):
    """
    Moves the contents of the source folder to the destination folder without moving the folder itself.

    Args:
    source_folder: the directory path of the source folder
    destination_folder: the directory path of the destination folder
    """
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Move each file and folder within the source folder to the destination folder
    for item in os.listdir(source_folder):
        source_item = os.path.join(source_folder, item)
        destination_item = os.path.join(destination_folder, item)
        
        # Move the item to the destination
        shutil.move(source_item, destination_item)

def convert_coco_to_yolo_segmentation(path_annotations, annotations_name, path_yolo, type_set):
    """
    Converts COCO annotations to YOLO format, moves and renames the resulting labels, and cleans up.

    Args:
    path_annotations: the directory path where the source annotations are located
    annotations_name: the file name of the source JSON file containing the annotations
    path_yolo: the base directory path for YOLO formatted data
    type_set: indicating if it is train, val or test
    """

    # to convert to coco to yolo, it is use the convert_coco function and this one has as input variables
    # the path where the annotations of the images are, that is why the annotations of the set is temporary copied
    # in the yolo set folder and then eliminated because there can be only one annotations file in the indicated folder
    # Copy JSON if it doesn't exist
    copy_json_if_not_exists(
        source=os.path.join(path_annotations, annotations_name), 
        destination=os.path.join(path_yolo, type_set)
    )

    # Convert COCO to YOLO labels
    # this function is from annalytics and it will create in the folder aux a folder called labels, that one, inside, has
    # another folder with the name of the annotations file with all the .txt of the images
    convert_coco(
        labels_dir=os.path.join(path_yolo, type_set), 
        save_dir=os.path.join(path_yolo, 'aux'), 
        use_segments=True
    )
    # the  labels created in aux are moved to the labels of the corresponding folder
    move_and_rename_folder(
        source_folder=os.path.join(os.path.join(path_yolo, 'aux/labels/'), os.path.splitext(annotations_name)[0]),
        destination_folder=os.path.join(path_yolo, type_set, 'labels')
    )
    
    # Remove the auxiliary directory
    shutil.rmtree(os.path.join(path_yolo, 'aux'))
    os.remove(os.path.join(os.path.join(path_yolo, type_set), annotations_name))

    print(f"Removed auxiliary directory {os.path.join(path_yolo, 'aux')}")
    
    # Print the final location of YOLO labels
    print(f"Yolo labels saved in {os.path.join(path_yolo, type_set, 'labels')}\n")
    
def create_yaml_file(file_path, train_path, val_path, nc, names):
    """
    Create yaml for yolo.

        Args:
            file_path: path of the yaml
            train_path: path of the train set
            val_path: path of the val set
            nc: number of categories
            names: names of the categories
    """
    data = {
            'train': train_path,
            'val': val_path,
            'nc': nc,
            'names': names
    }
    
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


def select_images(source_folder, destination_folder, num_images, images_folder, labels_folder, seed):
    """
    Selects a random subset of images and their corresponding labels from a source folder and copies them to a destination folder.

    Args:
        source_folder (str): Path to the source folder containing images and labels.
        destination_folder (str): Path to the destination folder where selected images and labels will be copied.
        num_images (int): Number of images to select.
        images_folder (str): Name of the folder within the source folder that contains the images.
        labels_folder (str): Name of the folder within the source folder that contains the labels.
        seed (int): seed set 

    Example:
        select_images(
            source_folder='./Data/Yoloimages/train',
            destination_folder='./Data/Yoloimages/train_prueba',
            num_images=20,
            images_folder='images',
            labels_folder='labels'
        )

        This will select 20 random images from './Data/Yoloimages/train/images' and their corresponding labels
        from './Data/Yoloimages/train/labels', then copy them to './Data/Yoloimages/train_prueba/images' and
        './Data/Yoloimages/train_prueba/labels' respectively. If the destination folder already exists, it will be
        deleted and recreated.
    """
    random.seed(seed)
    # Delete the destination folder if it already exists
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    
    # Create the full paths for the images and labels folders
    images_folder_path = os.path.join(source_folder, images_folder)
    labels_folder_path = os.path.join(source_folder, labels_folder)

    destination_images_folder = os.path.join(destination_folder, images_folder)
    destination_labels_folder = os.path.join(destination_folder, labels_folder)

    # Create the destination folders
    os.makedirs(destination_images_folder)
    os.makedirs(destination_labels_folder)

    # Get the list of images and select a random subset
    images = os.listdir(images_folder_path)
    selected_images = random.sample(images, num_images)

    # Copy the images and their corresponding labels
    for image in selected_images:
        image_name = os.path.splitext(image)[0]
        source_image_path = os.path.join(images_folder_path, image)
        destination_image_path = os.path.join(destination_images_folder, image)
        
        source_label_path = os.path.join(labels_folder_path, f"{image_name}.txt")
        destination_label_path = os.path.join(destination_labels_folder, f"{image_name}.txt")
        
        shutil.copy(source_image_path, destination_image_path)
        if os.path.exists(source_label_path):
            shutil.copy(source_label_path, destination_label_path)



def print_styled_metrics_table(metrics, names, color):
    """
    Prints a table with the class names and the corresponding mAP50, precision, recall, and F1 score
    values for boxes and segmentation using Pandas Styler for custom formatting.

    Parameters:
    - metrics (ultralytics.utils.metrics.SegmentMetrics): A metrics object from a YOLO model's validation or testing process that contains
               the mAP, precision, recall, and F1 score values for segmentation and bounding boxes.
    - names (str): dict with the names of the classes
    - color (str): color to use.
    """

    # Extract class names
    class_names = [names[i] for i in range(len(names))]
    
    # Extract mAP50, precision, recall, and F1 score values for boxes and segmentation
    box_ap5095 = metrics.box.ap
    seg_ap5095 = metrics.seg.ap
    seg_ap50 = metrics.seg.ap50
    box_ap50 = metrics.box.ap50
    box_precision = metrics.box.p
    seg_precision = metrics.seg.p
    box_recall = metrics.box.r
    seg_recall = metrics.seg.r
    box_f1 = metrics.box.f1
    seg_f1 = metrics.seg.f1

    # Create a DataFrame
    df = pd.DataFrame({
        "Class Name": class_names,
        "Box Precision": box_precision,
        "Box Recall": box_recall,
        "Box F1 Score": box_f1,        
        "Box AP50": box_ap50,
        "Box AP50-95": box_ap5095,
        "Segmentation Precision": seg_precision,
        "Segmentation Recall": seg_recall,
        "Segmentation F1 Score": seg_f1,
        "Segmentation AP50": seg_ap50,
        "Segmentation AP50-95": seg_ap5095        
    })

    all_results = pd.DataFrame({
    "Class Name": ['all'],
    "Box Precision": [metrics.box.mp],
    "Box Recall": [metrics.box.mr],
    "Box F1 Score": [2 * metrics.box.mr * metrics.box.mp / (metrics.box.mp + metrics.box.mr)],
    "Box AP50": [metrics.box.map50],
    "Box AP50-95": [metrics.box.map],    
    "Segmentation Precision": [metrics.seg.mp],
    "Segmentation Recall": [metrics.seg.mr],
    "Segmentation F1 Score": [2 * metrics.seg.mr * metrics.seg.mp / (metrics.seg.mp + metrics.seg.mr)],
    "Segmentation AP50": [metrics.seg.map50],
    "Segmentation AP50-95": [metrics.seg.map]
    })

    df = pd.concat([df, all_results], ignore_index=True)
    
    # Apply custom styling
    styled_df = df.style.set_table_styles(
        [
            {'selector': 'thead th', 'props': [('background-color', color), ('color', 'white'), ('text-align', 'center')]},
            {'selector': 'tbody td', 'props': [('text-align', 'center'), ('border', '1px solid black')]},
            {'selector': 'tbody th', 'props': [('background-color', color), ('color', 'white'), ('text-align', 'center')]},
            {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%')]},
        ]
    ).format(precision=4)

    # Display the styled DataFrame
    display(styled_df)

def confusion_matrix_yolo(metrics, names, color1):
    """
    Generates and plots both the non-normalized and normalized confusion matrices side by side
    for a YOLO model's performance metrics. The function takes in the model's metrics and a 
    color choice for customizing the heatmap gradient.

    Parameters:
    - metrics (ultralytics.utils.metrics.SegmentMetrics): A metrics object from a YOLO model's validation or testing process that contains 
               the confusion matrix data.
    - color1 (str): A string representing the color (in hexadecimal or named color format) to be used 
              as the gradient endpoint for the heatmaps.
    """

    # Extract the confusion matrix and class names
    conf_matrix = metrics.confusion_matrix.matrix
    class_names = [names[i] for i in range(len(names))]

    # Normalize the confusion matrix and handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=0)
        conf_matrix_normalized[np.isnan(conf_matrix_normalized)] = 0  # Replace NaNs with 0

    # Define the color palette for the heatmap
    colors = ["white", color1]  # Custom gradient color
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Set up the matplotlib figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot the non-normalized confusion matrix with class names
    sns.heatmap(conf_matrix.astype(int), annot=True, fmt='d', cmap=custom_cmap, cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    # Plot the normalized confusion matrix with class names
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap=custom_cmap, cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')

    # Display the plots
    plt.show()


def plot_losses_side_by_side(data, color_train="#62b6cb", color_val="#fb8500"):
    """
    Plots each loss metric (box, segmentation, classification, DFL) side by side with training on the left
    and validation on the right for each metric.

    Parameters:
    - data (DataFrame): A DataFrame containing the loss metrics for each epoch.
    - color_train (str): Color for training losses.
    - color_val (str): Color for validation losses.
    """

    epochs = data['epoch']

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    # Box Loss
    axes[0, 0].plot(epochs, data['train/box_loss'], label='Train Box Loss', color=color_train)
    axes[0, 0].set_title('Train Box Loss')
    axes[0, 1].plot(epochs, data['val/box_loss'], label='Validation Box Loss', color=color_val)
    axes[0, 1].set_title('Validation Box Loss')

    # Segmentation Loss
    axes[1, 0].plot(epochs, data['train/seg_loss'], label='Train Segmentation Loss', color=color_train)
    axes[1, 0].set_title('Train Segmentation Loss')
    axes[1, 1].plot(epochs, data['val/seg_loss'], label='Validation Segmentation Loss', color=color_val)
    axes[1, 1].set_title('Validation Segmentation Loss')

    # Classification Loss
    axes[2, 0].plot(epochs, data['train/cls_loss'], label='Train Classification Loss', color=color_train)
    axes[2, 0].set_title('Train Classification Loss')
    axes[2, 1].plot(epochs, data['val/cls_loss'], label='Validation Classification Loss', color=color_val)
    axes[2, 1].set_title('Validation Classification Loss')

    # DFL Loss
    axes[3, 0].plot(epochs, data['train/dfl_loss'], label='Train DFL Loss', color=color_train)
    axes[3, 0].set_title('Train DFL Loss')
    axes[3, 1].plot(epochs, data['val/dfl_loss'], label='Validation DFL Loss', color=color_val)
    axes[3, 1].set_title('Validation DFL Loss')

    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def metrics_yolo(model, path_results_yolo, color1, color2):
    """
    This function provides a comprehensive analysis of YOLO model performance by generating a styled table of key metrics,
    visualizing the confusion matrix, and loading the results of the training/validation process from  CSV results file generated by 
    yolo.

    Parameters:
    - metrics: A metrics object from the YOLO model's validation or testing process, containing various performance metrics.
    - path_results_yolo: The file path to the directory where YOLO training/validation results are stored, including the 'results.csv' file.
    - color1: A string representing the color (in hexadecimal or named color format) to be used for certain visualizations, particularly in the styled table and confusion matrix.
    - color2: A string representing an additional color to be used for visualizations if needed.
    """
    metrics = model.metrics
    names = model.names
    print_styled_metrics_table(metrics, names , color1)
    confusion_matrix_yolo(metrics, names, color1)
    
    df_epochs = pd.read_csv(os.path.join(path_results_yolo,"results.csv"))
    df_epochs.columns = df_epochs.columns.str.strip()

    plot_losses_side_by_side(df_epochs, color1, color2)



def upload_folder_to_s3(local_folder, s3_bucket):
    """
    Uploads all files from a local directory to an S3 bucket using the AWS CLI, displaying a progress bar.

    Args:
        local_folder (str): The path to the local directory containing the files to upload.
        s3_bucket (str): The name of the S3 bucket to upload the files to.

    Example:
        >>> upload_folder_to_s3("Data/Yoloimages/train", "sagemaker-eu-west-1-project-danielteresa")
        Uploading Data/Yoloimages/train: 100%|██████████| 100/100 [00:30<00:00, 3.30file/s]
    """
    # Count the total number of files to be uploaded for the progress bar
    total_files = sum([len(files) for r, d, files in os.walk(local_folder)])

    # Run the aws s3 cp command with --recursive option to upload the directory
    with tqdm(total=total_files, desc=f"Uploading {local_folder}", unit="file") as pbar:
        command = f"aws s3 cp {local_folder} s3://{s3_bucket}/ --recursive"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        while True:
            output = process.stdout.readline()
            if process.poll() is not None:
                break
            if output:
                pbar.update(1)

        rc = process.poll()
        return rc

def upload_file_to_s3(local_file, s3_bucket):
    """
    Uploads a single file to an S3 bucket using the AWS CLI.

    Args:
        local_file (str): The path to the local file to upload.
        s3_bucket (str): The name of the S3 bucket to upload the file to.

    Example:
        >>> upload_file_to_s3("./Notebooks/data.yaml", "sagemaker-eu-west-1-project-danielteresa")
    """
    # Construct the S3 path
    s3_path = os.path.basename(local_file)
    command = f"aws s3 cp {local_file} s3://{s3_bucket}/{s3_path}"
    
    # Execute the command
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Check for errors
    if result.returncode != 0:
        print(f"Error uploading {local_file} to {s3_bucket}: {result.stderr.decode('utf-8')}")
    else:
        print(f"Successfully uploaded {local_file} to s3://{s3_bucket}/{s3_path}")





