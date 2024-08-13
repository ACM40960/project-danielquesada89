import streamlit as st
import base64
import os
from ultralytics import YOLO
import joblib
import pandas as pd



#  charge models and obtain the prediciton of the photo
path_upload = 'API/uploads' 
picture_name=os.listdir(path_upload)[0]
full_path= os.path.join(path_upload,picture_name)
# we import both models
yolo_model = YOLO("Models/best_modeltuning_YOLO-tuning8/train/weights/best.pt")
cost_model=joblib.load("Models/cost_model.pkl")

# we predict the image
results = yolo_model.predict(full_path,conf=0.15)

# we save the image
path_output = os.path.join(path_upload, 'pred_'+picture_name)
results[0].save(filename=path_output)






st.set_page_config(
    #page_title="Cellphone App",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide the Streamlit sidebar menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Define the icons paths
icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Icons')

# Function to load an image and convert it to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string


# Image policyholder_person
image_base64 = get_image_base64(os.path.join(icon_path, 'policyholder_person.png'))
image_policyholder_person = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'top: 2%; '
               f'left: 15%; '
               f'width: 40px; '
               f'height: 40px;">')

# Image claimholder_person
image_base64 = get_image_base64(os.path.join(icon_path, 'claimholder_person.png'))
image_claimholder_person = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'top: 2%; '
               f'left: 65%; '
               f'width: 40px; '
               f'height: 40px;">')

# Container of Policy details and Policy id
text_container2 = (f'<div style="position: absolute; '
                  f'top: 3%; '
                  f'left: 220px; '
                  f'width: calc(90% - 300px); '  # Adjust width to account for padding
                  f'display: flex; '
                  f'justify-content: space-between; '
                  f'align-items: center;">'
                  f'<p style="font-family: Arial, sans-serif; '
                  f'font-size: 18px; '
                  f'font-weight: bold; '  # Make text bold
                  f'color: #023047; '
                  f'margin: 0;">Policyholder View</p>'
                  f'<p style="font-family: Arial, sans-serif; '
                  f'font-size: 18px; '
                  f'font-weight: bold; '  # Make text bold
                  f'color: #023047; '
                  f'margin: 0;">Claim Handler View</p>'
                  f'</div>')

# we fix the width and height of the rectangle
width=400*2.9
height1=370*1.6
height2=500*1.3

# Image policy_details2
image_base64 = get_image_base64(os.path.join(icon_path, 'policy_details2.png'))
image_policy_details2 = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'top: 20px; '
               f'left: 20px; '
               f'width: 35px; '
               f'height: 35px;">')

# Container of Policy details and Policy id
text_container = (f'<div style="position: absolute; '
                  f'top: 25px; '
                  f'left: 70px; '
                  f'width: calc(90% - 40px); '  # Adjust width to account for padding
                  f'display: flex; '
                  f'justify-content: space-between; '
                  f'align-items: center;">'
                  f'<p style="font-family: Arial, sans-serif; '
                  f'font-size: 18px; '
                  f'font-weight: bold; '  # Make text bold
                  f'color: white; '
                  f'margin: 0;">File a Claim</p>'
                  f'<p style="font-family: Arial, sans-serif; '
                  f'font-size: 16px; '
                  f'color: white; '
                  f'margin: 0;">Policy ID 4532609</p>'
                  f'</div>')

# Image home
image_base64 = get_image_base64(os.path.join(icon_path, 'home.png'))
image_home = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'bottom: 10px; '
               f'left: 15%; '  # Position to the left
               f'transform: translateX(-50%); '
               f'width: 30px; '
               f'height: 30px;">')

# Image policyholder
image_base64 = get_image_base64(os.path.join(icon_path, 'policyholder.png'))
image_policyholder = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'bottom: 10px; '
               f'left: 50%; '  # Centered horizontally
               f'transform: translateX(-50%); '
               f'width: 30px; '
               f'height: 30px;">')

# Image settings
image_base64 = get_image_base64(os.path.join(icon_path, 'settings.png'))
image_settings = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'bottom: 10px; '
               f'left: 85%; '  # Position to the right
               f'transform: translateX(-50%); '
               f'width: 30px; '
               f'height: 30px;">')

width_policyholder=500


### objects

title_container2 = (f'<div style="position: relative; top: 110px; left: 50px;">'  # Position title within the rectangle
                    f'<p style="font-family: Arial, sans-serif; '
                    f'font-size: 18px; '
                    f'font-weight: bold; '
                    f'color: #023047; '
                    f'display: inline-block; margin-right: 70px;">'  # Spacing to the right of Claim ID
                    f'Claim ID 29382921 '
                    f'</p>'
                    f'<span style="font-family: Arial, sans-serif; '
                    f'font-size: 18px; '
                    f'font-weight: bold; '
                    f'color: #8ED973;">'  # Green color for Approved
                    f'Approved'
                    f'</span>'
                    f'</div>')

text_container3 = (f'<div style="position: relative; top: 105px; left: 50px;">'  # Increased top value for more separation
                  f'<p style="font-family: Arial, sans-serif; '
                  f'font-size: 14px; '
                  f'font-weight: bold; '
                  f'color: #FB8500; '
                  f'display: inline-block; margin-right: 70px;">'  # Spacing to the right of the text
                  f'Thank you for your patience'
                  f'</p>'  # Properly close the paragraph tag
                  f'</div>')


workshop_container = (f'<div style="position: relative; top: 130px; left: 50px;">'
                      f'<p style="font-family: Arial, sans-serif; '
                      f'font-size: 14px; '
                      f'color: #FB8500; '
                      f'margin-bottom: 10px;">'
                      f'Workshop for Vehicle Reparation'
                      f'</p>'
                      f'<select style="font-family: Arial, sans-serif; '
                      f'font-size: 14px; '
                      f'color: #666; '
                      f'padding: 8px 12px; '
                      f'border: 1px solid #ccc; '
                      f'border-radius: 5px; '
                      f'width: 300px;">'
                      f'<option value="" disabled selected style="font-weight: normal;">Select among available workshops</option>'
                      f'<option value="smithfield_autotech">Smithfield Autotech</option>'
                      f'<option value="sandyford_mccann_motor">Sandyford McCann Motor</option>'
                      f'<option value="mobile_mechanic">Mobile Mechanic</option>'
                      f'</select>'
                      f'</div>')

date_picker_container = (f'<div style="position: relative; top: 160px; left: 50px;">'
                         f'<p style="font-family: Arial, sans-serif; '
                         f'font-size: 14px; '
                         f'color: #FB8500; '
                         f'margin-bottom: 10px;">'
                         f'Schedule the appointment'
                         f'</p>'
                         f'<input type="date" style="font-family: Arial, sans-serif; '
                         f'font-size: 14px; '
                         f'color: #666; '
                         f'padding: 8px 12px; '
                         f'border: 1px solid #ccc; '
                         f'border-radius: 5px; '
                         f'width: 300px;" '
                         f'placeholder="Select among available dates">'
                         f'</div>')



########


# white rectangle with centered image
white_rectangle = f"""
<div style="
    width: {width_policyholder}px;
    height: 375px;
    background-color: #FFFFFF;
    position: absolute;
    bottom: 55px;
    left: 50%;
    transform: translateX(-50%);
">
    
</div>
"""


# HTML and CSS to create the main rectangle with button directly in the HTML
rectangle_html3 = f"""
<div style="
    width: {width_policyholder}px;
    height: 500px;
    background-color: #62B6CB;
    margin: 75px 30px;
    box-shadow: 0px 0px 30px 5px #006FAB;
    position: relative;
">
    {image_policy_details2}
    {text_container}
    {image_home}
    {image_policyholder}
    {image_settings}
    {white_rectangle}


</div>
"""

# Date Picker Container
# date_picker_container = (f'<div style="margin-left: 50px;; margin-top: 310px;">'
#                          f'<p style="font-family: Arial, sans-serif; '
#                          f'font-size: 12px; '
#                          f'color: #FF6200;margin-bottom: 6px">Schedule the Appointment</p>'
#                          f'<input type="date" placeholder="Enter the Claim Date" '
#                          f'style="width: 30%; padding: 7px; font-size: 10px; '
#                          f'border-radius: 3px; border: 1px solid "#fb8500";">'

#                          f'</div>')




# outline rectangle with centered image
rectangle_outline_1 = f"""
<div style="
    width: {width_policyholder * 1.07}px;
    height: 530px;
    background-color: #white;
    position: absolute;
    top: 15%;
    left: 24%;
    transform: translateX(-50%);
    box-shadow:
        inset -1px -1px 0px 0px #BFBFBF,   /* Top-left shadow for bevel */
        inset 1px 1px 0px 0px #BFBFBF,    /* Bottom-right shadow for bevel */
        inset 0px 0px 5px 1.5px #BFBFBF;    /* Inner shadow for additional depth */
">
    {title_container2} 
    {text_container3}
    {workshop_container}
    {date_picker_container}


</div>
"""

# rectagule with the details
rectangle2_1 = f"""
<div style="
    width: {width}px;
    height: {height1}px;
    background-color: #FFFFFF;
    position: absolute;
    top: 40px;
    left: 50%;
    transform: translateX(-50%);
">
    {image_policyholder_person}
    {image_claimholder_person}
    {text_container2}
    {rectangle_html3}
</div>
"""

######## Claim handler view 


# HTML and CSS to create the main rectangle
rectangle_html2 = f"""
<div style="
    width: {width}px;
    height: {height2}px;
    background-color: #62B6CB;
    margin: 30px -50px -30px -200px;
    box-shadow: 0px 0px 30px 5px #006FAB;
    position: relative;
">
     
     
     {rectangle2_1}
     {rectangle_outline_1}

</div>
"""
#     {rectangle_outline_2}   

#st.title("Cellphone App")

button_style = """
    <style>
    div.stButton > button {
        color: white;
        background-color: #023047; 
        width: 100px; 
        height: 45px;  
        box-shadow: 0px 0px 30px 5px #006FAB;
        font-size: 12px; /* Adjusted font size */
        padding: 2px; 
        border-radius: 10px; /* Optional: rounded corners */
    }
    </style>
    """

### COSAS PÒR HACER
# CREAR VAIRABLE QUE INDIQUE MODELO A TOMAR
# CREAR VAIRABLE QUE INDIQUE DONDE ESTAN LAS FOTOS


# Display the rectangle immediately after the title
st.markdown(rectangle_html2, unsafe_allow_html=True)



# CSS for the selectbox and button styling
button_style = """
    <style>
    div.stButton > button {
        color: white;
        background-color: #023047; 
        width: 150px; 
        height: 45px;  
        box-shadow: 0px 0px 30px 5px #006FAB;
        font-size: 12px; /* Adjusted font size */
        padding: 2px; 
        border-radius: 10px; /* Optional: rounded corners */
    }
    div.stSelectbox > div > div > select {
        height: 45px;
        padding-left: 10px;
        padding-right: 30px;
        border-radius: 10px;
    }
    </style>
    """

# Use Streamlit's selectbox widget to capture the selection
buttom_workshop = st.selectbox(
        '',
        ["Select among available workshops", "Smithfield Autotech", "Sandyford McCann Motor", "Mobile Mechanic"],
        index=0
)


col1, col2 = st.columns([3, 3])
st.markdown(button_style, unsafe_allow_html=True)

with col2:
    if st.button("Home", key="home"):
        st.switch_page(os.path.join(os.getcwd(), "API/Homepage.py"))


# Display the selected workshop
if buttom_workshop != "Select among available workshops":
    # workshop selection
    workshops={
    'Smithfield Autotech':'Low',
    'Sandyford McCann Motors':'Medium',
    'Mobile Mechanic':'High'
    }

    selected_workshop_qual= workshops[buttom_workshop]
    st.write(selected_workshop_qual)

    # we predict the cost of the claims
    case = pd.DataFrame({
    'const': [1],
    'brand_Volkswagen': [0],
    'model_Corolla': [0],
    'model_Golf': [0],
    'model_Polo': [0],
    'model_Tiguan': [0],
    'model_Yaris': [1],
    'veh_age_range_Newer': [1],
    'veh_age_range_Old': [0],
    'workshop_quality_Low': [0],
    'workshop_quality_Medium': [0],
    'counties_group2': [0],
    'counties_group3': [0],
    'damage_type_met_dent_medium': [0],
    'damage_type_met_dent_minor': [0],
    'damage_type_met_dent_severe': [0],
    'damage_type_met_tear': [0],
    'damage_type_mis_lamp': [0],
    'damage_type_mis_lost': [0],
    'damage_type_mis_punct': [0]
    })


    # Adjust case DataFrame based on selected_workshop_qual
    if selected_workshop_qual == 'High':
        case['workshop_quality_Low'] = [0]
        case['workshop_quality_Medium'] = [0]
    elif selected_workshop_qual == 'Medium':
        case['workshop_quality_Low'] = [0]
        case['workshop_quality_Medium'] = [1]
    elif selected_workshop_qual == 'Low':
        case['workshop_quality_Low'] = [1]
        case['workshop_quality_Medium'] = [0]


    damage_types = [results[0].names[int(cls)] for cls in results[0].boxes.cpu().numpy().cls]

    # Function to generate the complete_case DataFrame
    def generate_complete_case(case, damage_types):
        complete_case = pd.DataFrame()

        for damage_type in damage_types:
            temp_case = case.copy()

            # Set the corresponding damage type to 1
            damage_column = f'damage_type_{damage_type}'
            if damage_column in temp_case.columns:
                temp_case[damage_column] = [1]

            # Append this row to the complete_case DataFrame
            complete_case = pd.concat([complete_case, temp_case], ignore_index=True)

        return complete_case


    # Generate the complete_case DataFrame
    complete_case = generate_complete_case(case, damage_types)

    # dataframe with the predictions
    cost_predictions= pd.DataFrame({'Damages': damage_types,
    'Cost Estimates': cost_model.predict(complete_case)
    })
    cost_predictions.to_csv("prueba.csv")




    # st.write(f'You selected: {selected_workshop}')
    
####################################################
###       Model Deployment                       ###
####################################################

# workshop selection
# workshops={
# 'Smithfield Autotech':'Low',
# 'Sandyford McCann Motors':'Medium',
# 'Mobile Mechanic':'High'
# }

# selected_workshop='Sandyford McCann Motors'

# selected_workshop_qual= workshops[selected_workshop]

# path upload
# path_upload = 'API/uploads'

# picture_name=os.listdir(path_upload)[0]

# full_path= os.path.join(path_upload,picture_name)

# we import both models
# yolo_model = YOLO("Models/best_modeltuning_YOLO-tuning8/train/weights/best.pt")
# cost_model=joblib.load("Models/cost_model.pkl")

# # we predict the image
# results = yolo_model.predict(full_path,conf=0.15)

# # we save the image
# path_output = os.path.join(path_upload, 'pred_'+picture_name)
# results[0].save(filename=path_output)

# # we predict the cost of the claims
# case = pd.DataFrame({
# 'const': [1],
# 'brand_Volkswagen': [0],
# 'model_Corolla': [0],
# 'model_Golf': [0],
# 'model_Polo': [0],
# 'model_Tiguan': [0],
# 'model_Yaris': [1],
# 'veh_age_range_Newer': [1],
# 'veh_age_range_Old': [0],
# 'workshop_quality_Low': [0],
# 'workshop_quality_Medium': [0],
# 'counties_group2': [0],
# 'counties_group3': [0],
# 'damage_type_met_dent_medium': [0],
# 'damage_type_met_dent_minor': [0],
# 'damage_type_met_dent_severe': [0],
# 'damage_type_met_tear': [0],
# 'damage_type_mis_lamp': [0],
# 'damage_type_mis_lost': [0],
# 'damage_type_mis_punct': [0]
# })


# # Adjust case DataFrame based on selected_workshop_qual
# if selected_workshop_qual == 'High':
#     case['workshop_quality_Low'] = [0]
#     case['workshop_quality_Medium'] = [0]
# elif selected_workshop_qual == 'Medium':
#     case['workshop_quality_Low'] = [0]
#     case['workshop_quality_Medium'] = [1]
# elif selected_workshop_qual == 'Low':
#     case['workshop_quality_Low'] = [1]
#     case['workshop_quality_Medium'] = [0]


# damage_types = [results[0].names[int(cls)] for cls in results[0].boxes.cpu().numpy().cls]
# # Function to generate the complete_case DataFrame
# def generate_complete_case(case, damage_types):
#     complete_case = pd.DataFrame()

#     for damage_type in damage_types:
#         temp_case = case.copy()

#         # Set the corresponding damage type to 1
#         damage_column = f'damage_type_{damage_type}'
#         if damage_column in temp_case.columns:
#             temp_case[damage_column] = [1]

#         # Append this row to the complete_case DataFrame
#         complete_case = pd.concat([complete_case, temp_case], ignore_index=True)

#     return complete_case


# # Generate the complete_case DataFrame
# complete_case = generate_complete_case(case, damage_types)

# print("Holaaaaaaaaaaaaaaaaaaaaaaaaaaa")
# # dataframe with the predictions
# cost_predictions= pd.DataFrame({'Damages': damage_types,
# 'Cost Estimates': cost_model.predict(complete_case)
# })
# cost_predictions.to_csv("prueba.csv")