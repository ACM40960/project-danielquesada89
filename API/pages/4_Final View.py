import streamlit as st
import base64
import os
from ultralytics import YOLO
import pandas as pd
import pickle

################################################
###            Detection Model Cal           ###
################################################

#  charge models and obtain the prediciton of the photo
path_upload = 'API/uploads'
picture_name=os.listdir(path_upload)[0]
full_path= os.path.join(path_upload,picture_name)
# we import model
yolo_model = YOLO("Models/yolo_model.pt")


# we predict the image
results = yolo_model.predict(full_path,conf=0.25)

# we save the image
path_output = os.path.join(path_upload, 'pred_'+picture_name)
results[0].save(filename=path_output)



################################################
###              General Set up              ###
################################################

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

################################################
###                   Objects                ###
################################################

############## Models Results ##############

# Image prediction
image_base64 = get_image_base64(path_output)

width_pred=300*1.1
height_pred=220*1.1

image_prediction = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'top: 25%; '
               f'left: 4%; '
               f'width: {width_pred}px; '
               f'height: {height_pred}px;">')


# Now create the HTML/CSS container
text_container_cost_results = (f'<div style="position: relative;  top: 32px; left: 360px;">'  # Adjusted positioning
                      f'<div style="font-family: Arial, sans-serif; '
                      f'font-size: 14px; '
                      f'color: #023047; '
                      f'display: flex; align-items: center;">'  # Use flexbox for alignment
                      f'<span style="font-weight: bold; margin-right: 10px;">Opening Reserve:</span> '
                      f'<span style="font-weight: normal;"> </span>'
                    f'</div>'
                      f'</div>')


############## Rest ##############

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
image_base64 = get_image_base64(os.path.join(icon_path, 'policy_details4.png'))
image_policy_details4 = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'top: 20px; '
               f'left: 20px; '
               f'width: 35px; '
               f'height: 35px;">')

# Image claimhandler_details
image_base64 = get_image_base64(os.path.join(icon_path, 'claimhandler_details.png'))
image_claimhandler_details = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'top: 20px; '
               f'left: 20px; '
               f'width: 35px; '
               f'height: 35px;">')

# Container of Policy details and Policy id
text_container_policyholder = (f'<div style="position: absolute; '
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

# Container of Claim handler open claim
text_container_claim_handler = (f'<div style="position: absolute; '
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
                  f'margin: 0;">Open Claims</p>'
                  f'<p style="font-family: Arial, sans-serif; '
                  f'font-size: 16px; '
                  f'color: white; '
                  f'margin: 0;"> </p>'
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

width_policyholder=540


### objects

approved_container = (f'<div style="position: relative; top: 20px; left: 50px;">'  # Position title within the rectangle
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

thankyou_container = (f'<div style="position: relative; top: 23px; left: 50px;">'  # Increased top value for more separation
                  f'<p style="font-family: Arial, sans-serif; '
                  f'font-size: 14px; '
                  f'font-weight: bold; '
                  f'color: #FB8500; '
                  f'display: inline-block; margin-right: 70px;">'  # Spacing to the right of the text
                  f'Thank you for your patience'
                  f'</p>'  # Properly close the paragraph tag
                  f'</div>')

approved_container2 = (f'<div style="position: relative; top: 10px; left: 20px;">'  # Adjusted positioning
                    f'<div style="font-family: Arial, sans-serif; '
                    f'font-size: 18px; '
                    f'font-weight: bold; '
                    f'color: #023047; '
                    f'display: flex; align-items: center;">'  # Use flexbox for alignment
                    f'Claim ID 29382921 '
                    f'<span style="margin-left: 20px; color: #8ED973;"> </span>'  # Green color for Approved with spacing
                    f'</div>'
                    f'</div>')

policyid_container = (f'<div style="position: relative; top: 10px; left: 20px;">'  # Adjusted positioning
                      f'<div style="font-family: Arial, sans-serif; '
                      f'font-size: 14px; '
                      f'color: #023047; '
                      f'display: flex; align-items: center;">'  # Use flexbox for alignment
                      f'<span style="font-weight: bold; margin-right: 10px;">Policy ID:</span> '
                      f'<span style="font-weight: normal;">4532609</span>'
                      f'<span style="margin-left: 206px; font-weight: bold;">Date:&nbsp;</span> '
                      f'<span style="font-weight: normal;">29/07/2024</span>'
                      f'</div>'
                      f'</div>')

workshop_approved_container = (f'<div style="position: relative; top: 10px; left: 20px;">'  # Adjusted positioning
                      f'<div style="font-family: Arial, sans-serif; '
                      f'font-size: 14px; '
                      f'color: #023047; '
                      f'display: flex; align-items: center;">'  # Use flexbox for alignment
                      f'<span style="font-weight: bold; margin-right: 10px;">Workshop:</span> '
                      f'<span style="font-weight: normal;"></span></span></span></span></span></span>'
                      f'<span style="margin-left: 250px; font-weight: bold;">Status:&nbsp;</span> '
                      f'<span style="font-weight: bold; color: #8ED973;">Approved</span>'  # Apply the green color to "Approved"
                      f'</div>'
                      f'</div>')


################################################
###         Policyholder View Display        ###
################################################

###############    policyholder buttons     ################

workshop_container = (f'<div style="position: relative; top: 200px; left: 58px;">'
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

date_picker_container = (f'<div style="position: relative; top: 240px; left: 58px;">'
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

###############    Rectangles     ################


# white rectangle with centered image
white_rect_policyholder_view = f"""
<div style="
    width: {width_policyholder-2}px;
    height: 375px;
    background-color: #FFFFFF;
    position: absolute;
    bottom: 55px;
    left: 50%;
    transform: translateX(-50%);
">
    {approved_container}
    {thankyou_container}
</div>
"""

# HTML and CSS to create the main rectangle with button directly in the HTML
policyholder_view = f"""
<div style="
    width: {width_policyholder}px;
    height: 500px;
    background-color: #62B6CB;
    position: absolute;
    top: 60px;
    left: 30px;  /* Position policyholder_view to the left */
    box-shadow: 0px 0px 30px 5px #006FAB;
">
    {image_policy_details4}
    {text_container_policyholder}
    {image_home}
    {image_policyholder}
    {image_settings}
    {white_rect_policyholder_view}


</div>
"""
# outline of over the policyholder_view rectangle
policyholder_view_outline = f"""
<div style="
    width: {width_policyholder * 1.02}px;
    height: 510px;
    background-color: #white;
    position: absolute;
    top: 9%;
    left: 25.8%;
    transform: translateX(-50%);
    box-shadow:
        inset -1px -1px 0px 0px #BFBFBF,   /* Top-left shadow for bevel */
        inset 1px 1px 0px 0px #BFBFBF,    /* Bottom-right shadow for bevel */
        inset 0px 0px 5px 1.5px #BFBFBF;    /* Inner shadow for additional depth */
">
    {workshop_container}
    {date_picker_container}

</div>
"""

################################################
###        Claim Handler View Display        ###
################################################

# white rectangle with centered image
white_rect_claim_handler_view = f"""
<div style="
    width: {width_policyholder-2}px;
    height: 429px;
    background-color: #FFFFFF;
    position: absolute;
    bottom: 1px;
    left: 50%;
    transform: translateX(-50%);
">
    {approved_container2}
    {policyid_container}
    {workshop_approved_container}
    {image_prediction}
    {text_container_cost_results}
</div>
"""

# HTML and CSS to create the main rectangle with button directly in the HTML
claim_handler_view = f"""
<div style="
    width: {width_policyholder}px;
    height: 500px;
    background-color: #4E95D9;
    position: absolute;
    top: 60px;
    right: 30px;  /* Position claim_handler_view to the right */
    box-shadow: 0px 0px 30px 5px #006FAB;
">
    {image_claimhandler_details}
    {text_container_claim_handler}
    {white_rect_claim_handler_view}


</div>
"""

# outline of over the policyholder_view rectangle
claim_handler_view_outline = f"""
<div style="
    width: {width_policyholder * 1.02}px;
    height: 510px;
    background-color: #white;
    position: absolute;
    top: 9%;
    left: 74%;
    transform: translateX(-50%);
    box-shadow:
        inset -1px -1px 0px 0px #BFBFBF,   /* Top-left shadow for bevel */
        inset 1px 1px 0px 0px #BFBFBF,    /* Bottom-right shadow for bevel */
        inset 0px 0px 5px 1.5px #BFBFBF;    /* Inner shadow for additional depth */
">

</div>
"""

################################################
###              Final View Display          ###
################################################

# rectagule with the details
rectangle_to_paint = f"""
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
    {policyholder_view}
    {policyholder_view_outline}
    {claim_handler_view}
    {claim_handler_view_outline}
</div>
"""

# HTML and CSS to create the main rectangle
final_rectangle = f"""
<div style="
    width: {width}px;
    height: {height2}px;
    background-color: #62B6CB;
    margin: -70px -50px -30px -200px;
    box-shadow: 0px 0px 30px 5px #006FAB;
    position: relative;
">


     {rectangle_to_paint}

</div>
"""

# Display the rectangle immediately after the title
st.markdown(final_rectangle, unsafe_allow_html=True)

################################################
###                 Button end               ###
################################################

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
        margin-top: 25px;  /* Align the button with the selectbox */
    }
    div.stSelectbox > div > div {
        width: 70%; /* Adjust the width percentage as needed */
    }
    div.stSelectbox > div > div > select {
        height: 35px;  /* Shorten the height of the selectbox */
        padding-left: 10px;
        padding-right: 30px;
        border-radius: 10px;
        font-size: 12px;  /* Adjust the font size to match the height */
        width: 100%;  /* Ensure the selectbox fills its container */
    }
    </style>
    """

# Apply the CSS
st.markdown(button_style, unsafe_allow_html=True)

# Use Streamlit's selectbox widget to capture the selection
col1, col2 = st.columns([3, 1])  # Adjust the column widths

with col1:
    buttom_workshop = st.selectbox(
        '',
        ["Select among available workshops",
         "Smithfield Autotech",
         "Sandyford McCann Motor",
         "Mobile Mechanic"],
        index=0
    )

with col2:
    if st.button("Home", key="home"):
        st.switch_page(os.path.join(os.getcwd(), "API/Homepage.py"))


################################################
###                Cost Model Cal            ###
################################################

# we import model
with open('./Models/cost_model.pkl', 'rb') as file:
    cost_model = pickle.load(file)

# Display the selected workshop
if buttom_workshop != "Select among available workshops":
    # workshop selection
    workshops={
    'Smithfield Autotech':'Low',
    'Sandyford McCann Motor':'Medium',
    'Mobile Mechanic':'High'
    }

    selected_workshop_qual = workshops[buttom_workshop]

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
    'Cost Estimates': cost_model.predict(complete_case).round(0)
    })

    # Calculate the total cost as the sum of all 'Cost Estimates'
    total_cost = cost_predictions['Cost Estimates'].sum()

    # Format total_cost as a string with no decimals and a comma as a thousands separator
    formatted_total_cost = f"{total_cost:,.0f}"

    # CSS to center the text vertically and horizontally
    text_cost_results = (f'<div style="position: absolute; '
                         f'top: -463px; '  # Adjust this value to position vertically
                         f'left: 880px; '  
                         f'font-family: Arial, sans-serif; '
                         f'font-size: 14px; '  # Font size for visibility
                         f'font-weight: bold; '
                         f'color: #023047;">'
                         f'€{formatted_total_cost}'
                         f'</div>')

    st.markdown(text_cost_results, unsafe_allow_html=True)

    # print selected workshop
    text_workshop_results = (f'<div style="position: absolute; '
                         f'top: -523px; '  # Adjust this value to position vertically
                         f'left: 490px; '  
                         f'font-family: Arial, sans-serif; '
                         f'font-size: 14px; '  # Font size for visibility
                         f'color: #023047;">'
                         f'{buttom_workshop}'
                         f'</div>')

    st.markdown(text_workshop_results, unsafe_allow_html=True)

    ##### we print the disclosure of damages

    # dictionary of damage types
    rename_damage_type = {
        'mis_lost': 'Lost Parts',
        'met_tear': 'Torn',
        'met_scratch': 'Paint Scratches',
        'glass_crack': 'Broken Glass',
        'mis_punct': 'Puncture',
        'mis_lamp': 'Broken Lamp',
        'met_dent_minor': 'Minor Dent',
        'met_dent_medium': 'Medium Dent',
        'met_dent_severe': 'Severe Dent'
    }

    cost_predictions["Damages"] = cost_predictions["Damages"].map(rename_damage_type)

    # Sort the "Damages" column alphabetically
    cost_predictions = cost_predictions.sort_values("Damages").reset_index(drop=True)

    # Add a sequential number to each group of damage types
    cost_predictions["count"] = cost_predictions.groupby("Damages").cumcount() + 1

    # Combine the original damage type with the sequential number
    cost_predictions["Damages"] = cost_predictions["Damages"].astype(str) + ' ' + cost_predictions["count"].astype(str)

    # we define the length of the dataset
    n=len(cost_predictions["Damages"])

    if n>0: # damage 1
        damage = cost_predictions["Damages"][0]
        damage_cost = cost_predictions["Cost Estimates"][0]
        formatted_damage_cost = f"{damage_cost:,.0f}"

        damage_disclosure_1_container = (
            f'<div style="position: absolute; top: -465px; left: 755px; white-space: nowrap;">'
            f'<p style="font-family: Arial, sans-serif; font-size: 14px; color: #023047; display: inline;">'
            f'{damage}:'
            f'</p>'
            f'<span style="font-family: Arial, sans-serif; font-size: 14px; color: #023047; margin-left: 10px;">'
            f'€{formatted_damage_cost}'
            f'</span>'
            f'</div>'
        )

        st.markdown(damage_disclosure_1_container, unsafe_allow_html=True)

    if n>1: # damage 2
        damage = cost_predictions["Damages"][1]
        damage_cost = cost_predictions["Cost Estimates"][1]
        formatted_damage_cost = f"{damage_cost:,.0f}"

        damage_disclosure_2_container = (
            f'<div style="position: absolute; top: {-465+10*1}px; left: 755px; white-space: nowrap;">'
            f'<p style="font-family: Arial, sans-serif; font-size: 14px; color: #023047; display: inline;">'
            f'{damage}:'
            f'</p>'
            f'<span style="font-family: Arial, sans-serif; font-size: 14px; color: #023047; margin-left: 10px;">'
            f'€{formatted_damage_cost}'
            f'</span>'
            f'</div>'
        )

        st.markdown(damage_disclosure_2_container, unsafe_allow_html=True)

    if n>2: # damage 3
        damage = cost_predictions["Damages"][2]
        damage_cost = cost_predictions["Cost Estimates"][2]
        formatted_damage_cost = f"{damage_cost:,.0f}"

        damage_disclosure_3_container = (
            f'<div style="position: absolute; top: {-465+10*2}px; left: 755px; white-space: nowrap;">'
            f'<p style="font-family: Arial, sans-serif; font-size: 14px; color: #023047; display: inline;">'
            f'{damage}:'
            f'</p>'
            f'<span style="font-family: Arial, sans-serif; font-size: 14px; color: #023047; margin-left: 10px;">'
            f'€{formatted_damage_cost}'
            f'</span>'
            f'</div>'
        )

        st.markdown(damage_disclosure_3_container, unsafe_allow_html=True)

    if n>3: # damage 4
        damage = cost_predictions["Damages"][3]
        damage_cost = cost_predictions["Cost Estimates"][3]
        formatted_damage_cost = f"{damage_cost:,.0f}"

        damage_disclosure_4_container = (
            f'<div style="position: absolute; top: {-465+10*3}px; left: 755px; white-space: nowrap;">'
            f'<p style="font-family: Arial, sans-serif; font-size: 14px; color: #023047; display: inline;">'
            f'{damage}:'
            f'</p>'
            f'<span style="font-family: Arial, sans-serif; font-size: 14px; color: #023047; margin-left: 10px;">'
            f'€{formatted_damage_cost}'
            f'</span>'
            f'</div>'
        )

        st.markdown(damage_disclosure_4_container, unsafe_allow_html=True)

    if n>4: # damage 5
        damage = cost_predictions["Damages"][4]
        damage_cost = cost_predictions["Cost Estimates"][4]
        formatted_damage_cost = f"{damage_cost:,.0f}"

        damage_disclosure_5_container = (
            f'<div style="position: absolute; top: {-465+10*4}px; left: 755px; white-space: nowrap;">'
            f'<p style="font-family: Arial, sans-serif; font-size: 14px; color: #023047; display: inline;">'
            f'{damage}:'
            f'</p>'
            f'<span style="font-family: Arial, sans-serif; font-size: 14px; color: #023047; margin-left: 10px;">'
            f'€{formatted_damage_cost}'
            f'</span>'
            f'</div>'
        )

        st.markdown(damage_disclosure_5_container, unsafe_allow_html=True)