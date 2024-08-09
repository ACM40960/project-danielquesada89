import streamlit as st
import base64
import os

st.set_page_config(
    page_title="Cellphone App",
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
                  f'margin: 0;">Policy Details</p>'
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
               f'left: 51%; '  # Centered horizontally
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





image_base64 = get_image_base64(os.path.join(icon_path, 'pol_det.png'))
image_pol_det = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'top: 10px; '
               f'width: 285px; '
               f'height: 330px;">')


# Title Container for Claim ID
title_container2 = (f'<div style="margin-bottom: 20px;">'  # Add margin to create space below the title
                    f'<p style="font-family: Arial, sans-serif; '
                    f'font-size: 18px; '
                    f'font-weight: bold; '
                    f'color: #023047;">Policy ID 29382921</p>'
                    f'</div>')

# Date Picker Container
# Date Picker Container
date_picker_container = (f'<div style="margin-bottom: 15px;">'
                         f'<p style="font-family: Arial, sans-serif; '
                         f'font-size: 15px; '  # Match font size with other containers
                         f'color: #FF6200; margin-bottom: 8px;">Date</p>'
                         f'<input type="date" placeholder="Enter the Claim Date" '
                         f'style="width: 40%; padding: 7px; font-size: 10px; '  # Match width and padding with other containers
                         f'border-radius: 5px; border: 2px solid #023047;">'  # Match border style with other containers
                         f'</div>')

# Time Picker Container
time_picker_container = (f'<div>'  
                         f'<p style="font-family: Arial, sans-serif; '
                         f'font-size: 15px; '
                         f'color: #FF6200; margin-bottom: 8px;">Time</p>'
                         f'<input type="time" placeholder="Enter the Claim Time" '
                         f'style="width: 40%; padding: 7px; font-size: 10px; '
                         f'border-radius: 5px; border: 2px solid #023047;">'
                         f'</div>')

# Yes/No Selector Container
yes_no_container = (f'<div style="margin-top: 10px;">'
                    f'<p style="font-family: Arial, sans-serif; '
                    f'font-size: 15px; '
                    f'color: #FF6200; margin-bottom: 8px;">Do you need road assistance?</p>'
                    f'<select style="width: 40%; padding: 7px; font-size: 10px; '
                    f'border-radius: 5px; border: 2px solid #023047;">'
                    f'  <option value="">Select Yes / No</option>'
                    f'  <option value="Yes">Yes</option>'
                    f'  <option value="No">No</option>'
                    f'</select>'
                    f'</div>')



# Now include this title container and the date picker inside the `rectangle2`
rectangle2 = f"""
<div style="
    width: 800px;
    height: 370px;
    background-color: #FFFFFF;
    position: absolute;
    bottom: 60px;
    left: 50%;
    transform: translateX(-50%);
    padding: 20px;
">
    {title_container2}
    {date_picker_container}  <!-- Add the date picker container here -->
    {time_picker_container}
    {yes_no_container }
</div>
"""

rectangle_html2 = f"""
<div style="
    width: 800px;
    height: 500px;
    background-color: #62B6CB;
    margin: 30px -50px;
    box-shadow: 0px 0px 30px 5px #006FAB;
    position: relative;
">
    {image_policy_details2}
    {text_container}
    {image_home}
    {image_policyholder}
    {image_settings}
    {rectangle2}  <!-- Updated rectangle with the new title and date input -->
</div>
"""


# Claim Details Picture Upload Container
uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to a directory
    save_directory = "uploads"  # Specify the directory where you want to save the file
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_path = os.path.join(save_directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"Photo saved successfully: {file_path}")

    # Optionally display the uploaded image
    st.image(file_path, caption="Uploaded Photo", use_column_width=True)

# HTML Container for the file upload (simulating placement within HTML)
file_upload_container = (f'<div style="margin-top: 20px;">'
                         f'<p style="font-family: Arial, sans-serif; '
                         f'font-size: 15px; '
                         f'color: #FF6200; margin-bottom: 8px;">Claim Details Picture</p>'
                         f'<input type="text" placeholder="Upload Photo" '
                         f'style="width: 40%; padding: 7px; font-size: 10px; '
                         f'border-radius: 5px; border: 2px solid #023047;">'
                         f'</div>')


# Now include this title container and the date picker inside the `rectangle2`
rectangle2 = f"""
<div style="
    width: 800px;
    height: 370px;
    background-color: #FFFFFF;
    position: absolute;
    bottom: 60px;
    left: 50%;
    transform: translateX(-50%);
    padding: 20px;
">
    {title_container2}
    {date_picker_container}  <!-- Add the date picker container here -->
    {time_picker_container}
    {yes_no_container}
    {file_upload_container}  <!-- File Upload HTML Placeholder -->
</div>
"""

rectangle_html2 = f"""
<div style="
    width: 800px;
    height: 500px;
    background-color: #62B6CB;
    margin: 30px -50px;
    box-shadow: 0px 0px 30px 5px #006FAB;
    position: relative;
">
    {image_policy_details2}
    {text_container}
    {image_home}
    {image_policyholder}
    {image_settings}
    {rectangle2}  <!-- Updated rectangle with the new title and date input -->
</div>
"""
# Display the rectangle immediately after the title
st.markdown(rectangle_html2, unsafe_allow_html=True)



