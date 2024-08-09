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

# White rectangle with centered image
white_rectangle = f"""
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
</div>
"""

# rectagule with the details
rectangle2 = f"""
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


# HTML and CSS to create the main rectangle
rectangle_html2 = f"""
<div style="
    width: {width}px;
    height: {height2}px;
    background-color: #62B6CB;
    margin: 30px -50px 30px -200px;
    box-shadow: 0px 0px 30px 5px #006FAB;
    position: relative;
">
     {rectangle2}
     {white_rectangle}
</div>
"""


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
# Inject custom CSS
st.markdown(button_style, unsafe_allow_html=True)

# Display the rectangle immediately after the title
st.markdown(rectangle_html2, unsafe_allow_html=True)

# Add buttons below the rectangle
col1, col2, col3 = st.columns([0.5, 1,2])


with col2:
    if st.button("Home", key="home"):
        st.switch_page(os.path.join(os.getcwd(), "API/Homepage.py"))