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

# Image policyholder_details
image_base64 = get_image_base64(os.path.join(icon_path, 'policyholder_details.png'))
image_policyholder_details = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'left: 50%; '
               f'top: 5%; '  # Position to the left
               f'transform: translateX(-50%); '
               f'width: 70px; '
               f'height: 70px;">')

# Image veh_details
image_base64 = get_image_base64(os.path.join(icon_path, 'veh_details.png'))
image_veh_details = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'left: 50%; '
               f'top: 45%; '  # Centered horizontally
               f'transform: translateX(-50%); '
               f'width: 70px; '
               f'height: 70px;">')

# Image resident_details
image_base64 = get_image_base64(os.path.join(icon_path, 'resident_details.png'))
image_resident_details = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'left: 50%; '
               f'top: 72%; '  # Position to the right
               f'transform: translateX(-50%); '
               f'width: 70px; '
               f'height: 70px;">')

# Rounded rectangle with centered image #F2F2F2 #E0E0E0
vertical_rectangle = f"""
<div style="
    width: 115px;
    height: 370px;
    background-color: #F2F2F2;
    position: absolute;
    left: 0px;
">
    {image_policyholder_details}
    {image_veh_details}
    {image_resident_details}
</div>
"""

image_base64 = get_image_base64(os.path.join(icon_path, 'pol_det.png'))
image_pol_det = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'top: 10px; '
               f'width: 285px; '
               f'height: 330px;">')

# Rounded rectangle with centered image #F2F2F2 #E0E0E0
vertical_rectangle2 = f"""
<div style="
    width: 285px;
    height: 370px;
    background-color: #FFFFFF;
    position: absolute;
    left: 115px;
">
    {image_pol_det}

</div>
"""

# rectagule with the details
rectangle2 = f"""
<div style="
    width: 400px;
    height: 370px;
    background-color: #FFFFFF;
    position: absolute;
    bottom: 60px;
    left: 50%;
    transform: translateX(-50%);
">
    {vertical_rectangle}
    {vertical_rectangle2}

</div>
"""

# HTML and CSS to create the main rectangle
rectangle_html2 = f"""
<div style="
    width: 400px;
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
    {rectangle2}
</div>
"""


st.title("Cellphone App")

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
