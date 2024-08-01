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
icon_path = os.path.join(os.path.dirname(__file__), 'Icons')

# Function to load an image and convert it to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Ensure correct file path

# Image person
image_base64 = get_image_base64(os.path.join(icon_path, 'person.png'))
image_person = (f'<img src="data:image/png;base64,'
              f'{image_base64}" style="position: absolute; '
              f'top: 40px; '
              f'left: 50%; '
              f'transform: translateX(-50%); '
              f'width: 150px; '
              f'height: 150px;">')

# Image magnifying_glass
image_base64 = get_image_base64(os.path.join(icon_path, 'magnifying_glass.png'))
image_glass = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'top: 20px; '
               f'right: 20px; '
               f'width: 30px; '
               f'height: 30px;">')

# Image more
image_base64 = get_image_base64(os.path.join(icon_path, 'more.png'))
image_more = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'top: 20px; '
               f'left: 20px; '
               f'width: 30px; '
               f'height: 30px;">')

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

# policyholder name
text_name = (f'<p style="position: absolute; '
             f'top: 180px; '  # Adjusted to be just below the image
             f'left: 50%; '
             f'transform: translateX(-50%); '
             f'font-family: Arial, sans-serif; '
             f'font-size: 18px; '
             f'font-weight: bold; '  # Make text bold
             f'color: white; '
             f'margin: 0;">Liam Murphy</p>')

# policy id
text_pol_id = (f'<p style="position: absolute; '
             f'top: 210px; '  # Adjusted to be just below the image
             f'left: 50%; '
             f'transform: translateX(-50%); '
             f'font-family: Arial, sans-serif; '
             f'font-size: 11px; '
             f'color: white; '
             f'margin: 0;">Policy ID 4532609</p>')

# Image policy_details with JavaScript onclick event
image_base64 = get_image_base64(os.path.join(icon_path, 'policy_details.png'))
image_policy_details = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'top: 55%; '
               f'left: 15%; '  # Position to the left
               f'transform: translate(-50%, -50%); '
               f'width: 100px; '
               f'height: 90px; cursor: pointer;" onclick="window.location.href=\'http://localhost:8501/Policy_Details\'">')

# Image file_a_claim
image_base64 = get_image_base64(os.path.join(icon_path, 'file_a_claim.png'))
image_file_a_claim = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'top: 55%; '
               f'left: 50%; '
               f'transform: translate(-50%, -50%); '
               f'width: 100px; '
               f'height: 90px;">')

# Image policy_details
image_base64 = get_image_base64(os.path.join(icon_path, 'manage_claims.png'))
image_manage_claims = (f'<img src="data:image/png;base64,'
               f'{image_base64}" style="position: absolute; '
               f'top: 55%; '
               f'left: 85%; '  # Position to the left
               f'transform: translate(-50%, -50%); '
               f'width: 120px; '
               f'height: 90px;">')


# Rounded rectangle with centered image
rounded_rectangle = f"""
<div style="
    width: 400px;
    height: 90px;
    background-color: #FFFFFF;
    border-radius: 10px;
    box-shadow: 0px 0px 30px 5px #006FAB;
    position: absolute;
    bottom: 60px;
    left: 50%;
    transform: translateX(-50%);
">
    {image_policy_details}
    {image_file_a_claim}
    {image_manage_claims}
</div>
"""

# HTML and CSS to create the main rectangle
rectangle_html = f"""
<div style="
    width: 400px;
    height: 500px;
    background-color: #62B6CB;
    margin: 30px -50px;
    box-shadow: 0px 0px 30px 5px #006FAB;
    position: relative;
">
    {image_person}
    {text_name}
    {text_pol_id}
    {image_glass}
    {image_more}
    {image_home}
    {image_policyholder}
    {image_settings}
    {rounded_rectangle}
</div>
"""

# Title of the Streamlit app
st.title("Cellphone App")

# Display the rectangle immediately after the title
st.markdown(rectangle_html, unsafe_allow_html=True)