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

# Custom CSS to position the button on top of the rectangle
button_css = """
<style>
#button-container {
    position: absolute;
    top: 60%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 1;
}

.stButton>button {
    width: 400px;
    height: 80px;
    background-color: #023047;
    color: white;
    font-size: 20px;
    border: none;
    border-radius: 20px;
    box-shadow: 0px 0px 10px 2px rgba(0,0,0,0.3);
}
</style>
"""
st.markdown(button_css, unsafe_allow_html=True)

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

# Display the title and description using Streamlit layout
st.title("Cellphone App")

# Center the content using a container
with st.container():
    # Display the main content block
    st.markdown(f"""
        <div style="
            width: 800px;
            height: 500px;
            background-color: #62B6CB;
            margin: 30px -50px;
            box-shadow: 0px 0px 30px 5px #006FAB;
            position: relative;
            text-align: center;
        ">
            {image_policy_details2}
            {text_container}
            {image_home}
            {image_policyholder}
            {image_settings}
        </div>
        """, unsafe_allow_html=True)
    
    # Position the button using a placeholder and custom CSS
    button_placeholder = st.empty()
    with button_placeholder.container():
        st.markdown('<div id="button-container"></div>', unsafe_allow_html=True)
        if st.button("Open a New Claim", key="open_claim"):
            st.switch_page(os.path.join(os.getcwd(),"API/pages/1_Policy_Details.py"))

