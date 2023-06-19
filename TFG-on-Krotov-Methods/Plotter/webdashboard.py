import os
import glob
import re
from PIL import Image
import base64
import io

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Function to get starting digits in a string
def get_starting_digits(string):
    # This pattern will match any sequence of digits at the start of the string
    pattern = r'^\d+'
    # re.search returns only the first match of the pattern in the string
    match = re.search(pattern, string)
    # If a match was found, return it. Otherwise, return None
    return match.group(0) if match else None

# directory of your images
img_dir = "C:\\Users\\eloyfernandez\\Documents\\Eloy\\Uni\\TFG\\TFG\\TFG-on-Krotov-Methods\\Analisis\\Graphs\\"

# create a dictionary to hold our images, grouped by the first number
image_dict = {}

# get list of images
images = glob.glob(os.path.join(img_dir, '*.png'))

# group the images by the first number
for image in images:
    # extract the leading number (or numbers separated by 'from')
    leading_number = get_starting_digits(os.path.basename(image).split('.')[0])

    # add image to the corresponding group
    if leading_number not in image_dict:
        image_dict[leading_number] = []
    image_dict[leading_number].append(image)

# create dash app with bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# layout of the app
app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H2("Image Dashboard"),
            html.P("Use the dropdown to select a group of images"),
            dcc.Dropdown(
                id='image-dropdown',
                options=[{'label': i, 'value': i} for i in image_dict.keys()],
                value=list(image_dict.keys())[0]
            ),
            html.Br(),
            html.Div(id='image-container')
        ])
    ])
])

@app.callback(
    Output('image-container', 'children'),
    [Input('image-dropdown', 'value')]
)
def update_image(value):
    # get the selected images
    selected_images = image_dict[value]

    num_images = len(selected_images)

    if num_images <= 6:
        col_width = 2  # two rows maximum, so up to 6 images per row
    else:
        col_width = 1  # if more than 6 images, use a smaller column width

    # organize the images into a list of rows, with each row containing up to 6 images
    rows = []
    for i in range(0, num_images, 6):  # start a new row every 6 images
        row = dbc.Row([
            dbc.Col([
                html.H6(os.path.basename(image).split('.')[0], style={'text-align': 'center', 'font-size': '10px'}),  # the header
                html.Img(src='data:image/png;base64,'+base64.b64encode(open(image, 'rb').read()).decode(), style={'width': '100%'})
            ], width=col_width) 
            for image in selected_images[i:i+6]  # up to 6 images in each row
        ])
        rows.append(row)

    return rows




if __name__ == '__main__':
    app.run_server(debug=True)
