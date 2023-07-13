import os
import glob
import re
import base64

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from PIL import Image
import io

# Function to get starting digits in a string
def get_starting_digits(string):
    pattern = r'^\d+'
    match = re.search(pattern, string)
    return match.group(0) if match else None

base_dir = "C:\\Users\\eloyfernandez\\Documents\\Eloy\\Uni\\TFG\\TFG\\TFG-on-Krotov-Methods\\Plotter\\Graphs\\"
tolerance_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

navbar = dbc.NavbarSimple(
    brand="Image Dashboard",
    brand_href="#",
    sticky="top",
    color="dark",
    dark=True,
)

dropdown_group = dbc.CardGroup([
    dbc.Label("Select Tolerance", html_for="tolerance-dropdown", width=4),
    dbc.Col(
        dcc.Dropdown(
            id='tolerance-dropdown',
            options=[{'label': i, 'value': i} for i in tolerance_dirs],
            value=tolerance_dirs[0]
        ),
        width=8,
    ),
    html.Br(),
    dbc.Label("Select Image Group", html_for="image-dropdown", width=4),
    dbc.Col(
        dcc.Dropdown(
            id='image-dropdown',
            value=None
        ),
        width=8,
    ),
    html.Br(),
])

image_div = html.Div(id='image-container')

app.layout = html.Div([
    navbar,
    dbc.Container([
        dbc.Row(dbc.Col(dropdown_group, md=6)),
        dbc.Row(dbc.Col(image_div))
    ])
])

# define graph style
graph_style = {
    'height': 'auto',  # adjust the height of the image
    'width': '100%',  # adjust the width of the image
    'object-fit': 'contain',  # keep aspect ratio and fit within dimension constraints
}

@app.callback(
    Output('image-dropdown', 'options'),
    [Input('tolerance-dropdown', 'value')]
)
def set_image_dropdown(tolerance):
    img_dir = os.path.join(base_dir, tolerance)
    image_dict = {}
    images = glob.glob(os.path.join(img_dir, '*.png'))
    for image in images:
        leading_number = get_starting_digits(os.path.basename(image).split('.')[0])
        if leading_number not in image_dict:
            image_dict[leading_number] = []
        image_dict[leading_number].append(image)

    return [{'label': i, 'value': i} for i in image_dict.keys()]

@app.callback(
    Output('image-container', 'children'),
    [Input('image-dropdown', 'value'), Input('tolerance-dropdown', 'value')]
)
def update_image(value, tolerance):
    img_dir = os.path.join(base_dir, tolerance)
    image_dict = {}
    images = glob.glob(os.path.join(img_dir, '*.png'))
    for image in images:
        leading_number = get_starting_digits(os.path.basename(image).split('.')[0])
        if leading_number not in image_dict:
            image_dict[leading_number] = []
        image_dict[leading_number].append(image)
    selected_images = image_dict[value]

    num_images = len(selected_images)
    if num_images <= 6:
        col_width = 4
    else:
        col_width = 2

    rows = []
    for i in range(0, num_images, 6):
        row = dbc.Row([
            dbc.Col([
                html.H4(os.path.basename(image).split('.')[0], style={'text-align': 'center', 'font-size': '8px'}),
                html.Img(src='data:image/png;base64,'+base64.b64encode(open(image, 'rb').read()).decode(), style=graph_style)
            ], width=col_width)
            for image in selected_images[i:i+6]
        ])
        rows.append(row)

    return rows

if __name__ == '__main__':
    app.run_server(debug=True)
