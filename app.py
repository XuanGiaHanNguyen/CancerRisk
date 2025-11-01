import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import io

# ---------------------- INITIALIZE APP ----------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Lung Cancer Risk Dashboard"

# ---------------------- LAYOUT ----------------------
app.layout = dbc.Container([
    html.H3("Lung Cancer Risk Dashboard",
            style={ "marginTop": "1rem", "marginBottom": "1rem"}),


    dbc.Row([
        # ---------- LEFT PANEL ----------
        dbc.Col([
            dbc.Card([

                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            "Drag and Drop or ",
                            html.A("Select a CSV File", style={"fontWeight": "500"})
                        ]),
                        multiple=False,
                        style={
                            'border': '1px dashed #ccc',
                            'borderRadius': '3px',
                            'textAlign': 'center',
                            'padding': '1.5rem',
                            'cursor': 'pointer'
                        }
                    ),
                    html.Div(id='upload-status', className="mt-2", style={"fontSize": "0.9rem"}),

                    html.Hr(),

                    html.H6("Data Cleaning", className="mt-3"),
                    dbc.Button("Fill Missing Values", id='fill-missing', color='secondary',
                               className='w-100 mb-2', style={'borderRadius': '3px'}),
                    dbc.Button("Encode Categoricals", id='encode-cat', color='secondary',
                               className='w-100 mb-2', style={'borderRadius': '3px'}),
                    dbc.Button("Normalize Numerics", id='normalize-num', color='secondary',
                               className='w-100 mb-2', style={'borderRadius': '3px'}),
                ])
            ], style={'borderRadius': '3px', 'boxShadow': 'none', 'border': '1px solid #ddd'})
        ], md=3),

        # ---------- RIGHT PANEL ----------
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Table", tab_id="preview"),
                dbc.Tab(label="Visualization", tab_id="viz"),
                dbc.Tab(label="Model", tab_id="model")
            ], id='tabs', active_tab='preview',
                style={'borderRadius': '3px'}),

            html.Div(id='tab-content', className="mt-3")
        ], md=9)
    ]),

    dcc.Store(id='stored-data')
], fluid=True)

# ---------------------- CALLBACKS ----------------------

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return df


@app.callback(
    Output('stored-data', 'data'),
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def load_data(contents, filename):
    if contents is None:
        return None, ""
    try:
        df = parse_contents(contents, filename)
        msg = f"Loaded {filename} ({df.shape[0]} rows, {df.shape[1]} columns)"
        return df.to_json(date_format='iso', orient='split'), msg
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


@app.callback(
    Output('stored-data', 'data', allow_duplicate=True),
    Input('fill-missing', 'n_clicks'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def fill_missing(n, json_data):
    df = pd.read_json(json_data, orient='split')
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna("Unknown", inplace=True)
    return df.to_json(date_format='iso', orient='split')


@app.callback(
    Output('stored-data', 'data', allow_duplicate=True),
    Input('encode-cat', 'n_clicks'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def encode_categoricals(n, json_data):
    df = pd.read_json(json_data, orient='split')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.Categorical(df[col]).codes
    return df.to_json(date_format='iso', orient='split')


@app.callback(
    Output('stored-data', 'data', allow_duplicate=True),
    Input('normalize-num', 'n_clicks'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def normalize_numerics(n, json_data):
    df = pd.read_json(json_data, orient='split')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    return df.to_json(date_format='iso', orient='split')


@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'active_tab'),
    State('stored-data', 'data')
)
def render_tabs(tab, json_data):
    if json_data is None:
        return html.Div("Please upload a dataset first.", style={"color": "#666"})

    df = pd.read_json(json_data, orient='split')

    if tab == "preview":
        return dbc.Card([
            dbc.CardHeader("Data Preview", style={"fontWeight": "600"}),
            dbc.CardBody([
                dash_table.DataTable(
                    data=df.head(50).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'fontSize': '0.85rem', 'padding': '6px',
                        'border': '1px solid #eee', 'textAlign': 'left'
                    },
                    style_header={
                        'fontWeight': 'bold', 'backgroundColor': '#f9f9f9'
                    }
                ),
                html.Br(),
                html.Div(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
                         style={"fontSize": "0.9rem", "color": "#777"})
            ])
        ], style={'borderRadius': '3px', 'border': '1px solid #ddd'})

    elif tab == "viz":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return html.Div("Not enough numeric columns for visualization.", style={"color": "#666"})

        return dbc.Card([
            dbc.CardHeader("Data Visualization", style={"fontWeight": "600"}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("X-Axis"),
                        dcc.Dropdown(
                            id='x-axis', 
                            options=[{'label': c, 'value': c} for c in df.columns],
                            value=numeric_cols[0],
                            clearable=False
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Y-Axis"),
                        dcc.Dropdown(
                            id='y-axis', 
                            options=[{'label': c, 'value': c} for c in numeric_cols],
                            value=numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0],
                            clearable=False
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Chart Type"),
                        dcc.Dropdown(
                            id='chart-type',
                            options=[
                                {'label': 'Scatter', 'value': 'scatter'},
                                {'label': 'Bar', 'value': 'bar'},
                                {'label': 'Line', 'value': 'line'},
                                {'label': 'Histogram', 'value': 'hist'}
                            ],
                            value='scatter', clearable=False
                        )
                    ], md=4)
                ], className="mb-3"),

                dcc.Loading(dcc.Graph(id='main-chart', config={'displayModeBar': True}))
            ])
        ], style={'borderRadius': '3px', 'border': '1px solid #ddd'})

    elif tab == "model":
        return dbc.Card([
            dbc.CardHeader("Model Setup (Frontend Only)", style={"fontWeight": "600"}),
            dbc.CardBody([
                html.P("This section is reserved for model training and prediction integration.",
                       style={"color": "#666"}),
                html.P("You can later connect this tab to your backend ML service or trained model.",
                       style={"color": "#666"})
            ])
        ], style={'borderRadius': '3px', 'border': '1px solid #ddd'})

    return html.Div()


@app.callback(
    Output('main-chart', 'figure'),
    Input('chart-type', 'value'),
    Input('x-axis', 'value'),
    Input('y-axis', 'value'),
    State('stored-data', 'data')
)
def update_chart(chart_type, x_col, y_col, json_data):
    df = pd.read_json(json_data, orient='split')
    if x_col not in df.columns or y_col not in df.columns:
        return px.scatter()

    if chart_type == 'scatter':
        fig = px.scatter(df, x=x_col, y=y_col)
    elif chart_type == 'bar':
        fig = px.bar(df, x=x_col, y=y_col)
    elif chart_type == 'line':
        fig = px.line(df, x=x_col, y=y_col)
    elif chart_type == 'hist':
        fig = px.histogram(df, x=x_col)
    else:
        fig = px.scatter(df, x=x_col, y=y_col)

    fig.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=40, b=20), height=500)
    return fig


# ---------------------- RUN APP ----------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
