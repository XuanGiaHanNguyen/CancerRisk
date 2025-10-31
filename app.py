import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import base64
import io

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "ML Explorer"

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 0;
                background: white;
            }
            .header {
                background: #000;
                color: white;
                padding: 1.5rem;
            }
            .card {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                margin-bottom: 1.5rem;
                background: white;
            }
            .upload-area {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 2rem;
                text-align: center;
                background-color: white;
                cursor: pointer;
            }
            .upload-area:hover {
                border-color: #666;
                background-color: #fafafa;
            }
            .stat-card {
                background: white;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 1rem;
            }
            .nav-tabs {
                border-bottom: 1px solid #e0e0e0;
            }
            .nav-tabs .nav-link {
                color: #666;
                border: none;
                border-bottom: 2px solid transparent;
                padding: 0.75rem 1.5rem;
                font-weight: 400;
            }
            .nav-tabs .nav-link:hover {
                color: #000;
            }
            .nav-tabs .nav-link.active {
                color: #000;
                border-bottom: 2px solid #000;
                background: transparent;
            }
            .feature-button {
                margin: 0.25rem;
                padding: 0.5rem 1rem;
                border-radius: 2px;
                border: 1px solid #ccc;
                background: white;
                cursor: pointer;
                transition: all 0.2s;
                font-size: 0.875rem;
            }
            .feature-button:hover {
                border-color: #666;
            }
            .feature-button.selected {
                background: #000;
                color: white;
                border-color: #000;
            }
            .btn-primary {
                background: #000 !important;
                border: 1px solid #000 !important;
                color: white !important;
            }
            .btn-primary:hover {
                background: #333 !important;
                border: 1px solid #333 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout
app.layout = dbc.Container([
    
    html.Div([
        # Upload Section
        dbc.Card([
            dbc.CardBody([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        html.Span("Drag and Drop or ", style={"fontSize": "1rem", "color": "#666"}),
                        html.A("Select a CSV File", style={"color": "#000", "fontWeight": "500", "cursor": "pointer"})
                    ]),
                    multiple=False
                ),
                html.Div(id='upload-status', className="mt-3")
            ])
        ]),
        
        # Tabs
        html.Div(id='tabs-container', style={"display": "none"}, children=[
            dbc.Tabs(
                id="tabs",
                active_tab="preview",
                children=[
                    dbc.Tab(label="Data Preview", tab_id="preview"),
                    dbc.Tab(label="Visualization", tab_id="viz"),
                    dbc.Tab(label="Machine Learning", tab_id="ml"),
                ],
                className="mb-4"
            ),
            html.Div(id='tab-content')
        ]),
        
        # Store components
        dcc.Store(id='stored-data'),
        dcc.Store(id='stored-columns'),
        dcc.Store(id='feature-columns', data=[]),
        
    ], style={"maxWidth": "1400px", "margin": "0 auto", "padding": "2rem"})
], fluid=True, style={"padding": "0"})


def parse_contents(contents, filename):
    """Parse uploaded CSV file"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return df
    except Exception as e:
        return None


@app.callback(
    [Output('stored-data', 'data'),
     Output('stored-columns', 'data'),
     Output('upload-status', 'children'),
     Output('tabs-container', 'style')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    """Handle file upload"""
    if contents is None:
        return None, None, "", {"display": "none"}
    
    df = parse_contents(contents, filename)
    
    if df is None:
        return None, None, dbc.Alert("Error parsing file. Please upload a valid CSV.", color="danger"), {"display": "none"}
    
    status = html.Div([
        f"Successfully loaded {filename} ({len(df)} rows, {len(df.columns)} columns)"
    ], style={"color": "#666", "fontSize": "0.875rem"})
    
    return df.to_json(date_format='iso', orient='split'), df.columns.tolist(), status, {"display": "block"}


@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'active_tab'),
     Input('stored-data', 'data'),
     Input('stored-columns', 'data')]
)
def render_tab_content(active_tab, json_data, columns):
    """Render content based on active tab"""
    if json_data is None:
        return html.Div()
    
    df = pd.read_json(json_data, orient='split')
    
    if active_tab == "preview":
        return create_preview_tab(df, columns)
    elif active_tab == "viz":
        return create_viz_tab(df, columns)
    elif active_tab == "ml":
        return create_ml_tab(df, columns)
    
    return html.Div()


def create_preview_tab(df, columns):
    """Create data preview tab"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Statistics cards
    stat_cards = []
    for col in columns[:4]:
        if col in numeric_cols:
            mean_val = df[col].mean()
            min_val = df[col].min()
            max_val = df[col].max()
            card = dbc.Col([
                html.Div([
                    html.Div(col, style={"fontSize": "0.75rem", "color": "#666", "marginBottom": "0.5rem"}),
                    html.Div(f"{mean_val:.2f}", style={"fontSize": "1.5rem", "fontWeight": "600", "color": "#000"}),
                    html.Div(f"Range: {min_val:.2f} - {max_val:.2f}", style={"fontSize": "0.7rem", "color": "#999"})
                ], className="stat-card")
            ], width=3)
        else:
            unique_count = df[col].nunique()
            card = dbc.Col([
                html.Div([
                    html.Div(col, style={"fontSize": "0.75rem", "color": "#666", "marginBottom": "0.5rem"}),
                    html.Div(str(unique_count), style={"fontSize": "1.5rem", "fontWeight": "600", "color": "#000"}),
                    html.Div("Unique values", style={"fontSize": "0.7rem", "color": "#999"})
                ], className="stat-card")
            ], width=3)
        stat_cards.append(card)
    
    return html.Div([
        
        dbc.Card([
            dbc.CardBody([
                html.H6("Data Preview", className="mb-2", style={"fontWeight": "500"}),
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    style_table={'overflowX': 'auto', 'maxHeight': '600px', 'overflowY': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',
                        'fontFamily': 'inherit',
                        'fontSize': '0.813rem',
                        'border': '1px solid #e0e0e0'
                    },
                    style_header={
                        'backgroundColor': '#fafafa',
                        'fontWeight': '500',
                        'color': '#000',
                        'border': '1px solid #e0e0e0'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#fafafa'
                        }
                    ],
                    page_size=50
                )
            ])
        ], className="card")
    ])


def create_viz_tab(df, columns):
    """Create visualization tab"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H6("Visualization Controls", className="mb-3", style={"fontWeight": "500"}),
                dbc.Row([
                    dbc.Col([
                        html.Label("Chart Type", style={"fontWeight": "400", "marginBottom": "0.5rem", "fontSize": "0.875rem"}),
                        dcc.Dropdown(
                            id='chart-type',
                            options=[
                                {'label': 'Scatter Plot', 'value': 'scatter'},
                                {'label': 'Line Chart', 'value': 'line'},
                                {'label': 'Bar Chart', 'value': 'bar'},
                            ],
                            value='scatter',
                            clearable=False
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("X-Axis", style={"fontWeight": "400", "marginBottom": "0.5rem", "fontSize": "0.875rem"}),
                        dcc.Dropdown(
                            id='x-axis',
                            options=[{'label': col, 'value': col} for col in columns],
                            value=columns[0]
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Y-Axis", style={"fontWeight": "400", "marginBottom": "0.5rem", "fontSize": "0.875rem"}),
                        dcc.Dropdown(
                            id='y-axis',
                            options=[{'label': col, 'value': col} for col in numeric_cols],
                            value=numeric_cols[0] if numeric_cols else None
                        )
                    ], md=4)
                ])
            ])
        ], className="card"),
        
        dbc.Card([
            dbc.CardBody([
                html.Div(id='chart-title'),
                dcc.Loading(
                    dcc.Graph(id='main-chart', config={'displayModeBar': True})
                )
            ])
        ], className="card")
    ])


def create_ml_tab(df, columns):
    """Create ML tab"""
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H6("Model Configuration", className="mb-3", style={"fontWeight": "500"}),
                dbc.Row([
                    dbc.Col([
                        html.Label("Model Type", style={"fontWeight": "400", "marginBottom": "0.5rem", "fontSize": "0.875rem"}),
                        dcc.Dropdown(
                            id='model-type',
                            options=[
                                {'label': 'Logistic Regression', 'value': 'logistic'},
                                {'label': 'Random Forest Classifier', 'value': 'rf_class'},
                                {'label': 'XGBoost Classifier', 'value': 'xgb_class'},
                                {'label': 'Linear Regression', 'value': 'linear'},
                                {'label': 'Random Forest Regressor', 'value': 'rf_reg'},
                                {'label': 'XGBoost Regressor', 'value': 'xgb_reg'}
                            ],
                            value='rf_class',
                            clearable=False
                        )
                    ], md=6),
                    dbc.Col([
                        html.Label("Target Column", style={"fontWeight": "400", "marginBottom": "0.5rem", "fontSize": "0.875rem"}),
                        dcc.Dropdown(
                            id='target-column',
                            options=[{'label': col, 'value': col} for col in columns],
                            value=columns[-1]
                        )
                    ], md=6)
                ], className="mb-3"),
                
                html.Div([
                    html.Label("Feature Columns", style={"fontWeight": "400", "marginBottom": "0.75rem", "fontSize": "0.875rem"}),
                    html.Div(id='feature-buttons')
                ], className="mb-3"),
                
                dbc.Button(
                    "Train Model",
                    id='train-button',
                    color="primary",
                    size="lg",
                    className="w-100"
                )
            ])
        ], className="card"),
        
        html.Div(id='model-results')
    ])


@app.callback(
    Output('feature-buttons', 'children'),
    [Input('target-column', 'value'),
     Input('stored-columns', 'data')]
)
def update_feature_buttons(target_col, columns):
    """Update feature selection buttons"""
    if not columns or not target_col:
        return html.Div()
    
    available_cols = [col for col in columns if col != target_col]
    
    buttons = []
    for col in available_cols:
        buttons.append(
            html.Button(
                col,
                id={'type': 'feature-btn', 'index': col},
                className='feature-button',
                n_clicks=0
            )
        )
    
    return html.Div(buttons, style={"display": "flex", "flexWrap": "wrap"})


@app.callback(
    Output('feature-columns', 'data'),
    Input({'type': 'feature-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
    [State('feature-columns', 'data'),
     State('stored-columns', 'data'),
     State('target-column', 'value')]
)
def toggle_features(n_clicks, current_features, columns, target_col):
    """Toggle feature selection"""
    if not dash.callback_context.triggered:
        return []
    
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if 'feature-btn' not in triggered_id:
        return current_features or []
    
    # Extract the column name from the triggered button
    import json
    button_id = json.loads(triggered_id.split('.')[0])
    col = button_id['index']
    
    current_features = current_features or []
    
    if col in current_features:
        current_features.remove(col)
    else:
        current_features.append(col)
    
    return current_features


@app.callback(
    Output('main-chart', 'figure'),
    [Input('chart-type', 'value'),
     Input('x-axis', 'value'),
     Input('y-axis', 'value'),
     Input('stored-data', 'data')]
)
def update_chart(chart_type, x_col, y_col, json_data):
    """Update visualization chart"""
    if json_data is None or x_col is None or y_col is None:
        return go.Figure()
    
    df = pd.read_json(json_data, orient='split')
    
    try:
        if chart_type == 'scatter':
            fig = px.scatter(df.head(100), x=x_col, y=y_col, title=f'{y_col} vs {x_col}')
        elif chart_type == 'line':
            fig = px.line(df.head(100), x=x_col, y=y_col, title=f'{y_col} over {x_col}')
        elif chart_type == 'bar':
            fig = px.bar(df.head(100), x=x_col, y=y_col, title=f'{y_col} by {x_col}')
        else:
            fig = go.Figure()
        
        fig.update_layout(
            template='plotly_white',
            font=dict(family="inherit", color="#000"),
            title_font_size=16,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    except Exception as e:
        return go.Figure()


@app.callback(
    Output('model-results', 'children'),
    Input('train-button', 'n_clicks'),
    [State('stored-data', 'data'),
     State('model-type', 'value'),
     State('target-column', 'value'),
     State('feature-columns', 'data')]
)
def train_model(n_clicks, json_data, model_type, target_col, feature_cols):
    """Train ML model"""
    if n_clicks is None or json_data is None or target_col is None or not feature_cols:
        return html.Div()
    
    try:
        df = pd.read_json(json_data, orient='split')
        
        # Prepare data
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target if categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model_map = {
            'logistic': LogisticRegression(max_iter=1000),
            'rf_class': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgb_class': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'linear': LinearRegression(),
            'rf_reg': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgb_reg': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        model = model_map[model_type]
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        model_names = {
            'logistic': 'Logistic Regression',
            'rf_class': 'Random Forest Classifier',
            'xgb_class': 'XGBoost Classifier',
            'linear': 'Linear Regression',
            'rf_reg': 'Random Forest Regressor',
            'xgb_reg': 'XGBoost Regressor'
        }
        
        return dbc.Card([
            dbc.CardBody([
                html.H6("Model Training Complete", className="mb-3", style={"fontWeight": "500"}),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Div("Training Score", style={"fontSize": "0.75rem", "color": "#666", "marginBottom": "0.5rem"}),
                            html.Div(f"{train_score*100:.1f}%", style={"fontSize": "1.5rem", "fontWeight": "600", "color": "#000"})
                        ], style={"textAlign": "center", "background": "white", "padding": "1rem", "border": "1px solid #e0e0e0", "borderRadius": "4px"})
                    ], md=6),
                    dbc.Col([
                        html.Div([
                            html.Div("Testing Score", style={"fontSize": "0.75rem", "color": "#666", "marginBottom": "0.5rem"}),
                            html.Div(f"{test_score*100:.1f}%", style={"fontSize": "1.5rem", "fontWeight": "600", "color": "#000"})
                        ], style={"textAlign": "center", "background": "white", "padding": "1rem", "border": "1px solid #e0e0e0", "borderRadius": "4px"})
                    ], md=6)
                ], className="mb-3"),
                
                html.Div([
                    html.Div(f"Model: {model_names[model_type]}", style={"marginBottom": "0.25rem"}),
                    html.Div(f"Features: {', '.join(feature_cols)}", style={"marginBottom": "0.25rem"}),
                    html.Div(f"Target: {target_col}", style={"marginBottom": "0.25rem"}),
                    html.Div(f"Training samples: {len(X_train)}", style={"marginBottom": "0.25rem"}),
                    html.Div(f"Testing samples: {len(X_test)}")
                ], style={"background": "#fafafa", "padding": "1rem", "border": "1px solid #e0e0e0", "borderRadius": "4px", "fontSize": "0.813rem", "fontFamily": "monospace"})
            ])
        ], className="card")
        
    except Exception as e:
        return html.Div(f"Error training model: {str(e)}", style={"color": "#666", "padding": "1rem", "border": "1px solid #e0e0e0"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)