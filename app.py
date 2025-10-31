import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import io, base64, pickle
from dash.exceptions import PreventUpdate

# ---------- Initialize ----------
app = dash.Dash(__name__)
app.title = "Interactive Data & Model Dashboard"

# ---------- Layout ----------
app.layout = html.Div(
    style={
        'fontFamily': 'Georgia, serif',
        'padding': '10px',
    },
    children=[
        html.Div([

            html.Div([
                dcc.Upload(
                    id='upload-data',
                    children=html.Button('Upload CSV File', style={
                        'padding': '10px 20px',
                        'fontSize': '14px',
                        'cursor': 'pointer',
                        'backgroundColor': '#ffffff',
                        'border': '1px solid #ccc',
                        'borderRadius': '5px'
                    }),
                    multiple=False,
                ),

                dcc.Dropdown(
                    id='chart-type',
                    options=[
                        {'label': 'Table', 'value': 'table'},
                        {'label': 'Scatter Plot', 'value': 'scatter'},
                        {'label': 'Box Plot', 'value': 'box'},
                        {'label': 'Bar Chart', 'value': 'bar'},
                        {'label': 'Histogram', 'value': 'histogram'},
                        {'label': 'Line Chart', 'value': 'line'},
                        {'label': 'Violin Plot', 'value': 'violin'}
                    ],
                    placeholder="Select visualization type...",
                    style={'width': '220px', 'marginRight': '10px'}
                ),

                dcc.Dropdown(id='x-axis', placeholder="Select X-axis", style={'width': '220px', 'marginRight': '10px'}),
                dcc.Dropdown(id='y-axis', placeholder="Select Y-axis", style={'width': '220px', 'marginRight': '10px'}),

            ], style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'gap': '8px',
                'marginBottom': '15px'
            }),

            html.Div(id='file-name', style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            html.Div(id='content-area'),
        ]),

        # ===== MODEL INFERENCE SECTION =====
        html.Div([
            html.H3("2️⃣ Load Pretrained Model", style={'marginBottom': '10px'}),

            html.Div([
                dcc.Upload(
                    id='upload-model',
                    children=html.Button('Upload Model File (.pkl)', style={
                        'padding': '10px 20px',
                        'fontSize': '14px',
                        'cursor': 'pointer',
                        'backgroundColor': '#ffffff',
                        'border': '1px solid #ccc',
                        'borderRadius': '5px'
                    }),
                    multiple=False
                ),
                dcc.Dropdown(
                    id='target-column',
                    placeholder="Select Target Column (optional for inference)",
                    style={'width': '250px', 'marginTop': '10px'}
                ),
                html.Button("Run Prediction", id='predict-btn', n_clicks=0,
                            style={'marginTop': '10px', 'padding': '8px 16px'}),
            ], style={'marginBottom': '10px'}),

            html.Div(id='model-status', style={'fontStyle': 'italic', 'color': '#555'}),
            html.Div(id='prediction-output', style={'marginTop': '20px'})
        ], style={
            'backgroundColor': '#ffffff',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        })
    ]
)

# ---------- Helper: Parse CSV ----------
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), on_bad_lines='skip')
        df.columns = df.columns.str.strip()
    except Exception as e:
        print("CSV Read Error:", e)
        return None, f"Error reading {filename}"
    return df, filename


# ---------- File Upload Callback ----------
@app.callback(
    Output('content-area', 'children'),
    Output('file-name', 'children'),
    Output('x-axis', 'options'),
    Output('y-axis', 'options'),
    Output('target-column', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    if contents is None:
        return "", "", [], [], []

    df, _ = parse_contents(contents, filename)
    if df is None:
        return html.Div("Error reading file."), filename, [], [], []

    options = [{'label': c, 'value': c} for c in df.columns]

    return (
        html.Div([
            html.Div(f"File uploaded: {filename}", style={'marginBottom': '10px'}),
            dcc.Store(id='stored-data', data=df.to_dict('records')),
            html.Div(id='display-area')
        ]),
        f"✅ {filename} uploaded successfully.",
        options, options, options
    )


# ---------- Visualization Callback ----------
@app.callback(
    Output('display-area', 'children'),
    Input('chart-type', 'value'),
    Input('x-axis', 'value'),
    Input('y-axis', 'value'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def update_visualization(chart_type, x_col, y_col, data):
    if data is None or chart_type is None:
        raise PreventUpdate

    df = pd.DataFrame(data)

    if chart_type == "table":
        return dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': c, 'id': c} for c in df.columns],
            page_size=12,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '14px'},
            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
        )

    if x_col is None:
        return html.Div("Please select an X-axis column.", style={'color': 'red'})

    try:
        if chart_type == "histogram":
            fig = px.histogram(df, x=x_col, title=f"Histogram of {x_col}")
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=y_col, title=f"{y_col} vs {x_col}")
        elif chart_type == "box":
            fig = px.box(df, x=x_col, y=y_col, color=x_col, title=f"Box Plot of {y_col} by {x_col}")
        elif chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=f"Bar Chart of {y_col} by {x_col}")
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart of {y_col} vs {x_col}")
        elif chart_type == "violin":
            fig = px.violin(df, x=x_col, y=y_col, color=x_col, title=f"Violin Plot of {y_col} by {x_col}")
        else:
            fig = px.scatter(title="Unsupported chart type selected")

        fig.update_layout(template="plotly_white", height=500)
        return dcc.Graph(figure=fig)
    except Exception as e:
        print("Visualization Error:", e)
        return html.Div("Error generating visualization.", style={'color': 'red'})


# ---------- Model Upload & Prediction ----------
@app.callback(
    Output('model-status', 'children'),
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('upload-model', 'contents'),
    State('stored-data', 'data'),
    State('target-column', 'value'),
    prevent_initial_call=True
)
def run_prediction(n_clicks, model_content, data, target_col):
    if model_content is None or data is None:
        raise PreventUpdate

    # Decode uploaded model
    content_type, content_string = model_content.split(',')
    decoded = base64.b64decode(content_string)

    try:
        model = pickle.loads(decoded)
    except Exception as e:
        return "❌ Failed to load model.", str(e)

    df = pd.DataFrame(data)
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()

    X = pd.get_dummies(X, drop_first=True)

    try:
        preds = model.predict(X)
    except Exception as e:
        return "⚠️ Model incompatible with this dataset.", str(e)

    output_df = pd.DataFrame({'Prediction': preds})
    return (
        "✅ Model loaded and predictions generated.",
        dash_table.DataTable(
            data=output_df.head(10).to_dict('records'),
            columns=[{'name': c, 'id': c} for c in output_df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '6px'},
            page_size=10
        )
    )


# ---------- Run ----------
if __name__ == '__main__':
    app.run(debug=True)
