import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import io, base64

# ---------- Initialize ----------
app = dash.Dash(__name__)
app.title = "Dashboard"

# ---------- Layout ----------
app.layout = html.Div(
    style={'fontFamily': 'serif', 'padding': '20px'},
    children=[
        # Upload + controls row
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload CSV File', style={
                    'padding': '10px 20px',
                    'fontSize': '14px',
                    'cursor': 'pointer',
                    'backgroundColor': '#f8f9fa',
                    'color': '#333',
                    'border': '1px solid #ced4da',
                    'borderRadius': '3px',
                    'marginRight': '10px'
                }),
                multiple=False,
            ),

            # Dropdown for chart type
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
                style={'width': '200px', 'marginRight': '10px'}
            ),

            # Dropdowns for X and Y axes
            dcc.Dropdown(
                id='x-axis',
                placeholder="Select X-axis",
                style={'width': '200px', 'marginRight': '10px'}
            ),

            dcc.Dropdown(
                id='y-axis',
                placeholder="Select Y-axis",
                style={'width': '200px', 'marginRight': '10px'}
            ),

            html.Div(id='file-name', style={
                'marginTop': '10px',
                'fontWeight': 'bold'
            }),
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'gap': '6px',
            'marginBottom': '10px',
            'flexWrap': 'wrap'
        }),

        html.Div(id='content-area')  # Dynamic display area
    ]
)


# ---------- Helper: Parse CSV ----------
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=',', engine='python', on_bad_lines='skip')
        df.columns = df.columns.str.strip().str.lower()
    except Exception as e:
        print("CSV Read Error:", e)
        return None, f"Error reading file: {filename}"

    return df, filename


# ---------- Upload Callback ----------
@app.callback(
    Output('content-area', 'children'),
    Output('file-name', 'children'),
    Output('x-axis', 'options'),
    Output('y-axis', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    if contents is None:
        return "", "", [], []

    df, _ = parse_contents(contents, filename)
    if df is None:
        return html.Div("Error reading file."), filename, [], []

    options = [{'label': col.title(), 'value': col} for col in df.columns]

    return (
        html.Div([
            html.Div(f"File uploaded: {filename}"),
            dcc.Store(id='stored-data', data=df.to_dict('records')),
            html.Div(id='display-area')
        ]),
        f"",
        options,
        options
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
        return ""

    df = pd.DataFrame(data)

    # Table view
    if chart_type == "table":
        return dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': c.title(), 'id': c} for c in df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '14px'},
            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
        )

    # Require X-axis for charts
    if x_col is None:
        return html.Div("Please select an X-axis column.", style={'color': 'red'})

    try:
        # Auto-handle missing Y-axis for some types
        if chart_type == "histogram":
            fig = px.histogram(df, x=x_col, title=f"Histogram of {x_col.title()}")
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=y_col, title=f"{y_col.title()} vs {x_col.title()}")
        elif chart_type == "box":
            fig = px.box(df, x=x_col, y=y_col, color=x_col, title=f"Box Plot of {y_col.title()} by {x_col.title()}")
        elif chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=f"Bar Chart of {y_col.title()} by {x_col.title()}")
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart of {y_col.title()} vs {x_col.title()}")
        elif chart_type == "violin":
            fig = px.violin(df, x=x_col, y=y_col, color=x_col, title=f"Violin Plot of {y_col.title()} by {x_col.title()}")
        else:
            fig = px.scatter(title="Unsupported chart type selected")

        fig.update_layout(template="plotly_white", height=500)
        return dcc.Graph(figure=fig)

    except Exception as e:
        print("⚠️ Visualization Error:", e)
        return html.Div("Error generating visualization.", style={'color': 'red'})


# ---------- Run ----------
if __name__ == '__main__':
    app.run(debug=True)
