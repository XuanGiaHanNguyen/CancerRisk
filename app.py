from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)

# Sample data
df = px.data.gapminder()

app.layout = html.Div([
    html.H2("Gapminder Data Dashboard"),
    dcc.Dropdown(
        id="country-dropdown",
        options=[{"label": c, "value": c} for c in df["country"].unique()],
        value="Canada"
    ),
    dcc.Graph(id="life-exp-graph")
])

@app.callback(
    Output("life-exp-graph", "figure"),
    Input("country-dropdown", "value")
)
def update_graph(country):
    filtered_df = df[df["country"] == country]
    fig = px.line(filtered_df, x="year", y="lifeExp", title=f"Life Expectancy in {country}")
    return fig

if __name__ == "__main__":
    app.run(debug=True)
