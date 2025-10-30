from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

df = pd.read_csv('cpi.csv')

app = Dash(__name__)
server = app.server

# (Optional) Force dark or light theme via body class used by style.css variables.
# Comment out if you prefer OS auto (prefers-color-scheme).
app.index_string = """
<!DOCTYPE html>
<html>
  <head>{%metas%}{%favicon%}{%css%}</head>
  <body class="dark">
    <div id="react-entry-point">
      {%app_entry%}
    </div>
    <footer>
      {%config%}{%scripts%}{%renderer%}
    </footer>
  </body>
</html>
"""

app.layout = html.Div(
    className="container",
    children=[
        html.H1('Inflation i Danmark 1980 - 2025'),

        # Dropdown wrapper (width & spacing handled in CSS)
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="col",
                    children=[
                        dcc.Dropdown(
                            df.Variable.unique(),
                            'INDHOLD',
                            id='dropdown-selection'
                        )
                    ]
                )
            ]
        ),

        # Two graphs (layout handled by .row / .col / .card in CSS)
        html.Div(
            className="row",
            children=[
                html.Div(className="col card", children=[dcc.Graph(id='graph-content')]),
                html.Div(className="col card", children=[dcc.Graph(id='graph-content-2')]),
            ]
        ),
    ]
)

@callback(
    Output('graph-content', 'figure'),
    Output('graph-content-2', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graphs(value):
    dff = df[df.Variable == value]

    # No figure-level theming or colors here; let CSS style containers.
    fig1 = px.line(dff, x='TID', y='Value', title='Inflation over tid')
    fig2 = px.bar(dff, x='TID', y='Value', title='Søjlediagram af inflation')

    return fig1, fig2

if __name__ == '__main__':
    app.run(debug=True)
