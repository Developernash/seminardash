from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Credit Risk Dashboard Layout"

# Helper function to make a 3-column "page"
def make_page(title_text):
    return html.Div(
        children=[
            html.H3(
                title_text,
                style={"color": "#660000", "marginBottom": "25px"}
            ),

            html.Div(
                style={
                    "backgroundColor": "white",
                    "boxShadow": "0px 4px 12px rgba(0,0,0,0.1)",
                    "borderRadius": "10px",
                    "width": "90%",
                    "height": "120vh",
                    "display": "flex",
                    "justifyContent": "space-around",
                    "padding": "10px",
                    "overflow": "hidden",
                    "marginBottom": "60px",
                },
                children=[
                    # Column 1: text
                    html.Div(
                        style={"width": "40%", "overflowY": "auto"},
                        children=[
                            html.H5("Information Used", style={"color": "#333"}),
                            html.P(
                                "We train a neural network to classify loan defaults using variables "
                                "such as age, income, and credit history length."
                            ),
                            html.Ul([
                                html.Li("Age (IV1)"),
                                html.Li("Annual Income (IV2)"),
                                html.Li("Loan Amount (IV7)"),
                                html.Li("Credit History (IV11)")
                            ]),
                            html.P(
                                "Dataset: 32,000 borrowers split 80/20 for training and testing.",
                                style={"fontSize": "14px", "color": "#555"}
                            ),
                        ]
                    ),

                    # Column 2: figure placeholder
                    html.Div(
                        style={"width": "30%", "textAlign": "center"},
                        children=[
                            html.H5("Model Diagram", style={"color": "#333"}),
                            html.Div(
                                "📈 Figure placeholder",
                                style={
                                    "border": "1px dashed #ccc",
                                    "borderRadius": "8px",
                                    "height": "80%",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "color": "#aaa"
                                }
                            )
                        ]
                    ),

                    # Column 3: equations or results
                    html.Div(
                        style={"width": "30%", "overflowY": "auto"},
                        children=[
                            html.H5("Model Equation", style={"color": "#333"}),
                            dcc.Markdown(r"""
                            The model predicts the probability of default as:

                            $$
                            P(D=1) = f(W_2 \cdot \sigma(W_1 X + b_1) + b_2)
                            $$

                            where:
                            - $f(\cdot)$ is the activation function (ReLU),
                            - $W_1, W_2$ are weight matrices,
                            - $b_1, b_2$ are biases.
                            """, mathjax=True),
                            html.Hr(),
                            html.H5("Results"),
                            html.P("Accuracy: **89.65%**", style={"color": "#006600"}),
                        ]
                    )
                ]
            )
        ]
    )


# Main app layout: stack 3 pages
app.layout = html.Div(
    style={
        "backgroundColor": "#f9f9f9",
        "fontFamily": "Georgia, serif",
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "center",
        "justifyContent": "flex-start",
        "padding": "20px",
    },
    children=[
        make_page("A First Machine Learning Approach to Credit Risk Assessment"),
        make_page("Second Type of Approach That Is Yet to Be Determined"),
        make_page("Third Assessment Approach to Credit Risk and Other Iffy Subjects"),
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
