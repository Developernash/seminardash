import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc

# ------------- Data: synthetic 'credit' style dataset -------------
np.random.seed(42)
n_samples = 32000
n_features = 11  # IV1..IV11 (like the screenshot)

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=8,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    class_sep=1.1,
    weights=[0.7, 0.3],
    random_state=42,
)

feature_names = [f"IV {i+1}" for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['DV'] = y

# Train/Test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['DV'])

# ------------- Helpers -------------
def corr_heatmap_figure(df_features: pd.DataFrame) -> go.Figure:
    corr = df_features.corr()
    fig = px.imshow(
        corr,
        x=corr.columns,
        y=corr.columns,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        coloraxis_colorbar=dict(title="Corr.")
    )
    return fig

def pca_figures(df_features: pd.DataFrame, labels: np.ndarray):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df_features.values)
    pca2 = PCA(n_components=2, random_state=42)
    comp2 = pca2.fit_transform(Xs)
    pca3 = PCA(n_components=3, random_state=42)
    comp3 = pca3.fit_transform(Xs)

    fig2d = px.scatter(
        x=comp2[:,0], y=comp2[:,1],
        color = np.where(labels == 1, 'Default', 'Non-default'),
        labels={'x':'PC1','y':'PC2','color':'Class'},
        opacity=0.7
    )
    fig2d.update_layout(margin=dict(l=10,r=10,t=30,b=10))

    fig3d = px.scatter_3d(
        x=comp3[:,0], y=comp3[:,1], z=comp3[:,2],
        color = np.where(labels == 0, 'Non-default',  'Default'),
        labels={'x':'PC1','y':'PC2','z':'PC3','color':'Class'},
        opacity=0.7
    )
    fig3d.update_layout(margin=dict(l=10,r=10,t=30,b=10))
    return fig2d, fig3d

def mlp_and_results(train_df, test_df, hidden_layers):
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(train_df[feature_names].values)
    ytrain = train_df['DV'].values
    Xtest = scaler.transform(test_df[feature_names].values)
    ytest = test_df['DV'].values

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        random_state=42,
        max_iter=300,
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        alpha=0.0001,
    )
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    acc = accuracy_score(ytest, ypred)
    cm = confusion_matrix(ytest, ypred, labels=[0,1])

    return clf, acc, cm, scaler

def network_figure(model: MLPClassifier, scaler: StandardScaler):
    # Build a simple layered network diagram using coefs_ and intercepts_
    # Layers: input (n_features), hidden(s), output (2 classes)
    if not hasattr(model, "coefs_"):
        return go.Figure()

    layer_sizes = [len(model.coefs_[0])]  # input size equals number of nodes in first weight matrix's rows
    for w in model.coefs_:
        layer_sizes.append(len(w[0]))  # columns = next layer size
    # layer_sizes now is [input, h1, h2, ..., output]

    # positions
    x_gap = 1.0 / (len(layer_sizes) - 1)
    xs, ys = [], []
    node_positions = []
    for li, sz in enumerate(layer_sizes):
        x = li * x_gap
        # distribute nodes vertically
        y_positions = np.linspace(0, 1, sz)
        for y in y_positions:
            xs.append(x); ys.append(y)
            node_positions.append((x, y))

    # edges derived from coefs_
    edge_x, edge_y, edge_width, edge_color = [], [], [], []
    max_abs = max(np.abs(w).max() for w in model.coefs_)
    for li, w in enumerate(model.coefs_):
        left_start = sum(layer_sizes[:li])
        right_start = sum(layer_sizes[:li+1])
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                x0, y0 = node_positions[left_start + i]
                x1, y1 = node_positions[right_start + j]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
                wt = w[i, j]
                edge_width.append(0.5 + 3.5 * (abs(wt)/max_abs if max_abs>0 else 0))
                edge_color.append(wt)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1),
        hoverinfo="none",
        showlegend=False,
        marker=dict(color=edge_color, colorscale="Blues", cmin=-max_abs, cmax=max_abs),
    )

    node_trace = go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(size=10, color="grey"),
        hoverinfo="text",
        text=[f"Layer {li} Node {ni}" for li, sz in enumerate(layer_sizes) for ni in range(sz)],
        showlegend=False
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Neural network structure (edge color/width ~ weight magnitude)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white"
    )
    return fig

def confusion_matrix_figure(cm: np.ndarray) -> go.Figure:
    # cm layout: rows true [0,1], cols pred [0,1]
    z = cm.astype(int)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Pred. ND", "Pred. D"],
        y=["Act. ND", "Act. D"],
        text=z,
        texttemplate="%{text}",
        colorscale=[
            [0.0, "#b71c1c"],  # dark red
            [0.5, "#ef9a9a"],  # light red
            [0.5, "#a5d6a7"],  # light green
            [1.0, "#1b5e20"],  # dark green
        ],
        showscale=False
    ))
    fig.update_layout(
        margin=dict(l=10,r=10,t=30,b=10),
    )
    return fig

# ------------- App -------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "A Machine learning approach to credit risk assessment"

controls = dbc.Card(
    [
        html.Label("Select the number of hidden layers in the network"),
        dcc.Slider(id="n_layers", min=1, max=6, step=1, value=5,
                   marks={i: f"{i} hidden layer{'s' if i>1 else ''}" for i in range(1,7)}),
        html.Br(),
        html.Label("Select the number of neurons per layer"),
        dcc.Slider(id="n_neurons", min=4, max=32, step=2, value=8,
                   marks={i: f"{i} neurons" for i in range(4,33,4)}),
    ],
    body=True,
    className="mb-3 shadow-sm rounded-4"
)

left_text = dbc.Card(
    dbc.CardBody([
        html.H5("Information used", className="card-title"),
        html.P("The following application trains a small neural network to classify whether a loan will default."),
        html.Ul([
            html.Li("Age (IV 1)"),
            html.Li("Annual Income (IV 2)"),
            html.Li("Home ownership (IV 3)"),
            html.Li("Employment length (IV 4)"),
            html.Li("Loan intent (IV 5)"),
            html.Li("Loan grade (IV 6)"),
            html.Li("Loan amount (IV 7)"),
            html.Li("Interest rate (IV 8)"),
            html.Li("Percent income (IV 9)"),
            html.Li("Historical default (IV 10)"),
            html.Li("Credit history length (IV 11)"),
        ]),
        html.Small("Dataset here is synthetic and for demo only."),
    ]),
    className="mb-3 shadow-sm rounded-4"
)

# initial figures
corr_fig = corr_heatmap_figure(df[feature_names])
p2d, p3d = pca_figures(df[feature_names], df['DV'].values)

# Layout
app.layout = dbc.Container(
    [
        html.H2("A Machine learning approach to credit risk assessment", className="mt-3 mb-4 text-center"),
        dbc.Row([
            dbc.Col([left_text], md=4, lg=3),
            dbc.Col([dcc.Graph(figure=corr_fig, id="corr_fig")], md=8, lg=5),
            dbc.Col([controls], md=12, lg=4),
        ], align="start"),
        dbc.Row([
            dbc.Col([
                html.H6("Two-dimensional PCA (red=Default, green=Non-default)"),
                dcc.Graph(id="pca2d", figure=p2d)
            ], md=6),
            dbc.Col([
                html.H6("Three-dimensional PCA"),
                dcc.Graph(id="pca3d", figure=p3d)
            ], md=6),
        ])
        # dbc.Row([
        #     dbc.Col([
        #         html.Div(id="model_accuracy", className="fw-bold mb-2"),
        #         dcc.Graph(id="conf_matrix")
        #     ], md=4),
        #     dbc.Col([dcc.Graph(id="network_fig")], md=8),
        # ], className="mb-4"),
        # html.Div(className="text-muted mb-4", children=[
        #     html.Small("Tip: adjust the sliders to change the network size and retrain the model.")
        # ])
    ],
    fluid=True
)

@callback(
    Output("model_accuracy", "children"),
    Output("conf_matrix", "figure"),
    Output("network_fig", "figure"),
    Input("n_layers", "value"),
    Input("n_neurons", "value"),
)
def update_model(n_layers, n_neurons):
    hidden = tuple([n_neurons]*n_layers)
    model, acc, cm, scaler = mlp_and_results(train_df, test_df, hidden)

    accuracy_text = f"Total accuracy of the model is {acc*100:.2f}%"
    cm_fig = confusion_matrix_figure(cm)
    net_fig = network_figure(model, scaler)

    return accuracy_text, cm_fig, net_fig

if __name__ == "__main__":
    app.run(debug=True)
