# Import the necessary libraries and load the data:
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output
# from dash import callback_context
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
# from pyngrok import ngrok
# from jupyter_dash import JupyterDash

# Set the ngrok authtoken
# ngrok.set_auth_token("2P4d1RHfUpQVrhLrNp0vlYJDt69_28fyJTnWGUeYmAKfGEZNJ")

from utils import load_pickle, predict_diagnosis

root_path = Path(__file__).parent
data_dir = root_path / "data"
pickle_dir = root_path / "pickled"

# Load the data
preprocessed_all_features_data = pd.read_csv(data_dir / "preprocessed_cancer_data_all_features.csv")
original_data = pd.read_csv(data_dir / "BreastCancer.csv")
path_to_log_reg_model_all_features = pickle_dir / "log_reg_model_all_features.pkl"
pipeline_fit_to_all_features_path = pickle_dir / "preprocessor_pipeline_all_features.pkl"

# Prepare the data for visualization:
# Separate features and target
X_all = preprocessed_all_features_data.drop(columns=['diagnosis'])
y_all = preprocessed_all_features_data['diagnosis']

X_original = original_data.drop(columns=["diagnosis", "Unnamed: 32", "id"])
y_original = original_data['diagnosis']

# Split the data into train and test sets
X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
    X_original, y_original, test_size=0.2, random_state=42)

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42)

# Load the model and make predictions
model_all = load_pickle(path_to_log_reg_model_all_features)
y_pred_all = model_all.predict(X_test_all)


# Calculate model performance metrics
accuracy_all = accuracy_score(y_test_all, y_pred_all)
precision_all = precision_score(y_test_all, y_pred_all)
recall_all = recall_score(y_test_all, y_pred_all)
f1_all = f1_score(y_test_all, y_pred_all)
roc_auc_all = roc_auc_score(y_test_all, y_pred_all)

# Start the ngrok tunnel
# public_url = ngrok.connect(addr="8050")
# print("Your Dash app is accessible at:", public_url)

# Create the app layout with the visualizations:
# app = dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Cancer Diagnosis Prediction Dashboard"), className="text-center")
    ], className="mt-4"),

    # Feature Importance Visualization
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='feature-importance-all', figure=px.bar(x=X_all.columns,
                      y=model_all.coef_[0], labels={'x': 'Features', 'y': 'Importance'},
                      title="Feature Importance (all)"))
        ])
    ], className="mt-4"),

    # Original Feature Relationships Visualization
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='x-axis-original', options=[{'label': i, 'value': i}
                         for i in X_original.columns], value='radius_mean', placeholder="Select Feature 1"),
            dcc.Dropdown(id='y-axis-original', options=[{'label': i, 'value': i}
                         for i in X_original.columns], value='texture_mean', placeholder="Select Feature 2"),
            dcc.Graph(id='scatter-plot-original')
        ])
    ], className="mt-4"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='performance-metrics-all', figure=px.bar(
                x=[accuracy_all, precision_all, recall_all, f1_all, roc_auc_all],
                y=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'],
                orientation='h',
                labels={'x': 'Metric Value', 'y': 'Metrics'},
                title="All features kept Model Performance Metrics",
                text=[round(accuracy_all, 2), round(precision_all, 2), round(
                    recall_all, 2), round(f1_all, 2), round(roc_auc_all, 2)]
            ))
        ])
    ], className="mt-4"),

    html.Div([
        html.H1("Cancer Diagnosis Prediction Dashboard", className="text-center mt-4"),
        dbc.Form(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div([
                                dbc.Label(f"Feature {i + 1}", html_for=f"feature-{i}"),
                                dbc.Input(type="number", id=f"feature-{i}", step="1e-18", required=True)
                            ]),
                            md=6,
                        )
                        for i in range(30)
                    ]
                ),
                dbc.Button("Predict Diagnosis", id="predict-btn", color="primary"),
                dbc.Progress(id="prediction-progress", className="mt-5", style={"height": "30px"})
            ],
            id="prediction-form",
        ),
    ]),
    html.Div(id='selected-point-index', style={'display': 'none'}),
    html.Div(id="prediction-result", style={"display": "none"}),
    html.Div(id="console-log-dummy", style={"display": "none"}),  # Add this line
])

# Update the scatter plot based on the selected features


@app.callback(
    Output('scatter-plot-original', 'figure'),
    [Input('x-axis-original', 'value'), Input('y-axis-original', 'value')]
)
def update_scatter_original(x_axis, y_axis):
    return px.scatter(data_frame=original_data, x=x_axis, y=y_axis, color='diagnosis',
                      title="Original Feature Relationships", labels={'diagnosis': 'Diagnosis'},
                      hover_data=X_all.columns, height=500)


@app.callback(
    Output('selected-point-index', 'children'),
    [Input('scatter-plot-original', 'clickData')]
)
def update_selected_point_index(clickData):
    if clickData:
        return clickData['points'][0]['pointIndex']
    return None


for i in range(30):
    @app.callback(
        Output(f"feature-{i}", "value"),
        [Input('selected-point-index', 'children')]
    )
    def update_input_field(value, index=i):
        if value is not None:
            selected_point = X_original.iloc[int(value)].values
            return selected_point[index]
        return None


@app.callback(
    Output("prediction-progress", "children"),
    Output("prediction-progress", "value"),
    Output("prediction-progress", "style"),
    Output("prediction-result", "children"),  # Add this line
    Input("predict-btn", "n_clicks"),
    [Input(f"feature-{i}", "value") for i in range(30)],
)
def update_prediction(n_clicks, *features):
    if n_clicks is not None and None not in features:
        features = list(map(float, features))
        prediction_proba = predict_diagnosis(model_all, features, pipeline_fit_to_all_features_path)
        print(prediction_proba)
        malignant_proba = prediction_proba[0][1] * 100

        return f"Malignant: {malignant_proba:.2f}%", malignant_proba, {"height": "30px"}, str(prediction_proba)

    return "Malignant: 0%", 0, {"height": "30px"}, ""


app.clientside_callback(
    """
    function log_prediction_proba(n_clicks, prediction_proba) {
        if (n_clicks) {
            console.log('prediction_proba:', prediction_proba);
        }
    }
    """,
    Output("console-log-dummy", "children"),  # Update this line
    Input("predict-btn", "n_clicks"),
    Input("prediction-result", "children"),
)

if __name__ == "__main__":
    # Run the app
    # app.run_server(mode='external', port=8050)
    app.run_server(debug=True)
