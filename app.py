# Import the necessary libraries and load the data:
from utils import load_pickle, predict_diagnosis
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output, State
# from dash import callback_context
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
from dash.exceptions import PreventUpdate
import numpy as np
import ast

external_scripts = [
    "https://code.jquery.com/jquery-3.6.0.min.js",
]

# from pyngrok import ngrok
# from jupyter_dash import JupyterDash

# Set the ngrok authtoken
# ngrok.set_auth_token("2P4d1RHfUpQVrhLrNp0vlYJDt69_28fyJTnWGUeYmAKfGEZNJ")


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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], external_scripts=external_scripts)

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
                            html.Div(
                                [
                                    dbc.Label(f"Feature {i + 1}", html_for=f"feature-{i}"),
                                    dbc.Input(
                                        type="number",
                                        id=f"feature-{i}",
                                        required=True,
                                    ),
                                ]
                            ),
                            md=6,
                        )
                        for i in range(30)
                    ]
                ),
                dcc.Interval(id="step-interval", interval=1000, n_intervals=0),
                dbc.Button("Predict Diagnosis", id="predict-btn", color="primary"),
                dbc.Progress(id="prediction-progress", className="mt-5", style={"height": "30px"}),
                dcc.Graph(id='line-graph'),
            ],
            id="prediction-form",
        ),
    ]),
    html.Div(id='selected-point-index', style={'display': 'none'}),
    html.Div(id="prediction-result", style={"display": "none"}),
    html.Div(id="console-log", style={"display": "none"}),
    html.Div(id='up-arrow-triggered-field', style={'display': 'none'}),
    html.Div(id='down-arrow-triggered-field', style={'display': 'none'}),
    html.Div(id='clientside-output', style={'display': 'none'}),
    html.Div(id='detect-arrow-key', style={'display': 'none'}),
    html.Div(id='step-called', style={'display': 'none'}),

],
    fluid=True,
)


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
    if clickData is None:
        return None
    values_to_search = clickData['points'][0]['customdata']
    # Iterate over the rows of the DataFrame
    for idx, row in original_data.iterrows():
        # Convert row values and search values to sets of strings
        row_set = set(row.values)
        search_values_set = set(values_to_search)
        # Check if the search_values_set is a subset of row_set
        if search_values_set.issubset(row_set):
            return idx
    return None


@app.callback(
    Output("step-called", "children"),
    [Input("step-interval", "n_intervals")],
)
def update_step_size(n_intervals):
    return n_intervals


for i in range(30):
    @app.callback(
        Output(f"feature-{i}", "value"),
        [Input('selected-point-index', 'children'),
         Input("up-arrow-triggered-field", "children"),
         Input("down-arrow-triggered-field", "children")],
        [State(f"feature-{i}", "value")]
    )
    def update_input_field(value, up_arrow_triggered_field, down_arrow_triggered_field, current_value, index=i):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if triggered_id == 'selected-point-index':
            if value is not None and index is not None:
                selected_point = X_original.iloc[int(value)].values
                return selected_point[index]
            raise PreventUpdate
        else:
            if (current_value and
                ((up_arrow_triggered_field and int(up_arrow_triggered_field) == index and
                  triggered_id == "up-arrow-triggered-field") or
                 (down_arrow_triggered_field and int(down_arrow_triggered_field) == index and
                    triggered_id == "down-arrow-triggered-field"))):

                factor = 1.05 if triggered_id == "up-arrow-triggered-field" else 0.95
                new_value = current_value * factor
                return new_value

            raise PreventUpdate


@app.callback(
    Output("prediction-progress", "children"),
    Output("prediction-progress", "value"),
    Output("prediction-progress", "style"),
    Output("prediction-result", "children"),  # Add this line
    Input("predict-btn", "n_clicks"),
    [Input(f"feature-{i}", "value") for i in range(30)],
)
def update_prediction(n_clicks, *features):
    if n_clicks is None and None in features:
        raise PreventUpdate

    features = list(map(float, features))
    prediction_proba = predict_diagnosis(model_all, features, pipeline_fit_to_all_features_path)
    malignant_proba = prediction_proba[0][1] * 100

    return f"Malignant: {malignant_proba:.2f}%", malignant_proba, {"height": "30px"}, str(prediction_proba)


app.clientside_callback(
    """
    function log_prediction_proba(n_clicks, prediction_proba) {
        if (n_clicks) {
            console.log('prediction_proba:', prediction_proba);
        }
    }
    """,
    Output("console-log", "children"),  # Update this line
    Input("predict-btn", "n_clicks"),
    Input("prediction-result", "children"),
)


# The callback function that updates both the scatter plot and the line graph
@app.callback(
    Output('line-graph', 'figure'),
    [Input('prediction-result', 'children')]
)
def update_graphs(input_value):
    # Parse the input_value string to extract the new_prediction value
    print(input_value)
    start = input_value.find(" ") + 1
    end = input_value.find("]]")
    new_prediction = float(input_value[start:end])
    print(new_prediction)

    # Update the line graph
    add_prediction(new_prediction)

    # Create a DataFrame for the line graph
    line_graph_df = pd.DataFrame({'index': x_axis_values, 'predictions': predictions})

    print(line_graph_df)

    # Return the updated line_graph as a plotly.express graph
    return px.line(data_frame=line_graph_df, x='index', y='predictions',
                   title="Prediction Line Graph", labels={'predictions': 'Predictions', 'index': 'Index'},
                   height=500)


# Step 1: Create empty lists to store data
predictions = []
x_axis_values = []


# Step 2: Add a new prediction and its corresponding x-axis value
def add_prediction(new_prediction):
    predictions.append(new_prediction)
    if not x_axis_values:
        x_value = 1
    else:
        x_value = x_axis_values[-1] + 1
    x_axis_values.append(x_value)


if __name__ == "__main__":
    # Run the app
    # app.run_server(mode='external', port=8050)
    app.run_server(debug=True)
