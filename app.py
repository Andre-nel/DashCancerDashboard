# Import the necessary libraries and load the data:
from utils import load_pickle, predict_diagnosis
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output, State
# from dash import callback_context
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
from dash.exceptions import PreventUpdate

# external_scripts = [
#     "https://code.jquery.com/jquery-3.6.0.min.js",
# ]

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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])  # , external_scripts=external_scripts

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
                                    dbc.Label(label, html_for=label),
                                    dbc.Input(
                                        type="number",
                                        id=f"feature-{i}",
                                        required=True,
                                    ),
                                ]
                            ),
                            md=6,
                        )
                        for label, i in zip(list(X_all.columns), range(30))
                    ]
                ),
                dbc.Button("Predict Diagnosis", id="predict-btn", color="primary"),
                dbc.Progress(id="prediction-progress", className="mt-5", style={"height": "30px"}),
                dcc.Graph(id='scatter-prediction', figure=px.line(
                    labels={'x': 'Prediction Number', 'y': 'Prediction for Malignancy (%)'},
                    title="Malignancy Prediction History"
                )),
            ],
            id="prediction-form",
        ),
    ]),
    html.Div(id='selected-point-index', style={'display': 'none'}),
    html.Div(id="prediction-result", style={"display": "none"}),
    html.Div(id="console-log", style={"display": "none"}),
    html.Div(id='clientside-output', style={'display': 'none'}),
],
    fluid=True,
)


# Create empty lists to store data
predictions = []
x_axis_values = []
metadata = []


# Update the scatter plot based on the selected features
@app.callback(
    Output('scatter-plot-original', 'figure'),
    [Input('x-axis-original', 'value'), Input('y-axis-original', 'value')]
)
def update_scatter_original(x_axis, y_axis):
    custom_color_scale = px.colors.qualitative.Plotly
    color_map = {'M': custom_color_scale[1], 'B': custom_color_scale[0]}  # Red for 'M', Blue for 'B'
    return px.scatter(data_frame=original_data, x=x_axis, y=y_axis, color='diagnosis',
                      title="Original Feature Relationships", labels={'diagnosis': 'Diagnosis'},
                      hover_data=X_all.columns, height=500,
                      color_discrete_map=color_map)  # Add color_discrete_map parameter


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


for i in range(30):
    @app.callback(
        Output(f"feature-{i}", "value"),
        [Input('selected-point-index', 'children')],
        [State(f"feature-{i}", "value")]
    )
    def update_input_field(value, current_value, index=i):
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
    global metadata

    if n_clicks is None and None in features:
        raise PreventUpdate

    features = list(map(float, features))

    point_meta = dict(zip(list(X_original.columns), features))
    metadata.append(point_meta)

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
    Output('scatter-prediction', 'figure'),
    [Input('prediction-result', 'children')]
)
def update_graphs(input_value):
    global predictions, x_axis_values, metadata
    # Parse the input_value string to extract the new_prediction value
    start = input_value.find(" ") + 1
    end = input_value.find("]]")
    new_prediction = float(input_value[start:end])

    # Update the graph
    predictions.append(new_prediction)
    if not x_axis_values:
        x_value = 1
    else:
        x_value = x_axis_values[-1] + 1
    x_axis_values.append(x_value)

    # Create a DataFrame for the line graph
    metadata_columns = list(metadata[0].keys()) if len(metadata) > 0 else []
    prediction_df = pd.DataFrame({'Prediction Number': x_axis_values,
                                  'Prediction for Malignancy': predictions,
                                  **{col: [item[col] for item in metadata] for col in metadata_columns}})

    # Add a new column for risk category
    prediction_df['Risk Category'] = prediction_df['Prediction for Malignancy'].apply(
                                        lambda x: 'High' if x > 0.5 else 'Low')

    custom_color_scale = px.colors.qualitative.Plotly
    # Red for 'High' risk, Blue for 'Low' risk
    color_map = {'High': custom_color_scale[1], 'Low': custom_color_scale[0]}

    # Return a plotly.express graph
    return px.scatter(data_frame=prediction_df, x='Prediction Number', y='Prediction for Malignancy',
                      title="Malignancy Prediction History", color='Risk Category',
                      height=500, hover_data=metadata_columns,
                      color_discrete_map=color_map)


if __name__ == "__main__":
    # Run the app
    # app.run_server(mode='external', port=8050)
    app.run_server(debug=True)
