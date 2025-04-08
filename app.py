from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import folium
from folium.plugins import HeatMap
import plotly.graph_objects as go
from recommendation import recommend_properties_with_scores
import joblib
import plotly.express as px
# import plotly.utils
import base64
import json
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from wordcloud import WordCloud
import zlib


app = Flask(__name__)
# Load Data and Model
with open("PickleFile/df.pkl", "rb") as file:
    df = pickle.load(file)
# with open("PickleFile/pipeline.pkl", "rb") as file:
#     pipeline = pickle.load(file)
# with open("PickleFile/pipeline.pkl", "rb") as file:
#     pipeline = pickle.load(file)


try:
    with open('PickleFile/pipeline2.pkl', "rb") as file:
        pipeline = joblib.load(file)
except zlib.error as e:
    raise RuntimeError("The pipeline file is corrupted or improperly compressed. Please re-save it using joblib with compression.") from e

# import os
# import pickle
# import requests

# # Directory to store the downloaded model
# MODEL_DIR = "PickleFile"
# MODEL_PATH = os.path.join(MODEL_DIR, "pipeline.pkl")

# # Modified Dropbox link for direct download
# DROPBOX_URL = "https://www.dropbox.com/scl/fi/a8f4u80ii291z139in1z9/pipeline.pkl?rlkey=btn3qsv37f3vzkmcbu1oni02c&st=auoieh52&dl=1"

# # Step 1: Download the file only if it doesn't exist
# if not os.path.exists(MODEL_PATH):
#     os.makedirs(MODEL_DIR, exist_ok=True)
#     print("Downloading model from Dropbox...")
#     response = requests.get(DROPBOX_URL)
#     if response.status_code == 200:
#         with open(MODEL_PATH, "wb") as f:
#             f.write(response.content)
#         print("✅ Model downloaded successfully.")
#     else:
#         raise Exception(f"❌ Download failed with status code: {response.status_code}")

# # Step 2: Load the model using pickle
# with open(MODEL_PATH, "rb") as file:
#     pipeline = pickle.load(file)
#     print("✅ Model loaded successfully.")



with open("PickleFile/location_distance.pkl", "rb") as file:
    location_df = pickle.load(file)

new_df = pd.read_csv('data/data_viz1.csv')
# feature_text = pickle.load(open('data/feature_text.pkl', 'rb'))
with open("data/feature_text.pkl", "rb") as file:
    feature_text = pickle.load(file)

@app.route("/")
def home():
    """Render the home page with dropdown options."""
    # print("hello3")
    return render_template(
        "index.html",
        sectors=sorted(df["sector"].unique().tolist()),
        bedrooms=sorted(df["bedRoom"].unique().tolist()),
        bathrooms=sorted(df["bathroom"].unique().tolist()),
        balconies=sorted(df["balcony"].unique().tolist()),
        property_ages=sorted(df["agePossession"].unique().tolist()),
        furnishing_types=sorted(df["furnishing_type"].unique().tolist()),
        luxury_categories=sorted(df["luxury_category"].unique().tolist()),
        floor_categories=sorted(df["floor_category"].unique().tolist()),
        location = sorted(location_df.columns.unique().to_list())
        # apartments = sorted(location_df.index.unique().to_list())

    
    )

@app.route("/predict", methods=["POST" , "GET"])
def predict():
    """Predicts the price of the property based on user input."""
    try:
        data = request.form
        required_fields = [
            "property_type", "sector", "bedroom", "bathroom", "balcony",
            "property_age", "built_up_area", "servant_room", "store_room",
            "furnishing_type", "luxury_category", "floor_category"
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        # Convert inputs to the correct type
        try:
            input_data = [[
                data["property_type"],
                data["sector"],
                int(data["bedroom"]),   # Ensure integer
                int(data["bathroom"]),  # Ensure integer
                data["balcony"],
                data["property_age"],
                float(data["built_up_area"]),  # Ensure float
                int(data["servant_room"]),  # Ensure integer
                int(data["store_room"]),  # Ensure integer
                data["furnishing_type"],
                data["luxury_category"],
                data["floor_category"]
            ]]
        except ValueError as ve:
            return jsonify({"error": f"Invalid input type: {ve}"}), 400

        # Convert input to DataFrame
        columns = [
            "property_type", "sector", "bedRoom", "bathroom", "balcony",
            "agePossession", "built_up_area", "servant room", "store room",
            "furnishing_type", "luxury_category", "floor_category"
        ]
        one_df = pd.DataFrame(input_data, columns=columns)
        predicted_price = pipeline.predict(one_df)
        base_price = np.expm1(predicted_price)[0]  # Apply inverse log transformation
        low = round(base_price - 0.22, 2)
        high = round(base_price + 0.22, 2)
        return jsonify({"prediction": f"The price of the flat is between {low} Cr and {high} Cr"})

    except Exception as e:
        app.logger.error(f"Error in prediction: {e}")
        return jsonify({"error": "An internal error occurred. Please try again."}), 500



@app.route('/filter-apartments', methods=['POST'])
def filter_apartments():
    """Filters apartments based on location & radius without reloading the page."""
    data = request.json  # Get JSON data from frontend
    selected_location = data.get('location')
    radius = float(data.get('radius'))

    try:
        # **Filter apartments within the given radius**
        filtered_df = location_df[location_df[selected_location] < radius * 1000][[selected_location]]
        filtered_df = filtered_df.sort_values(by=selected_location)

        if filtered_df.empty:
            return jsonify({'status': 'error', 'message': "No properties found within the given radius."})

        # Convert DataFrame to a list of dictionaries with name & distance
        apartments = [{"name": name, "distance": round(distance, 2)} for name, distance in filtered_df.itertuples()]

        return jsonify({'status': 'success', 'apartments': apartments})

    except KeyError:
        return jsonify({'status': 'error', 'message': "Invalid location selected."})
    except ValueError:
        return jsonify({'status': 'error', 'message': "Invalid input. Please enter a valid number for radius."})


# **********Going to recommendation.py with apartment, radius, location***************************88
@app.route('/recommend-results', methods=['POST'])
def get_recommendation_results():
    """Gets recommendations for a selected apartment without refreshing."""
    data = request.json
    selected_apartment = data.get('apartment')
    selected_location = data.get('location')
    radius = float(data.get('radius'))

    # **Filter apartments within the given radius again**
    filtered_df = location_df[location_df[selected_location] < radius * 1000][[selected_location]]
    filtered_df = filtered_df.sort_values(by=selected_location)

    if selected_apartment not in filtered_df.index:
        return jsonify({'status': 'error', 'message': "Invalid apartment selection. Please select from the list."})

    recommendation_df = recommend_properties_with_scores(selected_apartment, filtered_df)

    recommendations = recommendation_df.to_dict(orient='records')

    return jsonify({'status': 'success', 'recommendations': recommendations})




try:
    new_df = pd.read_csv('data/data_viz1.csv')
    with open("data/feature_text.pkl", "rb") as file:
        feature_text = pickle.load(file)
except Exception as e:
    print(f"Error loading data: {e}")
    new_df = pd.DataFrame()
    feature_text = ""

# Group Data for Visualization
group_df = new_df.groupby('sector', as_index=True)[['price', 'price_per_sqft', 'built_up_area', 'latitude', 'longitude']].mean()
# print(group_df.head())

# *********************analytics module*******************************************************


@app.route('/geoplot', methods=['GET'])
def geoplot():
    try:
        fig = px.scatter_mapbox(
            group_df, lat="latitude", lon="longitude", color="price_per_sqft", size="built_up_area",
            color_continuous_scale=px.colors.cyclical.IceFire, zoom=9, mapbox_style="open-street-map",
            hover_name=group_df.index
        )

        # Modify color bar title to be vertical
        fig.update_layout(
     coloraxis_colorbar=dict(
        title="Price per Sqft",  
        title_side="right",  # Keep the title vertical on the right
        thickness=5,  # Make colorbar thinner
        x=0.97,  # Shift color bar closer to the right edge (default is ~1.0)
        len=0.75  # Reduce height if needed
    ),
    margin=dict(l=10, r=10, t=50, b=50)
)

        graphJSON = fig.to_json()  # Convert plot to JSON
        return jsonify({'status': 'success', 'plot': graphJSON})  # Send properly formatted JSON

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}) 

@app.route('/area_vs_price', methods=['GET'])
def area_vs_price():
    property_type = request.args.get('property_type', 'flat')
    try:
        df_filtered = new_df[new_df['property_type'] == property_type].copy()
        fig = px.scatter(df_filtered, x="built_up_area", y="price", color="bedRoom")
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            title_x=0.5  # Center the title
        )
        return jsonify({'status': 'success', 'plot': fig.to_json()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})



@app.route('/bhk_pie_chart', methods=['GET'])
def bhk_pie_chart():
    sector = request.args.get('sector', 'overall')
    try:
        df_filtered = new_df if sector == "overall" else new_df[new_df['sector'] == sector].copy()
        # print(df_filtered.head())
        fig = fig = px.pie(df_filtered, names='bedRoom', title='Total Bill Amount by Day')
        return jsonify({'status': 'success', 'plot': fig.to_json()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})




@app.route('/bhk_boxplot', methods=['GET'])
def bhk_boxplot():
    try:
        fig = px.box(
            new_df[new_df['bedRoom'] <= 4].copy(),
            x='bedRoom', y='price',
            title='BHK vs Price Distribution'
        )

        # Update layout for better fitting
        fig.update_layout(
            autosize=True,
            margin=dict(l=40, r=40, t=40, b=40),
            height=400,  # Set height (adjust as needed)
            title_x=0.5  # Center title
        )

        return jsonify({'status': 'success', 'plot': fig.to_json()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/wordcloud', methods=['GET'])
def wordcloud():
    try:
        wordcloud = WordCloud(width=800, height=800, background_color='black', min_font_size=10).generate(feature_text)
        img = BytesIO()
        plt.figure(figsize=(8, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(img, format='png')
        img.seek(0)
        return jsonify({'status': 'success', 'image': base64.b64encode(img.getvalue()).decode()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/property_type_distplot', methods=['GET'])
def property_type_distplot():
    try:
        img = BytesIO()
        plt.figure(figsize=(10, 4))
        sns.histplot(new_df[new_df['property_type'] == 'house']['price'], kde=True, label='house')
        sns.histplot(new_df[new_df['property_type'] == 'flat']['price'], kde=True, label='flat')
        plt.legend()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        return jsonify({'status': 'success', 'image': base64.b64encode(img.getvalue()).decode()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})






def get_filtered_data(sector):
    return df if sector == "overall" else new_df[new_df['sector'] == sector].copy()

@app.route('/heatmap', methods=['GET'])
def geospatial_price_heatmap():
    sector = request.args.get('sector', 'overall')
    try:
        

        # Create a folium map
        m = folium.Map(location=[group_df['latitude'].mean(), group_df['longitude'].mean()], zoom_start=12)
        
        # Prepare heatmap data
        heat_data = group_df[['latitude', 'longitude', 'price_per_sqft']].values.tolist()
        HeatMap(heat_data, radius=15, blur=10).add_to(m)

        # Render map to HTML
        map_html = m._repr_html_()
        return jsonify({'status': 'success', 'map': map_html})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})



# 3️⃣ **Box Plot of Price Per Sqft Across Sectors**
@app.route('/price_boxplot', methods=['GET'])
def price_boxplot():
    try:
        fig = px.box(
            new_df, x="sector", y="price_per_sqft", title="Price per Sqft Across Sectors", color="sector"
        )

        # Remove legend (right-side sector list)
        fig.update_layout(
            showlegend=False,  # 🔥 hides the legend
            autosize=True,
            height=500,
            xaxis={'automargin': True},
            margin=dict(l=40, r=40, t=60, b=100),
            title_x=0.5
        )

        return jsonify({'status': 'success', 'plot': fig.to_json()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/avgprice', methods=['GET'])
def avgprice():
    try:
        if new_df.empty or 'sector' not in new_df.columns or 'price_per_sqft' not in new_df.columns:
            return jsonify({'status': 'error', 'message': 'Data is missing or invalid for plotting.'})

        grouped_data = new_df.groupby('sector')['price_per_sqft'].mean().reset_index()
        if grouped_data.empty:
            return jsonify({'status': 'error', 'message': 'No data available for the selected sectors.'})

        fig = px.bar(
            grouped_data,
            x='sector', y='price_per_sqft', title='Average Price Per Sqft by Sector'
        )

        # Update layout for better visualization
        fig.update_layout(
            autosize=True,
            margin=dict(l=40, r=40, t=60, b=100),  # Adjust margins for sector labels
            title_x=0.5  # Center the title
        )

        return jsonify({'status': 'success', 'plot': fig.to_json()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})



@app.route('/importance', methods=['GET'])
def feature_importance():
    try:
        # Get feature names from the preprocessor
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

        # Get feature importances from the final model
        importances = pipeline.named_steps['regressor'].feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

        # Sort and select top 10
        top_features = importance_df.sort_values('importance', ascending=False).head(10)

        # Plot
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Feature Importances',
            labels={'importance': 'Importance', 'feature': 'Feature Name'}
        )

        fig.update_layout(
            autosize=True,
            margin=dict(l=40, r=40, t=60, b=40),
            title_x=0.5,
            yaxis=dict(showticklabels=False)  # 👈 Hide Y-axis labels
        )

        return jsonify({'status': 'success', 'plot': fig.to_json()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/furnshied', methods=['GET'])
def furnshied():
    try:
        # Map numeric codes to meaningful labels
        furnishing_map = {
            0: 'Unfurnished',
            1: 'Semi-Furnished',
            2: 'Fully-Furnished'
        }

        # Replace codes with labels
        new_df['furnishing_label'] = new_df['furnishing_type'].map(furnishing_map)

        # Group by new label
        avg_price = new_df.groupby('furnishing_label')['price'].mean().reset_index()

        # Plot
        fig = px.bar(
            avg_price, x='furnishing_label', y='price',
            title='Average Price by Furnishing Type',
            color='furnishing_label',
            labels={'furnishing_label': 'Furnishing Type', 'price': 'Average Price'}
        )

        fig.update_layout(
            autosize=True,
            # height=600,
            # width=900,
            margin=dict(l=60, r=40, t=80, b=120),
            title_x=0.5,
            xaxis=dict(
                tickangle=-90,
                title='Furnishing Type'
            ),
            yaxis=dict(
                title='Average Price',
                gridwidth=0.2
            )
        )

        return jsonify({'status': 'success', 'plot': fig.to_json()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
