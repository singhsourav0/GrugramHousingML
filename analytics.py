import pandas as pd
import pickle
import json
import plotly.express as px
import plotly.utils 
import seaborn as sns
from wordcloud import WordCloud
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
from flask import render_template

# Load Data with Exception Handling
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

# def generate_geoplot():
#     try:
#         fig = px.scatter_mapbox(
#             group_df, lat="latitude", lon="longitude", color="price_per_sqft", size="built_up_area",
#             color_continuous_scale=px.colors.cyclical.IceFire, zoom=10, mapbox_style="open-street-map",
#             width=1200, height=700, hover_name=group_df.index
#         )
#         return fig.to_json()  # Return JSON directly
#     except Exception as e:
#         return json.dumps({"status": "error", "message": str(e)})  # Return error as JSON



def generate_wordcloud():
    try:
        wordcloud = WordCloud(width=800, height=800, background_color='black', min_font_size=10).generate(feature_text)
        img = BytesIO()
        plt.figure(figsize=(8, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(img, format='png')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()
    except Exception as e:
        return json.dumps({"error": str(e)})

def generate_area_vs_price(property_type):
    try:
        df_filtered = new_df[new_df['property_type'] == property_type].copy()
        fig = px.scatter(df_filtered, x="built_up_area", y="price", color="bedRoom", title="Area Vs Price")
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        return json.dumps({"error": str(e)})

def generate_bhk_pie_chart(sector):
    try:
        df_filtered = new_df.copy() if sector == "overall" else new_df[new_df['sector'] == sector].copy()
        fig = px.pie(df_filtered, names='bedRoom', title=f"BHK Distribution in {sector}")
        return json.dumps(fig, cls=px.utils.PlotlyJSONEncoder)
    except Exception as e:
        return json.dumps({"error": str(e)})

def generate_bhk_boxplot():
    try:
        fig = px.box(new_df[new_df['bedRoom'] <= 4].copy(), x='bedRoom', y='price', title='BHK Price Range')
        return json.dumps(fig, cls=px.utils.PlotlyJSONEncoder)
    except Exception as e:
        return json.dumps({"error": str(e)})

def generate_property_type_distplot():
    img = BytesIO()
    plt.figure(figsize=(10, 4))
    sns.histplot(new_df[new_df['property_type'] == 'house']['price'], kde=True, label='house')
    sns.histplot(new_df[new_df['property_type'] == 'flat']['price'], kde=True, label='flat')
    plt.legend()
    plt.savefig(img, format='png')
    plt.close()  # ✅ Close figure after saving
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

