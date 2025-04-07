import pickle
import pandas as pd
import numpy as np

# Load pre-saved similarity matrices and location data
with open("PickleFile/location_distance.pkl", "rb") as file:
    location_df = pickle.load(file)
with open("PickleFile/cosine_sim1.pkl", "rb") as file:
    cosine_sim1 = pickle.load(file)
with open("PickleFile/cosine_sim2.pkl", "rb") as file:
    cosine_sim2 = pickle.load(file)
with open("PickleFile/cosine_sim3.pkl", "rb") as file:
    cosine_sim3 = pickle.load(file)



def recommend_properties_with_scores(property_name, filtered_df, top_n=5):
    """Recommends properties based on similarity scores from the filtered list."""
    cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1 * cosine_sim3  # Adjusted weights

    try:
        # Get similarity scores for the selected property
        property_index = location_df.index.get_loc(property_name)
        sim_scores = list(enumerate(cosine_sim_matrix[property_index]))

        # Sort properties by similarity score (descending)
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top_n most similar properties
        top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
        top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]

        # Retrieve property names from indices
        top_properties = location_df.index[top_indices].tolist()

        # **Extract only properties within the selected radius**
        filtered_recommendations = [
            (prop, round(score, 2), round(filtered_df.loc[prop].values[0] / 1000, 2))  # Round similarity score and extract distance
            for prop, score in zip(top_properties, top_scores)
            if prop in filtered_df.index
        ]

        # **Create DataFrame with recommendations**
        recommendations_df = pd.DataFrame(filtered_recommendations, columns=['PropertyName', 'SimilarityScore', 'Distance (km)'])

        return recommendations_df

    except KeyError:
        print(f"Error: '{property_name}' not found in location data.")
        return None
