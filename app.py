from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mean_motion')
def mean_motion():
    mean_motion_df = pd.read_csv('mean_motion.csv')
    predicted_mean_motion_df = pd.read_csv('predicted_mean_motion.csv')
    accuracy = calculate_accuracy(mean_motion_df['Mean Motion'], predicted_mean_motion_df['Predicted Mean Motion'])
    return render_template('mean_motion.html', 
                           mean_motion=mean_motion_df['Mean Motion'].values[0],
                           predicted_mean_motion=predicted_mean_motion_df['Predicted Mean Motion'].values[0],
                           accuracy=accuracy)

@app.route('/positions')
def positions():
    position_df = pd.read_csv('predicted_positions.csv')
    return render_template('position.html', position_data=position_df.to_dict(orient='records'))

@app.route('/collisions', methods=['GET', 'POST'])
def collisions():
    collision_df = pd.read_csv('collision_detection.csv')
    collision_count = len(collision_df)

    if request.method == 'POST':
        # Get selected objects from the request
        selected_objects = request.json.get('selected_objects')

        if selected_objects:  # Check if selected_objects is not None
            filtered_collisions = collision_df[collision_df['OBJECT_NAME'].isin(selected_objects)]
            
            # Assuming distance can be calculated somehow from your data
            distances = calculate_distances(filtered_collisions)
            return jsonify(distances)  # Return distances as JSON

        return jsonify({"error": "No selected objects provided"}), 400

    return render_template('collision.html', 
                           collisions=collision_df.to_dict(orient='records'), 
                           collision_count=collision_count)

def calculate_distances(collisions_df):
    # Example logic to calculate distances (replace with actual logic)
    distances = {}
    for index, row in collisions_df.iterrows():
        object_name = row['OBJECT_NAME']
        # Replace the following logic with your distance calculation method
        distances[object_name] = np.random.uniform(1, 100)  # Example random distance
    return distances

def calculate_accuracy(mean_motion, predicted_mean_motion):
    return 100 - (abs(mean_motion - predicted_mean_motion) / mean_motion * 100)

if __name__ == '__main__':
    app.run(debug=True)
