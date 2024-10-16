from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mean_motion')
def mean_motion():
    mean_motion_df = pd.read_csv('mean_motion_and_predicted_mean_motion.csv')
    return render_template('csv_view.html', title='Mean Motion and Predicted Mean Motion', 
                           table=mean_motion_df.to_html(classes='data', index=False))

@app.route('/predicted_positions')
def predicted_positions():
    predicted_positions_df = pd.read_csv('predicted_positions.csv')
    return render_template('positions_view.html', title='Predicted Positions',
                           table=predicted_positions_df.to_html(classes='data', index=False))

@app.route('/potential_collisions')
def potential_collisions():
    potential_collisions_df = pd.read_csv('potential_collisions.csv')
    return render_template('collisions_view.html', title='Potential Collisions',
                           table=potential_collisions_df.to_html(classes='data', index=False),
                           collisions=potential_collisions_df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
