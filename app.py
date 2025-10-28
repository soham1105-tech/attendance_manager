from flask import Flask, render_template, request
from datetime import datetime
import sqlite3
import os

# Ensure the attendance.db file is set up if it doesn't exist
# Note: This is usually done in the Face Recognizer script, but for app.py safety:
def setup_database():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    # Create the attendance table if it doesn't exist
    create_table_sql = "CREATE TABLE IF NOT EXISTS attendance (name TEXT, time TEXT, date DATE, UNIQUE(name, date))"
    cursor.execute(create_table_sql)
    conn.commit()
    conn.close()

# Run database setup once when the application starts
setup_database()


app = Flask(__name__)

@app.route('/')
def index():
    # Pass an empty list for attendance_data on the initial load to ensure the HTML loop doesn't break
    return render_template('index.html', selected_date='', no_data=False, attendance_data=[])

@app.route('/attendance', methods=['POST'])
def attendance():
    selected_date = request.form.get('selected_date')
    
    # Safely parse the date, assuming the format from the HTML date picker is correct
    try:
        selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
        formatted_date = selected_date_obj.strftime('%Y-%m-%d')
    except ValueError:
        # Handle case where date might be missing or malformed, although required=True should prevent this
        return render_template('index.html', selected_date='', no_data=True, attendance_data=[], 
                               error_message="Invalid date selected.")

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    # --- FIX: SELECT name, time, AND date to provide the 3 values expected by the HTML template ---
    cursor.execute("SELECT name, time, date FROM attendance WHERE date = ?", (formatted_date,))
    attendance_data = cursor.fetchall()

    conn.close()

    if not attendance_data:
        # Pass an empty list when no data is found
        return render_template('index.html', selected_date=selected_date, no_data=True, attendance_data=[])
    
    # Pass the full list of (name, time, date) tuples to the template
    return render_template('index.html', selected_date=selected_date, attendance_data=attendance_data, no_data=False)

if __name__ == '__main__':
    # Running in debug mode allows Flask to reload on code changes
    app.run(debug=True)