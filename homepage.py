from flask import Flask, render_template, request, redirect, url_for
import subprocess
import os

app = Flask(__name__)

# --- Helper Function to Execute Scripts ---
def execute_script(script_name):
    """Executes a non-web Python script using subprocess."""
    script_path = os.path.join(os.getcwd(), f'{script_name}')
    
    # NOTE ON EXECUTION:
    # We use Popen (non-blocking) for GUI scripts like Tkinter (or camera streams) 
    # to prevent the web server from hanging while the camera window is open.
    if script_name in ['get_faces_from_camera_tkinter.py', 'attendance_taker.py']:
        try:
            # Popen starts the script and immediately returns
            # This is critical for camera/GUI applications
            subprocess.Popen(['python', script_path])
            return f"{script_name} started successfully in a new process.", 200
        except Exception as e:
            return f"Error launching {script_name}: {e}", 500
    
    # For simple file viewing (like app.py, if it generates a simple output)
    try:
        # For non-GUI tasks, run() (blocking) is safer if you want the output
        result = subprocess.run(['python', script_path], capture_output=True, text=True, check=True)
        return f"{script_name} executed. Output: {result.stdout}", 200
    except subprocess.CalledProcessError as e:
        return f"Error in {script_name}: {e.stderr}", 500
    except FileNotFoundError:
        return f"Script file not found: {script_name}", 500


@app.route('/')
def index():
    """Serves the main landing page (lan_page.html)."""
    return render_template('lan_page.html')


@app.route('/run_action', methods=['POST'])
def run_action():
    """Handles the form submission from the switches."""
    action = request.form.get('action')
    message = ""
    status_code = 200

    if action == ' Login to enroll users':
        # Mapped to: get_faces_from_camera_tkinter.py
        message, status_code = execute_script('/get_faces_from_camera_tkinter.py')
        # We redirect back to the home page or a simple confirmation page
        return render_template('confirmation.html', title="Login/Enrollment Started", message=message, status=status_code)

    elif action == 'take_attendance':
        # Mapped to: attendance_taker.py
        message, status_code = execute_script('/attendance_taker.py')
        return render_template('confirmation.html', title="Attendance Taker Started", message=message, status=status_code)

    elif action == 'view_attendance':
        # Mapped to: app.py (Attendance Sheet/Viewer)
        # Note: If app.py is a full web application, you should likely REDIRECT to it.
        # If it's a script that generates a report, use the execution below.
        
        # --- Option A: Redirect to the other Flask app/route (Recommended for web views) ---
        return redirect('/view_sheet') 
        
        # --- Option B: Execute script and return output (Use if app.py is just a command-line report generator) ---
        # message, status_code = execute_script('app.py')
        # return render_template('result.html', title="Attendance Sheet Output", message=message, status=status_code)

    return "Invalid action selected.", 400

# Placeholder route for the Attendance Sheet view (if app.py is a separate Flask app,
# you would integrate it here or use a reverse proxy like Nginx/Apache).
@app.route('/view_sheet')
def view_sheet():
    return "This is the Attendance Sheet Web View (Data should be displayed here)."

# A simple confirmation page for the camera actions
@app.route('/confirmation')
def confirmation():
    return render_template('confirmation.html', title="Action Completed", message="Action triggered successfully.")

if __name__ == '__main__':
    # Setting debug=True for development. Use port 5000.
    app.run(debug=True, port=5000)