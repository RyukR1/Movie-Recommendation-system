import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

def run_flask_api():
    print("Starting Flask API server...")
    flask_process = subprocess.Popen(
        [sys.executable, "app/app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return flask_process

def run_streamlit_app():
    print("Starting Streamlit frontend...")
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return streamlit_process

def open_browsers():
    time.sleep(3)  # Wait a bit for servers to start
    
    # Open Flask web UI
    webbrowser.open("http://localhost:5000")
    
    # Open Streamlit UI
    webbrowser.open("http://localhost:8501")

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    os.makedirs("app/models", exist_ok=True)
    
    # Start Flask API in a separate process
    flask_process = run_flask_api()
    
    # Start Streamlit app in a separate process
    streamlit_process = run_streamlit_app()
    
    # Open the web browsers in a separate thread
    browser_thread = Thread(target=open_browsers)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        # Wait for user to interrupt with Ctrl+C
        print("\nMovie Recommendation System is running!")
        print("Flask API: http://localhost:5000")
        print("Streamlit UI: http://localhost:8501")
        print("\nPress Ctrl+C to stop all services...\n")
        
        # Keep the main thread alive to allow catching keyboard interrupts
        while True:
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nShutting down services...")
        
        # Terminate processes
        flask_process.terminate()
        streamlit_process.terminate()
        
        print("All services have been stopped. Goodbye!") 