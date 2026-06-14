import os
import time
import getpass
import keyring
import cv2
from datetime import datetime
import pytz
from apscheduler.schedulers.background import BackgroundScheduler

APP_NAME = "rtsp_capture_app"
CAMERA_IP = "192.168.1.154:554"
STREAM_PATH = "stream1"
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")

def get_credentials():
    username = keyring.get_password(APP_NAME, "username_placeholder")
    password = None
    
    if username:
        password = keyring.get_password(APP_NAME, username)
        
    if not username or not password:
        print(f"--- First-time Setup for {APP_NAME} ---")
        username = input("Enter RTSP Username: ").strip()
        password = getpass.getpass("Enter RTSP Password: ")
        
        # We store the username in a placeholder so we can retrieve it next time
        keyring.set_password(APP_NAME, "username_placeholder", username)
        keyring.set_password(APP_NAME, username, password)
        print("Credentials saved securely in Windows Credential Manager.")
    
    return username, password

def capture_frame():
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    
    username, password = get_credentials()
    rtsp_url = f"rtsp://{username}:{password}@{CAMERA_IP}/{STREAM_PATH}"
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Connecting to RTSP stream...")
    
    # Connect to the stream
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("ERROR: Could not open RTSP stream. Check connection or credentials.")
        return
        
    # Read a single frame
    ret, frame = cap.read()
    
    # Release the stream immediately
    cap.release()
    
    if not ret or frame is None:
        print("ERROR: Connected to stream but failed to grab a frame.")
        return
        
    # Format current time in US/Eastern (EDT/EST)
    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz)
    # Requested: mmddyyyy-hh:mm:ss, implemented as mmddyyyy-hh-mm-ss
    filename = now.strftime("%m%d%Y-%H-%M-%S.jpg")
    
    filepath = os.path.join(CAPTURE_DIR, filename)
    
    # Save the image
    cv2.imwrite(filepath, frame)
    print(f"Successfully saved screenshot: {filepath}")

if __name__ == "__main__":
    print("Initializing RTSP Capture App...")
    # Ensure credentials are set on startup
    get_credentials()
    
    scheduler = BackgroundScheduler()
    # Schedule to run at the top of every hour (minute=0)
    scheduler.add_job(capture_frame, 'cron', minute=0)
    scheduler.start()
    
    print("Scheduler started. The app will capture a screenshot every hour on the hour.")
    print("Keep this terminal window open to run in the background. Press Ctrl+C to exit.")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        print("\nShutting down...")
        scheduler.shutdown()
