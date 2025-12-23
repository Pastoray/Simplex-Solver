import webbrowser
from app import app
import threading

def open_browser():
    webbrowser.open("http://localhost:5000")

if __name__ == '__main__':
    print("Starting server...")
    threading.Timer(1.25, open_browser).start()
    app.run(host='127.0.0.1', port=5000, debug=False)
