from flask import Flask
app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>Hello from Pi!</h1><p>This page is served by Flask on your Raspberry Pi 5.</p>"

app.run(host="0.0.0.0", port=5000)

