from flask import Flask,render_template
from flask.globals import request

import movies as maa


app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/pred",methods = ["POST"])
def sub():
    name = request.form["name"]
    pred = maa.get_movie_recommendation(name)

    return render_template("sub.html",data = pred)

if __name__ == "__main__":
    app.run(debug=True)