import pickle
from flask import Flask, render_template, request
app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    clf = pickle.load(file)


@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == 'POST':
        mydict = request.form
        # print(mydict)
        cough = int(mydict['cough'])
        headache = int(mydict['headache'])
        age = int(mydict['age'])
        temperature = int(mydict['temperature'])
        bodypain = int(mydict['bodypain'])
        breath = int(mydict['breath'])

        # code for prediction
        input = [age, temperature, cough, breath, headache, bodypain]
        inf_prob = clf.predict_proba([input])[0][1]*100
        # print(inf_prob)
        return render_template('show.html', inf=round(inf_prob, 2))

    return render_template('index.html')
    # return 'Hello, World!' + str(inf_prob)


if __name__ == '__main__':
    app.run(debug=True)
