from flask import Flask, render_template, request
import SimpleGrid

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    algorithm = request.form['algorithm']
    result = SimpleGrid.run_algorithm(algorithm)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

