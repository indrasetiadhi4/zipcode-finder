"""
zipcode-finder
API for the zipcode-finder engine

@author: Indra Setiadhi
"""

# Importing libraries
import re
import csv
import joblib
from flask import Flask, request
from ftfy import fix_text

# Def API
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "<h1>Zipcode Finder</h1><p>This engine is a prototype API for zipcode finder.</p>"

@app.route('/findzipcode', methods=['POST'])
def findzipcode():
    # Get body json input
    body = request.get_json(force=True)
    address = body['address']
    
    # Predict zipcode of the inputed address
    distances, indices = getNearestN([address])
    
    # Response json
    response = dict(address_from_database = database[indices[0][0]+1][0],
                    predicted_zipcode = database[indices[0][0]+1][1],
                    distances = distances[0][0])
    return response

def ngrams(string, n=3):
    string = fix_text(string) # fix text encoding issues
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = re.sub('[^a-zA-Z]', ' ', string) # remove char other than alphabet
    string = string.lower() #make lower case
    string = re.sub(r'\bjl\b', ' ', string)
    string = re.sub(r'\bjln\b', ' ', string)
    string = re.sub(r'\bjalan\b', ' ', string)
    string = re.sub(r'\bblok\b', ' ', string)
    string = re.sub(r'\bno\b', ' ', string)
    string = re.sub(r'\brt\b', ' ', string)
    string = re.sub(r'\brw\b', ' ', string)
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single space
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def getNearestN(query):
  queryTFIDF_ = vectorizer.transform(query)
  distances, indices = nbrs.kneighbors(queryTFIDF_)
  return distances, indices

if __name__ == '__main__':
    vectorizer = joblib.load('model/vectorizer.pkl')
    nbrs = joblib.load('model/nbrs.pkl')
    with open('data/training_set.csv', newline='') as f:
        reader = csv.reader(f)
        database = list(reader)
    
    # Run API
    app.run(host='127.0.0.1', port='5000')