"""
zipcode-finder
Create Model then save it to pickle file

@author: Indra Setiadhi
"""

# Import libraries
import re
import csv
from ftfy import fix_text
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Def analyzer
def ngrams(string, n=3):
    string = fix_text(string) # fix text encoding issues
    string = string.encode('ascii', errors='ignore').decode() #remove non ascii chars
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

# Import data
with open('data/training_set.csv', newline='') as f:
    reader = csv.reader(f)
    training_set = list(reader)
training_set_address = [row[0] for row in training_set[1:]]
    
# Training model
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
tfidf = vectorizer.fit_transform(training_set_address)
nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)

# Save model into pickle file
joblib.dump(vectorizer, 'model/vectorizer.pkl')
joblib.dump(nbrs, 'model/nbrs.pkl')
