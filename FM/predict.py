import pickle

def predict(model, x) :


model = pickle.load(open("fm.model", "rb"))

x_test =