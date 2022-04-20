import pickle


def warm_start(sim):
    with open('filename.pickle', 'rb') as handle:
        data = pickle.load(handle)

    for k, v in data.items():
        sim[k] = v
