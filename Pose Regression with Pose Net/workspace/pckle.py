import pickle

print("Loading pretrained InceptionV1 weights...")
file = open('pretrained_models/places-googlenet.pickle', "rb")
weights = pickle.load(file, encoding="bytes")

#print("weights",weights)

for key, value in weights.items():
    print(key)
file.close()