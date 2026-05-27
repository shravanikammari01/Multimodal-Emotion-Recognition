import pickle

with open("text_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("text_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

text = input("Enter text: ")

text_vector = vectorizer.transform([text])

prediction = model.predict(text_vector)

print("\nPredicted Emotion:", prediction[0])