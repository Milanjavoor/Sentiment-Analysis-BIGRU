import matplotlib.pyplot as plt
import matplotlib.image as mimg
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

maxilen=250
model=load_model("sentiment_bigru.keras")
print("model loaded succesfully")
with open("tokenizer.pkl","rb") as f:
  tokenizer=pickle.load(f)
with open("labelencoder.pkl","rb") as f:
  encoder=pickle.load(f)
print("Tokenizer and encoder are loaded and ready to roll")

def clean(text):
  text=text.lower()
  #removing html tags
  text=re.sub(r'<.*?','',text)
  #removing URLs
  text=re.sub(r'http\S+','',text)
  #removing special characters
  text=re.sub(r'[^a-zA-Z\s]','',text)
  #removing extra spaces
  text=re.sub(r'\s+','',text).strip()
  return text
review=input("Enter a review:/n")
review=clean(review)
review=tokenizer.texts_to_sequences(review)
padded=pad_sequences(review,maxlen=maxilen,padding="post",truncating="post")
prediction=model.predict(padded)
probability=prediction[0][0]
print("Prediction probability:",probability)

# Classification as positive or negative
if probability>=0.5:
  print("positive Sentiment")
  img=mimg.imread("/postive.png")
  plt.axis("off")
  plt.title("Thank you little human")
  plt.show()
else:
  print("Negative sentiment")
  img=mimg.imread("/negative.png")
  plt.axis("off")
  plt.title("We are sorry little guy")
  plt.show()
