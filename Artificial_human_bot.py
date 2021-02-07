# import the necessary libraries and packages
import random
import json
import speech_recognition as sr
import torch
from gtts import gTTS
import pyglet
import time,os
from Dataset_model import NeuralNetwork_Data
from nltk_utils import Array_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read the json file and process it
with open('Conversations.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNetwork_Data(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Artificial Human"
print("Let's chat! (type 'end' to exit) if you want to speak type 's' / 'S' or else no need")
print("I am Artificial Human")


def tts(text, lang):
    file = gTTS(text=text, lang=lang)
    filename = '/tmp/temp.mp3'
    file.save(filename)

    music = pyglet.media.load(filename, streaming=False)
    music.play()

    time.sleep(music.duration)
    os.remove(filename)


text = " Hii welcome to our cafe ask your doubts i will rectify it and if you want to speak with me just type s . Lets start to chat ! I am a Artificial Human"
lang = 'en'
tts(text, lang)
while True:
    sentence = input("You: ")
    if (sentence == "s" or sentence == "S"):
        r = sr.Recognizer()
        with sr.Microphone()as source:
            audio = r.listen(source)
        try:
            print("System Predicts: "+r.recognize_google(audio))
        except Exception:
            print("Something went wrong")
        sentence = r.recognize_google(audio)
    if sentence == "end":
        break

    sentence = tokenize(sentence)
    X = Array_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                s= random.choice(intent['responses'])
                print("Artificial Human: ",s)
                def tts(text, lang):
                    file = gTTS(text=text, lang=lang)
                    filename = '/tmp/temp.mp3'
                    file.save(filename)

                    music = pyglet.media.load(filename, streaming=False)
                    music.play()

                    time.sleep(music.duration)
                    os.remove(filename)

                text = s
                lang = 'en'
                tts(text,lang)
    else:
      print("Artificial Human: I do not understand......")


      def tts(text, lang):
          file = gTTS(text=text, lang=lang)
          filename = '/tmp/temp.mp3'
          file.save(filename)

          music = pyglet.media.load(filename, streaming=False)
          music.play()

          time.sleep(music.duration)
          os.remove(filename)
      text = "I do not understand"
      lang = 'en'
      tts(text, lang)
