import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences





def remove_special_charctor(x):
    x = x.replace('.', ' ')
    x = x.replace(',', ' ')
    x = x.replace('!', ' ')
    x = x.replace('@', ' ')
    x = x.replace('#', ' ')
    x = x.replace('$', ' ')
    x = x.replace('%', ' ')
    x = x.replace('^', ' ')
    x = x.replace('&', ' ')
    x = x.replace(')', ' ')
    x = x.replace('(', ' ')
    x = x.replace('-', ' ')
    x = x.replace('_', ' ')
    x = x.replace('=', ' ')
    x = x.replace('+', ' ')
    x = x.replace('/', ' ')
    x = x.replace('?', ' ')
    x = x.replace(';', ' ')
    x = x.replace('~', ' ')
    return x


def clean(text:str):
    text = text.lower() # lowercasing
    text = remove_special_charctor(text)
    return text



with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


def token_txt(text):
  token = tokenizer
  x = text
  token.fit_on_texts(x)
  sequences = token.texts_to_sequences(x)
  x = pad_sequences(sequences ,maxlen=37)
  return x