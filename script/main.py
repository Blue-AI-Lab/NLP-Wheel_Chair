import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


token = pickle.load(open('tokenizer.pkl','rb'))

def preprocessing(text):
  x = text
  token.fit_on_texts(x)
  sequences = token.texts_to_sequences(x)
  x = pad_sequences(sequences ,maxlen=12)
  return x


label = {
1:'on',
2:'off',
3:'forward',
4:'back',
5:'left',
6:'right',
7:'stop',
8:'speed_up',
9:'slow_down',
10:'slight_left',
11:'slight_right',
12:'turn_on_lights',
13:'turn_off_lights',
14:'u-turn',
15:'don’t_forward',
16:'don’t_back',
17:'don’t_left',
18:'don’t_right'
}

# print(token.word_index)

word = preprocessing("approach the coe block")

print(word)