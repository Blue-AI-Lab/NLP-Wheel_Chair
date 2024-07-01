import pickle
import numpy as np
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean,token_txt


with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)

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
18:'don’t_right',
19:'go to somewear'
}

# print(token.word_index)
text = "Approach the coe block"
word = clean(text)


token = token_txt([word])


output = model.predict(token)

print(output)