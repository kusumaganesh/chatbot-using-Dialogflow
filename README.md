# chatbot-using-Dialogflow
import pickle
import numpy as np
with open("train_qa.txt","rb") as fp:
 train_data = pickle.load(fp)
train_data
with open("test_qa.txt","rb") as fp:
 test_data = pickle.load(fp)
test_data
type(test_data)
type(train_data)
len(train_data)
len(test_data)
train_data[0]
' '.join(train_data[0][0])
' '.join(train_data[0][1])
27
train_data[0][2]
#Set up vocabulary
vocab = set()
all_data = test_data+train_data
type(all_data)
all_data
for story,question,answer in all_data:
 vocab = vocab.union(set(story))
 vocab = vocab.union(set(question))
vocab.add('yes')
vocab.add('no')
vocab
len(vocab)
vocab_len = len(vocab)+1
for data in all_data:
 print(len(data[0]))
28
 print("\n")
max_story_len = max([len(data[0]) for data in all_data])
max_story_len
max_ques_len = max([len(data[1]) for data in all_data])
max_ques_len
#Vectorsize
vocab
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(filters = [])
tokenizer.fit_on_texts(vocab)
tokenizer.word_index
train_story_text = []
train_question_text = []
train_answers = []
29
for story, question, answer in train_data:
 train_story_text.append(story)
 train_question_text.append(question)
train_story_seq = tokenizer.texts_to_sequences(train_story_text)
len(train_story_text)
len(train_story_seq)
train_story_seq
train_story_text
def vectorize_stories(data, word_index = tokenizer.word_index,
 max_story_len = max_story_len, max_ques_len = max_ques_len):
 X = [] #stories
 Xq = [] #query/question
 Y = [] #correct answer
 for story, query, answer in data:
 x = [word_index[word.lower()] for word in story]
 xq = [word_index[word.lower()] for word in query]
 y = np.zeros(len(word_index) +1)
 y[word_index[answer]] =1
30
 
 X.append(x)
 Xq.append(xq)
 Y.append(y)
 
 return(pad_sequences(X,maxlen = max_story_len),
 pad_sequences(Xq, maxlen = max_ques_len),
 np.array(Y))
 
 
inputs_train, queries_train, answers_train = vectorize_stories(train_data)
inputs_test, queries_test, answers_test = vectorize_stories(test_data)
inputs_train
queries_test
answers_test
tokenizer.word_index['yes']
tokenizer.word_index['no']
from keras.models import Sequential, Model
31
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, 
concatenate, LSTM
input_sequence = Input((max_story_len,))
question = Input((max_ques_len,))
#Input Encoder m
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim = vocab_len,output_dim = 64))
input_encoder_m.add(Dropout(0.3))
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim = vocab_len,output_dim = 
max_ques_len))
input_encoder_c.add(Dropout(0.3))
#Question Encoder
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim = vocab_len,output_dim = 
max_ques_len))
question_encoder.add(Dropout(0.3))
#Encode the sequences
input_encoded_m = input_encoder_m(input_sequence)
32
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)
input_encoded_m.shape
match = dot([input_encoded_m, question_encoded], axes = (2,2))
match = Activation('softmax')(match)
response = add([match,input_encoded_c ])
response = Permute((2,1))(response)
#Concetenate
answer = concatenate([response,question_encoded ])
answer
answer = LSTM(32)(answer)
answer = Dropout(0.5)(answer)
answer = Dense(vocab_len)(answer)
answer = Activation('softmax')(answer)
model = Model([input_sequence, question],answer )
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',metrics = 
['accuracy'])
33
model.summary()
#Evaluation on the test set
history = model.fit([inputs_train, queries_train], answers_train, 
 batch_size = 32, epochs = 20, 
 validation_data = ([inputs_test, queries_test], answers_test)
 )
import matplotlib.pyplot as plt
print(history.history.keys)
plt,plot(history.history['accuracy'])
plt,plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epochs")
#save
model.save("chatbt_model")
#Evaluation on the Test set
model.load_weights("chatbot_model")
pred_results = model.predict(([inputs_test, queries_test]))
34
test_data[0][0]
story = ' '.join(word for word in test_data[0][0])
story
query = ' '.join(word for word in test_data[0][0])
query
test_data[10][2]
val_max = np.argmax(pred_results[13])
for key, val in tokenizer.word_index.items():
 if val == val_max:
 k = key
 
print("Predicted Answer is",k)
print("Probability of certainity", pred_reults[13][val_max])
vocab
story = "Marry dropped the football . Sandra discard apple in kitchen"
story.split()
35
my_question = "Is apple in the kitchen ? "
my_question.split()
mydata = [(story.split(), my_question.split, 'yes')]
my_story, my_ques, my_ans = vectorize_stories(mydata)
pred_results = model.predict(([my_story, my_ques]))
val_max = np.argmax(pred_results[0])
for key, val in tokenizer.word_index.items():
 if val == val_max:
 k = key
 
print("Predicted Answer is",k)
print("Probability of certainity", pred_reults[0][val_max])
