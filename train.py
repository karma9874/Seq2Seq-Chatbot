import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import math
from tqdm import tqdm
import pickle
import os
from cornell_Data_Utils import preparing_data
from model import seq2seq_model,pad_sentence,get_accuracy,sentence_to_seq
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)

BATCH_SIZE = config.BATCH_SIZE
RNN_SIZE = config.RNN_SIZE
EMBED_SIZE = config.EMBED_SIZE
LEARNING_RATE = config.LEARNING_RATE
KEEP_PROB = config.KEEP_PROB
EPOCHS = config.EPOCHS
MODEL_DIR = config.MODEL_DIR
SAVE_PATH = config.SAVE_PATH

movie_line = 'Datasets/cornell movie-dialogs corpus/movie_lines.txt'
movie_convo = 'Datasets/cornell movie-dialogs corpus/movie_conversations.txt'

max_conversation_length = 5
min_consersation_length = 2
min_frequency_words = 3

questions_int,answers_int,vocabs_to_index,index_to_vocabs,question_vocab_size,answer_vocab_size = preparing_data(movie_line,
	movie_convo,max_conversation_length,
	min_consersation_length,min_frequency_words)

vocab_size = len(index_to_vocabs)

if os.path.exists("Web_Chat/vocab2index.p") and os.path.exists("Web_Chat/index2vocab.p"):
    print("vocab2index and index2vocab file is present")
else:
    pickle.dump(vocabs_to_index, open("Web_Chat/vocab2index.p", "wb"))
    pickle.dump(index_to_vocabs, open("Web_Chat/index2vocab.p", "wb"))


train_data = questions_int[BATCH_SIZE:]
test_data = answers_int[BATCH_SIZE:]
val_train_data = questions_int[:BATCH_SIZE]
val_test_data = answers_int[:BATCH_SIZE]

pad_int = vocabs_to_index['<PAD>']

val_batch_x,val_batch_len = pad_sentence(val_train_data,pad_int)
val_batch_y,val_batch_len_y = pad_sentence(val_test_data,pad_int)
val_batch_x = np.array(val_batch_x)
val_batch_y = np.array(val_batch_y)

no_of_batches = math.floor(len(train_data)//BATCH_SIZE)
round_no = no_of_batches*BATCH_SIZE

input_data,target_data,input_data_len,target_data_len,lr_rate,keep_probs,inference_logits,cost,train_op = seq2seq_model(question_vocab_size,
	EMBED_SIZE,RNN_SIZE,KEEP_PROB,answer_vocab_size,
	BATCH_SIZE,vocabs_to_index)

translate_sentence = 'how are you'
translate_sentence = sentence_to_seq(translate_sentence, vocabs_to_index)

acc_plt = []
loss_plt = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCHS):
        total_accuracy = 0.0
        total_loss = 0.0
        for bs in tqdm(range(0,round_no  ,BATCH_SIZE)):
          index = min(bs+BATCH_SIZE, round_no )
          
          batch_x,len_x = pad_sentence(train_data[bs:index],pad_int)
          batch_y,len_y = pad_sentence(test_data[bs:index],pad_int)
          batch_x = np.array(batch_x)
          batch_y = np.array(batch_y)
          pred,loss_f,opt = sess.run([inference_logits,cost,train_op], 
                                      feed_dict={input_data:batch_x,
                                                target_data:batch_y,
                                                input_data_len:len_x,
                                                target_data_len:len_y,
                                                lr_rate:LEARNING_RATE,
                                                keep_probs:KEEP_PROB})

          train_acc = get_accuracy(batch_y, pred)
          total_loss += loss_f 
          total_accuracy+=train_acc
    
        total_accuracy /= (round_no // BATCH_SIZE)
        total_loss /=  (round_no//BATCH_SIZE)
        acc_plt.append(total_accuracy)
        loss_plt.append(total_loss)
        translate_logits = sess.run(inference_logits, {input_data: [translate_sentence]*BATCH_SIZE,
                                         input_data_len: [len(translate_sentence)]*BATCH_SIZE,
                                         target_data_len: [len(translate_sentence)]*BATCH_SIZE,              
                                         keep_probs: KEEP_PROB,
                                         })[0]

        print('Epoch %d,Average_loss %f, Average Accucracy %f'%(EPOCHS+1,total_loss,total_accuracy))
        print('  Inputs Words: {}'.format([index_to_vocabs[i] for i in translate_sentence]))
        print('  Replied Words: {}'.format(" ".join([index_to_vocabs[i] for i in translate_logits])))
        print('\n')
        saver = tf.train.Saver() 
        saver.save(sess,MODEL_DIR+"/"+SAVE_PATH)
    
plt.plot(range(epochs),acc_plt)
plt.title("Change in Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


plt.plot(range(epochs),loss_plt)
plt.title("Change in loss")
plt.xlabel('Epoch')
plt.ylabel('Lost')
plt.show()


    