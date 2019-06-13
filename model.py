import tensorflow as tf
import numpy as np

def lstm(rnn_size, keep_prob,reuse=False):
    lstm_cell =tf.nn.rnn_cell.LSTMCell(rnn_size,reuse=reuse)
    drop =tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    return drop

def model_input():
    input_data = tf.placeholder(tf.int32, [None, None],name='input')
    target_data = tf.placeholder(tf.int32, [None, None],name='target')
    input_data_len = tf.placeholder(tf.int32,[None],name='input_len')
    target_data_len = tf.placeholder(tf.int32,[None],name='target_len')
    lr_rate = tf.placeholder(tf.float32,name='lr')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    return input_data,target_data,input_data_len,target_data_len,lr_rate,keep_prob

def encoder_input(source_vocab_size,embed_size,input_data):
    encoder_embeddings = tf.Variable(tf.random_uniform([source_vocab_size, embed_size], -1, 1))
    encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, input_data)
    return encoder_embedded


def encoder_layer(stacked_cells,encoder_embedded,input_data_len):
    ((encoder_fw_outputs,encoder_bw_outputs),
        (encoder_fw_final_state,encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked_cells, 
                                                                 cell_bw=stacked_cells, 
                                                                 inputs=encoder_embedded, 
                                                                 sequence_length=input_data_len, 
                                                                 dtype=tf.float32)
    encoder_outputs = tf.concat((encoder_fw_outputs,encoder_bw_outputs),2)
    encoder_state_c = tf.concat((encoder_fw_final_state.c,encoder_bw_final_state.c),1)
    encoder_state_h = tf.concat((encoder_fw_final_state.h,encoder_bw_final_state.h),1)
    encoder_states = tf.nn.rnn_cell.LSTMStateTuple(c=encoder_state_c,h=encoder_state_h)

    return encoder_outputs,encoder_states

def attention_layer(rnn_size,encoder_outputs,dec_cell,target_data_len,batch_size,encoder_states):
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size*2,encoder_outputs,
                                                                   memory_sequence_length=target_data_len)

    attention_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention_mechanism,
                                                             attention_layer_size=rnn_size/2)

    state = attention_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
    state = state.clone(cell_state=encoder_states)

    return attention_cell

def decoder_embedding(target_vocab_size,embed_size,decoder_input):
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, embed_size], -1, 1))
    dec_cell_inputs = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    return decoder_embeddings,dec_cell_inputs

def decoder_input(target_data,batch_size,vocabs_to_index):
    main = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1],vocabs_to_index['<GO>']), main], 1)
    return decoder_input

def decoder_train_layer(rnn_size,decoder_input,
    dec_cell_inputs,target_vocab_size,target_data_len,
    encoder_outputs,encoder_states,batch_size,attention_cell,state,dense_layer):
    train_helper = tf.contrib.seq2seq.TrainingHelper(dec_cell_inputs, target_data_len)
    
    decoder_train = tf.contrib.seq2seq.BasicDecoder(cell=attention_cell, helper=train_helper, 
                                                  initial_state=state,
                                                  output_layer=dense_layer)

    outputs_train, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_train, 
                                                  impute_finished=True, 
                                                  maximum_iterations=tf.reduce_max(target_data_len))

    return outputs_train

def decoder_infer_layer(decoder_embeddings,batch_size,vocabs_to_index,
    attention_cell,state,dense_layer,target_data_len):

    infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, 
                                                          tf.fill([batch_size], vocabs_to_index['<GO>']), 
                                                          vocabs_to_index['<EOS>'])

    decoder_infer = tf.contrib.seq2seq.BasicDecoder(cell=attention_cell, helper=infer_helper, 
                                                  initial_state=state,
                                                  output_layer=dense_layer)

    outputs_infer, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_infer, impute_finished=True,
                                                          maximum_iterations=tf.reduce_max(target_data_len))

    return outputs_infer

def opt_loss(outputs_train,outputs_infer,target_data_len,target_data,lr_rate):
    training_logits = tf.identity(outputs_train.rnn_output, name='logits')
    inference_logits = tf.identity(outputs_infer.sample_id, name='predictions')
    masks = tf.sequence_mask(target_data_len, tf.reduce_max(target_data_len), dtype=tf.float32, name='masks')
    cost = tf.contrib.seq2seq.sequence_loss(training_logits,target_data,masks)
    optimizer = tf.train.AdamOptimizer(lr_rate)
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

    return inference_logits,cost,train_op

def pad_sentence(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens

def get_accuracy(target, logits):
    max_seq = max(len(target[1]), logits.shape[1])
    if max_seq - len(target[1]):
        target = np.pad(
            target,
            [(0,0),(0,max_seq - len(target[1]))],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))

def sentence_to_seq(sentence, vocabs_to_index):
    results = []
    for word in sentence.split(" "):
        if word in vocabs_to_index:
            results.append(vocabs_to_index[word])
        else:
            results.append(vocabs_to_index['<UNK>'])        
    return results

def decoder_layer(rnn_size,encoder_outputs,target_data_len,
    dec_cell,encoder_states,target_data,vocabs_to_index,target_vocab_size,
    embed_size,dense_layer,attention_cell,state,batch_size):

    decoder_input_tensor = decoder_input(target_data,batch_size,vocabs_to_index)
    decoder_embeddings,dec_cell_inputs = decoder_embedding(target_vocab_size,embed_size,decoder_input_tensor)
    outputs_train = decoder_train_layer(rnn_size,decoder_input_tensor,dec_cell_inputs,target_vocab_size,target_data_len,encoder_outputs,encoder_states,batch_size,attention_cell,state,dense_layer)
    outputs_infer = decoder_infer_layer(decoder_embeddings,batch_size,vocabs_to_index,attention_cell,state,dense_layer,target_data_len)

    return outputs_train,outputs_infer

def seq2seq_model(source_vocab_size,embed_size,rnn_size,keep_prob,
    target_vocab_size,batch_size,vocabs_to_index):

    input_data,target_data,input_data_len,target_data_len,lr_rate,keep_probs = model_input()
    
    encoder_embedded = encoder_input(source_vocab_size,embed_size,input_data)
    
    stacked_cells = lstm(rnn_size, keep_prob)
    
    encoder_outputs,encoder_states = encoder_layer(stacked_cells,
            encoder_embedded,
            input_data_len)

    dec_cell = lstm(rnn_size*2,keep_prob)

    dense_layer = tf.layers.Dense(target_vocab_size)
    
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size*2,encoder_outputs,
                                                                   memory_sequence_length=target_data_len)
    attention_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention_mechanism,
                                                             attention_layer_size=rnn_size/2)
    state = attention_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
    state = state.clone(cell_state=encoder_states)

    # attention_cell = attention_layer(rnn_size,encoder_outputs,target_data_len,dec_cell,batch_size,encoder_states)
    
    outputs_train,outputs_infer = decoder_layer(rnn_size,encoder_outputs,target_data_len,
        dec_cell,encoder_states,target_data,vocabs_to_index,target_vocab_size,
        embed_size,dense_layer,attention_cell,state,batch_size)
    
    inference_logits,cost,train_op = opt_loss(outputs_train,outputs_infer,target_data_len,target_data,lr_rate)

    return input_data,target_data,input_data_len,target_data_len,lr_rate,keep_probs,inference_logits,cost,train_op







 





