def sentence_to_seq(sentence, vocabs_to_index):
    results = []
    for word in sentence.split(" "):
        if word in vocabs_to_index:
            results.append(vocabs_to_index[word])
        else:
            results.append(vocabs_to_index['<UNK>'])        
    return results

def print_data(i,batch_x,index_to_vocabs):
    data = []
    for n in batch_x:
        if n == 3373:
            break
        else:
            if n not in [3772,3373,3774,3775]:
                data.append(index_to_vocabs[n])
    return data

def make_pred(sess,input_data,input_data_len,target_data_len,keep_prob,sentence,batch_size,logits,index_to_vocabs):
    translate_logits = sess.run(logits, {input_data: [sentence]*batch_size,
                                         input_data_len: [len(sentence)]*batch_size,
                                         target_data_len : [len(sentence)]*batch_size,
                                         keep_prob: 1.0})[0]
    answer = print_data(0,translate_logits,index_to_vocabs)
    output = " ".join(answer)
    if not output:
        output = "Sorry, I dint understand your context"

    return output


    
     

