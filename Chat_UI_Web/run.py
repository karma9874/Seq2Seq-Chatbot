from flask import Flask, render_template, request
from flask import jsonify
from flask_wtf.csrf import CSRFProtect

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pickle
tf.logging.set_verbosity(tf.logging.ERROR)

sys.path.append('..')
import config
from graph_Utils import sentence_to_seq,make_pred


vocabs_to_index = pickle.load(open("vocab2index.p", "rb"))
index_to_vocabs = pickle.load(open("index2vocab.p", "rb"))
batch_size = config.BATCH_SIZE
model_dir = config.MODEL_DIR
save_path = config.SAVE_PATH

loaded_graph = tf.Graph()
sess = tf.InteractiveSession(graph=loaded_graph)
save_path = '../'+model_dir+'/'+save_path
loader = tf.train.import_meta_graph(save_path + '.meta')
loader.restore(sess, save_path)
input_data = loaded_graph.get_tensor_by_name('input:0')
logits = loaded_graph.get_tensor_by_name('predictions:0')
input_data_len = loaded_graph.get_tensor_by_name('input_len:0')
target_data_len = loaded_graph.get_tensor_by_name('target_len:0')
keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')




app = Flask(__name__,static_url_path="/static")
csrf = CSRFProtect(app)
csrf.init_app(app)

@csrf.exempt
@app.route('/test', methods=['POST'])
def reply():
	json1 = request.json
	text = json1['msg']
	mode_input = sentence_to_seq(text,vocabs_to_index)
	output = make_pred(sess,
		input_data,
		input_data_len,
		target_data_len,
		keep_prob,
		mode_input,
		batch_size,
		logits,
		index_to_vocabs)
	return jsonify(text=output)


@app.route("/")
def index():
    return render_template("index.html")

if (__name__ == "__main__"):
    app.run(port = 5000,debug=True)
