from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import re
import tensorflow as tf
import tensorflow_datasets as tfds

# import classes
from tranformer_classes import MultiHeadAttentionLayer
from tranformer_classes import PositionalEncoding
from tranformer_classes import preprocess_sentence
from tranformer_classes import load_conversations

tf.keras.utils.set_random_seed(1234)
strategy = tf.distribute.get_strategy()

print(f"Tensorflow version {tf.__version__}")

# Maximum sentence length
MAX_LENGTH = 40

questions, answers = load_conversations()

corpus = answers + questions
with open('tokenizer.tf.subwords', 'r', encoding='utf-8') as f:
   for inx, line in enumerate(f):
       if inx > 1:
          sent = line.lower().strip()
          sent = sent.replace('\n', '')
          sent = re.sub(r"[^а-яА-Я?.!,_]+", " ", sent)
          sent = sent.strip()
          corpus.append(sent)
tokenizer = tfds.features.text.SubwordTextEncoder(vocab_list = corpus)

# Build tokenizer using tfds for both questions and answers
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    answers + questions, target_vocab_size=2**13
)

# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2
print(f"Tokenized sample question: {tokenizer.encode(questions[20])}")



def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]

#print(create_padding_mask(tf.constant([[1, 2, 0, 3, 0], [0, 0, 0, 4, 5]])))



def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


model_filename_boss = "model_boss.h5"
model_filename = "model_dilbert.h5"

model = tf.keras.models.load_model(
    model_filename,
    custom_objects={
        "PositionalEncoding": PositionalEncoding,
        "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
    },
    compile=False,
)


def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0
    )

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)
    #print(prediction)
    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size]
    )
    return predicted_sentence



model_boss = tf.keras.models.load_model(
    model_filename_boss,
    custom_objects={
        "PositionalEncoding": PositionalEncoding,
        "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
    },
    compile=False,
)


def evaluate_boss(sentence):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0
    )

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model_boss(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict_boss(sentence):
    prediction = evaluate_boss(sentence)
    #print(prediction)
    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size]
    )
    return predicted_sentence


#sentence = "Do you finish?"
sentence_quest = ""
sentence_answ = "business"
sentence_cumulative = sentence_answ
#sentence = predict(sentence)
#print(f"Boss: {sentence}\n")


#print(f"Dilbert: {sentence}")
#sentence = predict_boss(sentence)
#print(f"Boss: {sentence}\n")
#sentence = predict(sentence)
#print(f"Dilbert: {sentence}")
#sentence = predict_boss(sentence)

#sys.exit()

# feed the model with its previous output
#sentence = "What's your plan?"
for _ in range(3):
    print(f"Dilbert: {sentence_answ}")
    sentence_quest = predict_boss("{}".format(sentence_cumulative))    
    sentence_cumulative = "{} {}".format(sentence_cumulative, sentence_quest)
    
    print(f"Boss: {sentence_quest}\n")
    sentence_answ = predict("{}".format(sentence_quest))
    sentence_cumulative = "{} {}".format(sentence_cumulative, sentence_answ)




