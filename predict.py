from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import re
import tensorflow as tf
import tensorflow_datasets as tfds
import random

# import classes
from tranformer_classes import MultiHeadAttentionLayer
from tranformer_classes import PositionalEncoding
from tranformer_classes import preprocess_sentence
from tranformer_classes import load_conversations
from tranformer_classes import create_padding_mask
from tranformer_classes import create_look_ahead_mask

#tf.keras.utils.set_random_seed(1234)
strategy = tf.distribute.get_strategy()

print(f"Tensorflow version {tf.__version__}")

# Maximum sentence length
MAX_LENGTH = 20

questions, answers = load_conversations()
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    answers + questions, target_vocab_size=2**13
)
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2

model_filename_boss = "model_boss.h5"
model_filename_dilbert = "model_dilbert.h5"

model_dilbert = tf.keras.models.load_model(
    model_filename_dilbert,
    custom_objects={
        "PositionalEncoding": PositionalEncoding,
        "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
    },
    compile=False,
)


def evaluate_dilbert(sentence):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0
    )

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model_dilbert(inputs=[sentence, output], training=False)

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


def predict_dilbert(sentence):
    prediction = evaluate_dilbert(sentence)
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


print("\n STARTS BOSS \n")
sentence_quest = "it's friday!"
#sentence_quest = questions[random.randint(0, len(questions))]
#sentence_quest = "are you trying to kill us? "
sentence_answ = ""
sentence_cumulative = sentence_quest
for _ in range(3):    
    print(f"Boss: {sentence_quest}")

    sentence_answ = predict_dilbert("{}".format(sentence_cumulative))
    sentence_cumulative = "{} {}".format(sentence_cumulative, sentence_answ)
    print(f"Dilbert: {sentence_answ}\n")
    
    sentence_quest = predict_boss("{}".format(sentence_cumulative))    
    sentence_cumulative = "{} {}".format(sentence_cumulative, sentence_quest)


print("\n STARTS DILBERT \n")
sentence_quest = "it's friday!"
#sentence_quest = answers[random.randint(0, len(answers))]
#sentence_quest = "are you trying to kill us? "
sentence_answ = ""
sentence_cumulative = sentence_quest
for _ in range(3):    
    print(f"Dilbert: {sentence_quest}")

    sentence_answ = predict_boss("{}".format(sentence_cumulative))
    sentence_cumulative = "{} {}".format(sentence_cumulative, sentence_answ)
    print(f"Boss: {sentence_answ}\n")

    sentence_quest = predict_dilbert("{}".format(sentence_cumulative))    
    sentence_cumulative = "{} {}".format(sentence_cumulative, sentence_quest)
