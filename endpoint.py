from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from pickle import load
from keras.applications.xception import Xception, preprocess_input
from keras.utils import img_to_array
from keras.utils import pad_sequences
from keras.models import load_model
import re


app = Flask(__name__)
encoder = tf.saved_model.load('encoder_model')
decoder = tf.saved_model.load('decoder_model')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = load(handle)
image_model = Xception(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
max_length = 30
language = 'en'

# Define the inference function
def generate_caption(image, encoder, decoder, tokenizer, max_length):
    # Resize and normalize the image
    img = img_to_array(image)
    img = tf.image.resize(img, (299, 299))
    img = preprocess_input(img)
    temp_input = tf.expand_dims(img, 0)
    img_ex = image_features_extract_model(temp_input)
    img_ex = tf.reshape(img_ex, (img_ex.shape[0], -1, img_ex.shape[3]))
    # Pass the image through the encoder
    encoded_features = encoder(img_ex)

    # Initialize the decoder's hidden state
    hidden = tf.zeros((1, 512))

    # Initialize the decoder input with the start token
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

    # Initialize the output sequence
    result = []

    # Run the decoder for each time step until it generates the end token or reaches the maximum sequence length
    for i in range(max_length):
        # Predict the next word and update the hidden state
        predictions, hidden, _ = decoder(dec_input, encoded_features, hidden)

        # Find the word with the highest predicted probability
        predicted_id = tf.argmax(predictions[0]).numpy()

        # Convert the predicted word ID to its corresponding word
        predicted_word = tokenizer.index_word[predicted_id]

        # Append the predicted word to the output sequence
        result.append(predicted_word)

        # If the end token is generated, return the output sequence
        if predicted_word == '<end>':
            return ' '.join(result)

        # Update the decoder input with the predicted word
        dec_input = tf.expand_dims([predicted_id], 0)

    # If the maximum sequence length is reached, return the output sequence
    return ' '.join(result)

@app.route('/image', methods=['POST'])
def image():
    # Check if a file was uploaded in the request
    if 'image' not in request.files:
        return 'No file uploaded', 400
    
    # Retrieve the file data from the request
    file = request.files['image']
    
    # Verify that the file is a JPEG image
    if file.mimetype != 'image/jpeg':
        return 'File must be a JPEG image', 400
    
    # Process the image data using Pillow (replace this with your image processing code)
    img = Image.open(io.BytesIO(file.read()))
    processed_data = generate_caption(img, encoder, decoder, tokenizer, max_length)
    processed_data = processed_data.replace("<end>", "")
    print(processed_data)
    
    # Create a JSON response containing the processed data
    response_data = {'text': processed_data}
    
    # Return the JSON response
    return jsonify(response_data)

if __name__ == '__main__':
    app.run()
