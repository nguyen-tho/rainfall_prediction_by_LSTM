from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import tensorflow.python.keras.backend as K
from tensorflow.python.keras import callbacks

def precision(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Cast y_true to float32
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Cast y_true to float32
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Cast y_true to float32
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))

def create_model(input_shape, y_encoded):
    model = Sequential([
    LSTM(128, input_shape=input_shape, return_sequences=True),
    Dropout(0.15),
    LSTM(64, return_sequences=True),
    Dropout(0.15),
    LSTM(32, return_sequences=False),
    Dense(32, activation='relu'),
    Dropout(0.15),
    Dense(y_encoded.shape[1], activation='softmax')  # Number of classes
    ])
    
    return model

def compile_model(model, loss_func, metrics, optimizer='adam'):
    model.compile(loss=loss_func, optimizer=optimizer, metrics=metrics)
    
def summary_model(model):
    model.summary()
    
def model_fit(model, training_data, epochs, batch_size, callback, validation_data):
    K.clear_session()
    history = model.fit(training_data[0], training_data[1], epochs=epochs, 
              batch_size=batch_size, validation_data=validation_data, callbacks=callback)
    return history

def save_model(model, save_path):
    model.save(save_path)
    
def loadModel(model_dir):
    model = load_model(model_dir)
    return model

def model_predict(model, X_test):
    return model.predict(X_test)

def callback(type='reduce_lr', logging=True, monitor='val_loss'):
    # create callback variable to store type of callback and logging if available
    callback = []
    # choose type of callback
    if type == 'reduce_lr':
        patience = 10
        min_lr = 1e-15
        factor = 0.5
        reduce_lr = callbacks.ReduceLROnPlateau(monitor=monitor, patience = patience, factor=factor, min_lr=min_lr)
        callback.append(reduce_lr)
    elif type == 'early_stopping':
        patience = 10
        restore_best_weight = True
        early_stopping = callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=restore_best_weight)
        callback.append(early_stopping)
    
    
    # choose logging available
    file_name = '../logging/training_log.csv' #logging file
    if logging:
        csv_logger = callbacks.CSVLogger(filename=file_name)
        callback.append(csv_logger)
    return callback
    