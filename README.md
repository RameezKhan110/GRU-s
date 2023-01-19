from keras.layers import GRU, Embedding, Dense, BatchNormalization, Dropout
from keras.models import Sequential

#variable
vocab_size = 10
embedding_dim = 10

# define the model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,
input_length=max_length))
model.add(BatchNormalization())
model.add(GRU(units=32, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(GRU(units=16, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(GRU(units=8))
model.add(Dense(units=1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
