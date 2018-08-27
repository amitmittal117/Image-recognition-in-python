# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# mnist = tf.keras.datasets.mnist


# new_model = tf.keras.models.load_model('epic_num_reader.model')

# print(new_model)

# # predictions = new_model.predict([x_test])

# # print(predictions)
# # print(x_train[0])


# # print(np.argmax(predictions[0]))

# # plt.imshow(x_test[1])
# # plt.show()


import pickle


pickle_in = open("X.pickle", "rb")

X = pickle.load(pickle_in)

print(X[1])