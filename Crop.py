# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import keras
# from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
# from keras.layers import Dense, Flatten
# from keras.models import Model
# from keras.callbacks import ModelCheckpoint, EarlyStopping
# from keras.models import load_model
#
# len(os.listdir
#     ("/Users/somanshusharma/Downloads/crop/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"))
#
# train_datagen = ImageDataGenerator(zoom_range=0.5, shear_range=0.3, horizontal_flip=True
#                                    , preprocessing_function=preprocess_input)
# val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
#
# train = train_datagen.flow_from_directory \
#     (directory="/Users/somanshusharma/Downloads/crop/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train",
#      target_size=(256, 256), batch_size=32)
# val = train_datagen.flow_from_directory(
#     directory="/Users/somanshusharma/Downloads/crop/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid",
#     target_size=(256, 256), batch_size=32)
#
# t_img, label = train.next()
#
#
# def plotImage(img_arr, label):
#     for im, l in zip(img_arr, label):
#         plt.figure(figsize=(5, 5))
#         # plt.imshow(im/255)
#         plt.show()
#
#
# plotImage(t_img[:3], label[:3])
#
# """#Building Our Model"""
#
# from keras.layers import Dense, Flatten
# from keras.models import Model
# from keras.applications.vgg19 import VGG19
# import keras
#
# base_model = VGG19(input_shape=(256, 256, 3), include_top=False)
#
# for layer in base_model.layers:
#     layer.trainable = False
#
# base_model.summary()
#
# X = Flatten()(base_model.output)
# X = Dense(units=38, activation='softmax')(X)
#
# # Creating our model
# model = Model(base_model.input, X)
#
# model.summary()
#
# model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
#
# """#Early Stopping and Model Check point"""
#
# from keras.callbacks import ModelCheckpoint, EarlyStopping
#
# # early stopping
# es = EarlyStopping(monitor=
#                    'val_accuracy', min_delta=0.01, patience=3, verbose=1)
#
# # model check point
# mc = ModelCheckpoint(filepath="/Users/somanshusharma/Downloads/crop/Best_Model.h5",
#                      monitor='val_accuracy',
#                      min_delta=0.01,
#                      patience=3,
#                      verbose=1,
#                      save_best_only=True)
# cb = [es, mc]
#
# his = model.fit_generator(train,
#                           steps_per_epoch=16,
#                           epochs=50,
#                           verbose=1,
#                           callbacks=cb,
#                           validation_data=val,
#                           validation_steps=16)
#
# h = his.history
# h.keys()
#
# plt.plot(h['accuracy'])
#
# plt.plot(h['val_accuracy'], c="red")
#
# plt.title("acc vs v-acc")
#
# plt.plot(h['loss'])
# plt.plot(h['val_loss'], c="red")
# plt.title("loss vs v-loss")
# plt.show()
#
# # load best model
# from keras.models import load_model
#
# model = load_model("/Users/somanshusharma/Downloads/crop/Best_Model.h5")
#
# model.summary()
#
# model = load_model("/Users/somanshusharma/Downloads/crop/Best_Model.h5")
#
# # Now, try evaluating the model without providing y_true_one_hot_encoded
# result = model.evaluate(val)
# acc = result[1]  # Assuming accuracy is at index 1, adjust if necessary based on your model's metrics
# print(f"The accuracy of your model is {acc * 100} %")
#
# ref = dict(zip(list(train.class_indices.values()), list(train.class_indices.keys())))
#
#
# def prediction(path):
#     img = load_img(path, target_size=(256, 256))
#     i = img_to_array(img)
#     im = preprocess_input(i)
#     img = np.expand_dims(im, axis=0)
#     pred = np.argmax(model.predict(img))
#     print(f" the image belongs to {ref[pred]}")
#
#
# path = "/Users/somanshusharma/Downloads/crop/archive/test/test/TomatoEarlyBlight1.JPG"
# prediction(path)
#
#

