# import h5py, pdb, math
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint

import numpy as np
import model 
import transform as T 
from utils import CustomDataset , unpickle 
import pdb
from keras import optimizers 


##############此处改成valid set 的目录

batch_size = 128
nb_classes = 17
nb_epoch = 200

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-5)
early_stopper = EarlyStopping(min_delta=0.001, patience=20)
csv_logger = CSVLogger('resnet50_base.csv')
checkpoint  = ModelCheckpoint('./check_file.h5', monitor='val_loss',
                              verbose=0, save_best_only=False,
                            save_weights_only=False, mode='auto', period=1)


#model = resnet.ResnetBuilder.build_resnet_18((18, 32, 32), nb_classes)
network = model.create_model('resnet50', input_shape=(18, 32, 32), 
                       num_outputs=nb_classes)

network.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.1, momentum=0.9, 
                                      nesterov=True),
              metrics=['accuracy'])

mean = unpickle("mean_channal.pkl")
trfs = T.Compose([
    T.Normalize(mean),
    T.RandomHorizontalFlip(0.5)
])
training_data = CustomDataset('/mnt/img1/yangqh/Germany_cloud/training.h5', transform = None)
validation_data = CustomDataset('/mnt/img1/yangqh/Germany_cloud/training.h5', transform = None)
##############此处改成train set 的目录
network.fit_generator(training_data.load_data(batch_size=batch_size),
                    steps_per_epoch = len(training_data) // batch_size,
                    validation_data=validation_data.load_data(batch_size=batch_size),
                    validation_steps = len(validation_data) // batch_size,
                    epochs=nb_epoch, verbose=1, max_q_size=100,
                    callbacks=[checkpoint,lr_reducer, early_stopper, csv_logger])
network.save('final.h5')
