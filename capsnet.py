'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the network definitions for the various capsule network architectures.
'''

from keras import layers, models
from keras import backend as K
K.set_image_data_format('channels_last')

from capsule_layers import ConvCapsuleLayer, DeconvCapsuleLayer, Mask, Length

def CapsNetR3(input_shape, n_class=2):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)

    # Reshape layer to be 1 capsule x [filters] atoms
    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                    routings=1, name='primarycaps')(conv1_reshaped)

    # Layer 2: Convolutional Capsule
    conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                    routings=3, name='conv_cap_2_1')(primary_caps)

    # Layer 2: Convolutional Capsule
    conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                    routings=3, name='conv_cap_2_2')(conv_cap_2_1)

    # Layer 3: Convolutional Capsule
    conv_cap_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=3, name='conv_cap_3_1')(conv_cap_2_2)

    # Layer 3: Convolutional Capsule
    conv_cap_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                    routings=3, name='conv_cap_3_2')(conv_cap_3_1)

    # Layer 4: Convolutional Capsule
    conv_cap_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=3, name='conv_cap_4_1')(conv_cap_3_2)

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_1_1')(conv_cap_4_1)

    # Skip connection
    up_1 = layers.Concatenate(axis=-2, name='up_1')([deconv_cap_1_1, conv_cap_3_1])

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=1,
                                      padding='same', routings=3, name='deconv_cap_1_2')(up_1)

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_2_1')(deconv_cap_1_2)

    # Skip connection
    up_2 = layers.Concatenate(axis=-2, name='up_2')([deconv_cap_2_1, conv_cap_2_1])

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1,
                                      padding='same', routings=3, name='deconv_cap_2_2')(up_2)

    # Layer 3 Up: Deconvolutional Capsule
    deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_3_1')(deconv_cap_2_2)

    # Skip connection
    up_3 = layers.Concatenate(axis=-2, name='up_3')([deconv_cap_3_1, conv1_reshaped])

    # Layer 4: Convolutional Capsule: 1x1
    seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                routings=3, name='seg_caps')(up_3)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(seg_caps)

    # Decoder network.
    _, H, W, C, A = seg_caps.get_shape()
    y = layers.Input(shape=input_shape[:-1]+(1,))
    masked_by_y = Mask()([seg_caps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(seg_caps)  # Mask using the capsule with maximal length. For prediction

    def shared_decoder(mask_layer):
        recon_remove_dim = layers.Reshape((H.value, W.value, A.value))(mask_layer)

        recon_1 = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_1')(recon_remove_dim)

        recon_2 = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_2')(recon_1)

        out_recon = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                  activation='sigmoid', name='out_recon')(recon_2)

        return out_recon

    # Models for training and evaluation (prediction)
    train_model = models.Model(inputs=[x, y], outputs=[out_seg, shared_decoder(masked_by_y)])
    eval_model = models.Model(inputs=x, outputs=[out_seg, shared_decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=((H.value, W.value, C.value, A.value)))
    noised_seg_caps = layers.Add()([seg_caps, noise])
    masked_noised_y = Mask()([noised_seg_caps, y])
    manipulate_model = models.Model(inputs=[x, y, noise], outputs=shared_decoder(masked_noised_y))

    return train_model, eval_model, manipulate_model


def CapsNetR1(input_shape, n_class=2):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)

    # Reshape layer to be 1 capsule x [filters] atoms
    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                    routings=1, name='primarycaps')(conv1_reshaped)

    # Layer 2: Convolutional Capsule
    conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                    routings=1, name='conv_cap_2_1')(primary_caps)

    # Layer 2: Convolutional Capsule
    conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                    routings=3, name='conv_cap_2_2')(conv_cap_2_1)

    # Layer 3: Convolutional Capsule
    conv_cap_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=1, name='conv_cap_3_1')(conv_cap_2_2)

    # Layer 3: Convolutional Capsule
    conv_cap_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                    routings=3, name='conv_cap_3_2')(conv_cap_3_1)

    # Layer 4: Convolutional Capsule
    conv_cap_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=1, name='conv_cap_4_1')(conv_cap_3_2)

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_1_1')(conv_cap_4_1)

    # Skip connection
    up_1 = layers.Concatenate(axis=-2, name='up_1')([deconv_cap_1_1, conv_cap_3_1])

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=1,
                                      padding='same', routings=1, name='deconv_cap_1_2')(up_1)

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_2_1')(deconv_cap_1_2)

    # Skip connection
    up_2 = layers.Concatenate(axis=-2, name='up_2')([deconv_cap_2_1, conv_cap_2_1])

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1,
                                      padding='same', routings=1, name='deconv_cap_2_2')(up_2)

    # Layer 3 Up: Deconvolutional Capsule
    deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_3_1')(deconv_cap_2_2)

    # Skip connection
    up_3 = layers.Concatenate(axis=-2, name='up_3')([deconv_cap_3_1, conv1_reshaped])

    # Layer 4: Convolutional Capsule: 1x1
    seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                routings=1, name='seg_caps')(up_3)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(seg_caps)

    # Decoder network.
    _, H, W, C, A = seg_caps.get_shape()
    y = layers.Input(shape=input_shape[:-1]+(1,))
    masked_by_y = Mask()([seg_caps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(seg_caps)  # Mask using the capsule with maximal length. For prediction

    def shared_decoder(mask_layer):
        recon_remove_dim = layers.Reshape((H.value, W.value, A.value))(mask_layer)

        recon_1 = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_1')(recon_remove_dim)

        recon_2 = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_2')(recon_1)

        out_recon = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                  activation='sigmoid', name='out_recon')(recon_2)

        return out_recon

    # Models for training and evaluation (prediction)
    train_model = models.Model(inputs=[x, y], outputs=[out_seg, shared_decoder(masked_by_y)])
    eval_model = models.Model(inputs=x, outputs=[out_seg, shared_decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=((H.value, W.value, C.value, A.value)))
    noised_seg_caps = layers.Add()([seg_caps, noise])
    masked_noised_y = Mask()([noised_seg_caps, y])
    manipulate_model = models.Model(inputs=[x, y, noise], outputs=shared_decoder(masked_noised_y))

    return train_model, eval_model, manipulate_model


def CapsNetBasic(input_shape, n_class=2):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)

    # Reshape layer to be 1 capsule x [filters] atoms
    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=1, name='primarycaps')(conv1_reshaped)

    # Layer 4: Convolutional Capsule: 1x1
    seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                routings=3, name='seg_caps')(primary_caps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(seg_caps)

    # Decoder network.
    _, H, W, C, A = seg_caps.get_shape()
    y = layers.Input(shape=input_shape[:-1]+(1,))
    masked_by_y = Mask()([seg_caps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(seg_caps)  # Mask using the capsule with maximal length. For prediction

    def shared_decoder(mask_layer):
        recon_remove_dim = layers.Reshape((H.value, W.value, A.value))(mask_layer)

        recon_1 = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_1')(recon_remove_dim)

        recon_2 = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_2')(recon_1)

        out_recon = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                  activation='sigmoid', name='out_recon')(recon_2)

        return out_recon

    # Models for training and evaluation (prediction)
    train_model = models.Model(inputs=[x, y], outputs=[out_seg, shared_decoder(masked_by_y)])
    eval_model = models.Model(inputs=x, outputs=[out_seg, shared_decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=((H.value, W.value, C.value, A.value)))
    noised_seg_caps = layers.Add()([seg_caps, noise])
    masked_noised_y = Mask()([noised_seg_caps, y])
    manipulate_model = models.Model(inputs=[x, y, noise], outputs=shared_decoder(masked_noised_y))

    return train_model, eval_model, manipulate_model

def CapsNetR3Custom(input_shape, n_class=2):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)
    #print("conv1", conv1.get_shape())

    # Layer 2: Just a conventional Conv2D layer
    conv2 = layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', name='conv2')(conv1)
    #print("conv2", conv2.get_shape())

    # Layer 3: Just a conventional Conv2D layer
    conv3 = layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same', activation='relu', name='conv3')(conv2)
    #print("conv3", conv3.get_shape())

    # Layer 3: Just a conventional Conv2D layer
    conv4 = layers.Conv2D(filters=256, kernel_size=5, strides=2, padding='same', activation='relu', name='conv4')(conv3)
    #print("++conv4", conv4.get_shape())

    # Reshape layer to be 1 capsule x [filters] atoms
    _, H, W, C = conv4.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 2, C.value//2))(conv4)
    #print("++conv1_reshaped", conv1_reshaped.get_shape())

    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=128, strides=1, padding='same',
                                    routings=1, name='conv_cap_0')(conv1_reshaped)
    #print("++conv_cap_0", primary_caps.get_shape())

    # Layer 2: Convolutional Capsule
    conv_cap_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=128, strides=2, padding='same',
                                    routings=3, name='conv_cap_1')(primary_caps)
    #print("+++conv_cap_1", conv_cap_1.get_shape())

    # Layer 2: Convolutional Capsule
    conv_cap_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=64, strides=1, padding='same',
                                    routings=3, name='conv_cap_2')(conv_cap_1)
    #print("+++conv_cap_2", conv_cap_2.get_shape())

    # Layer 3: Convolutional Capsule
    conv_cap_3 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=128, strides=2, padding='same',
                                    routings=3, name='conv_cap_3')(conv_cap_2)
    #print("++++conv_cap_3", conv_cap_3.get_shape())

    # Layer 3: Convolutional Capsule
    conv_cap_4 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=1, padding='same',
                                    routings=3, name='conv_cap_4')(conv_cap_3)
    #print("++++conv_cap_4", conv_cap_4.get_shape())

    # Layer 4: Convolutional Capsule
    digit_caps = DeconvCapsuleLayer(kernel_size=5, num_capsule=0, num_atoms=16, scaling=8, padding='same',
                                    routings=3, name='digit_caps')(conv_cap_4)
    #print("++++digit_caps", digit_caps.get_shape())

    deconv_cap_5 = ConvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=64, strides=8, padding='same', 
                                      routings=3, name='deconv_cap_1')(digit_caps)
    #print("++++deconv_cap_5", deconv_cap_5.get_shape())

    
    deconv_cap_6 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=128, strides=1,
                                      padding='same', routings=3, name='deconv_cap_2')(deconv_cap_5)
    #print("++++deconv_cap_6", deconv_cap_6.get_shape())
     
    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_7 = DeconvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=64,
                                      padding='same', routings=3, name='deconv_cap_3')(deconv_cap_6)
    #print("++++deconv_cap_7", deconv_cap_7.get_shape())
    
    deconv_cap_8 = ConvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=128, strides=1,
                                      padding='same', routings=3, name='deconv_cap_4')(deconv_cap_7)
    #print("++++deconv_cap_8", deconv_cap_8.get_shape())
    
    deconv_cap_9 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=128,
                                      padding='same', routings=3, name='deconv_cap_5')(deconv_cap_8)
    #print("++++deconv_cap_9", deconv_cap_9.get_shape())
    
    deconv_cap_10 = ConvCapsuleLayer(kernel_size=4, num_capsule=1, num_atoms=256, strides=1,
                                      padding='same', routings=3, name='deconv_cap_6')(deconv_cap_9)
    #print("++++deconv_cap_10", deconv_cap_10.get_shape())
    
    econv_cap_11 = DeconvCapsuleLayer(kernel_size=4, num_capsule=1, num_atoms=128,
                                      padding='same', routings=3, name='econv_cap_7')(deconv_cap_10)
    #print("++++econv_cap_11", econv_cap_11.get_shape())
    
    deconv_cap_12 = ConvCapsuleLayer(kernel_size=4, num_capsule=1, num_atoms=64, strides=1,
                                      padding='same', routings=3, name='deconv_cap_8')(econv_cap_11)
    #print("++++deconv_cap_12", deconv_cap_12.get_shape())
    
    deconv_cap_13 = ConvCapsuleLayer(kernel_size=4, num_capsule=1, num_atoms=32, strides=1,
                                      padding='same', routings=3, name='deconv_cap_9')(deconv_cap_12)
    #print("++++deconv_cap_13", deconv_cap_13.get_shape())
    
    deconv_cap_14 = ConvCapsuleLayer(kernel_size=4, num_capsule=1, num_atoms=16, strides=1,
                                      padding='same', routings=3, name='deconv_cap_10')(deconv_cap_13)
    #print("++++deconv_cap_14", deconv_cap_14.get_shape())
    
    out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(deconv_cap_14)
    
    #print("++++out_seg", out_seg.get_shape())
    
    from keras.layers import Conv2D

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(out_seg)
    
    train_model = models.Model(inputs=[x], outputs=[conv10])
    
    
    return train_model


    """# Skip connection
    up_1 = layers.Concatenate(axis=-2, name='up_1')([deconv_cap_1_1, conv_cap_3_1])
    print("up_1", up_1.get_shape())

   

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_2_1')(deconv_cap_1_2)
    print("deconv_cap_2_1", deconv_cap_2_1.get_shape())

    # Skip connection
    up_2 = layers.Concatenate(axis=-2, name='up_2')([deconv_cap_2_1, conv_cap_2_1])

    print("up_2", up_2.get_shape())

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1,
                                      padding='same', routings=3, name='deconv_cap_2_2')(up_2)
    print("deconv_cap_2_2", deconv_cap_2_2.get_shape())

    # Layer 3 Up: Deconvolutional Capsule
    deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_3_1')(deconv_cap_2_2)
    print("deconv_cap_3_1", deconv_cap_3_1.get_shape())

    # Skip connection
    print(deconv_cap_3_1.get_shape(), conv1_reshaped.get_shape())
    up_3 = layers.Concatenate(axis=-2, name='up_3')([deconv_cap_3_1, conv1_reshaped])
    print("up_3", up_3.get_shape())


    # Layer 4: Convolutional Capsule: 1x1
    seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                routings=3, name='seg_caps')(up_3)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(seg_caps)

    # Decoder network.
    _, H, W, C, A = seg_caps.get_shape()
    y = layers.Input(shape=input_shape[:-1]+(1,))
    masked_by_y = Mask()([seg_caps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(seg_caps)  # Mask using the capsule with maximal length. For prediction

    def shared_decoder(mask_layer):
        recon_remove_dim = layers.Reshape((H.value, W.value, A.value))(mask_layer)

        recon_1 = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_1')(recon_remove_dim)

        recon_2 = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_2')(recon_1)

        out_recon = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                  activation='sigmoid', name='out_recon')(recon_2)

        return out_recon

    # Models for training and evaluation (prediction)
    train_model = models.Model(inputs=[x, y], outputs=[out_seg, shared_decoder(masked_by_y)])
    eval_model = models.Model(inputs=x, outputs=[out_seg, shared_decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=((H.value, W.value, C.value, A.value)))
    noised_seg_caps = layers.Add()([seg_caps, noise])
    masked_noised_y = Mask()([noised_seg_caps, y])
    manipulate_model = models.Model(inputs=[x, y, noise], outputs=shared_decoder(masked_noised_y))

    return train_model, eval_model, manipulate_model"""

