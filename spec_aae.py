import tensorflow as tf
import tflearn
import math
import numpy as np

ncells=64

def add_ij(input_tensor, x_dim, y_dim, with_r=False):
    """
    input_tensor: (batch, x_dim, y_dim, c)
    """
    batch_size_tensor = tf.shape(input_tensor)[0]
    xx_ones = tf.ones([batch_size_tensor, x_dim],dtype=tf.int32)
    xx_ones = tf.expand_dims(xx_ones, -1)
    xx_range = tf.tile(tf.expand_dims(tf.range(y_dim), 0), [batch_size_tensor, 1])
    xx_range = tf.expand_dims(xx_range, 1)
    xx_channel = tf.matmul(xx_ones, xx_range)
    xx_channel = tf.expand_dims(xx_channel, -1)
    yy_ones = tf.ones([batch_size_tensor, y_dim],dtype=tf.int32)
    yy_ones = tf.expand_dims(yy_ones, 1)
    yy_range = tf.tile(tf.expand_dims(tf.range(x_dim), 0),[batch_size_tensor, 1])
    yy_range = tf.expand_dims(yy_range, -1)
    yy_channel = tf.matmul(yy_range, yy_ones)
    yy_channel = tf.expand_dims(yy_channel, -1)
    xx_channel = tf.cast(xx_channel, 'float32') / (x_dim - 1)
    yy_channel = tf.cast(yy_channel, 'float32') / (y_dim - 1)
    xx_channel = xx_channel*2 - 1
    yy_channel = yy_channel*2 - 1
    ret = tf.concat([input_tensor, xx_channel,yy_channel], axis=-1)
    if with_r:
        rr = tf.sqrt( tf.square(xx_channel-0.5) + tf.square(yy_channel-0.5))
        ret = tf.concat([ret, rr], axis=-1)
    return ret

# CNN as encoder
def CNN_encoder_cat(x, tsamples, nsamples, n_output, nlabels=5):
    with tf.variable_scope("CNN_encoder_cat"):
        x = tf.reshape(x, shape=[-1, tsamples, nsamples, 1])
        #x = add_ij(x, tsamples, nsamples)
        encoder = tflearn.conv_2d(x, 32, [3,3], activation='tanh')
        print(encoder.get_shape())
        #encoder = tflearn.batch_normalization(encoder)
        encoder = tflearn.max_pool_2d(encoder, 2)
        print(encoder.get_shape())
        encoder = tflearn.conv_2d(encoder, 32, [3,3], activation='tanh')
        print(encoder.get_shape())
        #encoder = tflearn.batch_normalization(encoder)
        encoder = tflearn.max_pool_2d(encoder, 2)
        print(encoder.get_shape())
        encoder = tflearn.conv_2d(encoder, 32, [3,3], activation='tanh')
        print(encoder.get_shape())
        #encoder = tflearn.batch_normalization(encoder)
        encoder = tflearn.max_pool_2d(encoder, 2)
        print(encoder.get_shape())
        cat = tflearn.fully_connected(encoder, nlabels, activation='softmax', name="catout")
        output = tflearn.fully_connected(encoder, n_output, activation='linear', name="zout")
    return output,cat
def upsample(x):
    shape = x.get_shape().as_list()

    r1 = tf.reshape(x, [shape[0], shape[1] * shape[2], 1, shape[3]])
    r1_l = tf.pad(r1, [[0, 0], [0, 0], [0, 1], [0, 0]])
    r1_r = tf.pad(r1, [[0, 0], [0, 0], [1, 0], [0, 0]])
    r2 = tf.add(r1_l, r1_r)
    r3 = tf.reshape(r2, [shape[0], shape[1], shape[2] * 2, shape[3]])
    r3_l = tf.pad(r3, [[0, 0], [0, 0], [0, shape[2] * 2], [0, 0]])
    r3_r = tf.pad(r3, [[0, 0], [0, 0], [shape[2] * 2, 0], [0, 0]])
    r4 = tf.add(r3_l, r3_r)
    r5 = tf.reshape(r4, [shape[0], shape[1] * 2, shape[2] * 2, shape[3]])
    return r5
# CNN as decoder
def CNN_decoder(z, tsamples, nsamples,reuse=False):
    with tf.variable_scope("CNN_decoder", reuse=reuse):
        x = tflearn.fully_connected(z, 38*32, activation='tanh')
        x = tf.reshape(x, shape=[-1, 1, 38, 32])
        decoder = tflearn.conv_2d(x, 32, [3,3], activation='tanh')
        print(decoder.get_shape())
        #decoder = tflearn.batch_normalization(decoder)
        decoder = tflearn.upsample_2d(decoder, [2,2])
        print(decoder.get_shape())
        decoder = tflearn.conv_2d(decoder, 64, [3,3], activation='tanh')
        print(decoder.get_shape())
        #decoder = tflearn.batch_normalization(decoder)
        #decoder= tflearn.upsample_2d(decoder, [3,2])
        decoder= tflearn.upsample_2d(decoder, [3,2])
        print(decoder.get_shape())
        decoder = tflearn.conv_2d(decoder, 64, [3,3], activation='tanh')
        print(decoder.get_shape())
        #decoder = tflearn.batch_normalization(decoder)
        decoder= tflearn.upsample_2d(decoder, [1,2])
        print(decoder.get_shape())
        y = tflearn.conv_2d(decoder, 1, [3,3], activation='tanh', name="sigout")
        print(y.get_shape())
        #y = tflearn.conv_2d(x, 1, [2,2], activation='relu', name="sigout0")
        y = tflearn.fully_connected(y,tsamples*nsamples, activation="linear", name="sigout")
        #y = tf.reshape(y,[-1,tsamples,nsamples])
        #x = tf.sigmoid(y)
        x = y
        x = tflearn.reshape(x ,[-1,tsamples,nsamples], name="reshaped")
    return x,y 

def CNN_decoder_test(z, tsamples, nsamples,reuse=False):
    with tf.variable_scope("CNN_decoder", reuse=reuse):
        x = tflearn.fully_connected(z, nsamples/8*32*nsamples/8, activation='tanh')
        x = tf.reshape(x, shape=[-1, nsamples/8, nsamples/8, 32])
        decoder = tflearn.conv_2d(x, 32, [3,3], activation='tanh')
        print(decoder.get_shape())
        #decoder = tflearn.batch_normalization(decoder)
        decoder = tflearn.upsample_2d(decoder, [2,2])
        print(decoder.get_shape())
        decoder = tflearn.conv_2d(decoder, 32, [3,3], activation='tanh')
        print(decoder.get_shape())
        #decoder = tflearn.batch_normalization(decoder)
        decoder= tflearn.upsample_2d(decoder, [3,2])
        #decoder= tflearn.upsample_2d(decoder, [2,2])
        print(decoder.get_shape())
        decoder = tflearn.conv_2d(decoder, 32, [3,3], activation='tanh')
        print(decoder.get_shape())
        #decoder = tflearn.batch_normalization(decoder)
        decoder= tflearn.upsample_2d(decoder, [2,2])
        print(decoder.get_shape())
        y = tflearn.conv_2d(decoder, 1, [3,3], activation='linear', name="sigout")
        print(y.get_shape())
        #y = tflearn.conv_2d(x, 1, [2,2], activation='relu', name="sigout0")
        #y = tflearn.fully_connected(y,tsamples*nsamples, activation="linear", name="sigout")
        #y = tf.reshape(y,[-1,tsamples,nsamples])
        #x = tf.sigmoid(y)
        x = y
        x = tflearn.reshape(x ,[-1,tsamples,nsamples], name="reshaped")
    return x,y 

# LSTM as decoder
def LSTM_decoder(z, tsamples, nsamples,reuse=False):
    with tf.variable_scope("LSTM_decoder", reuse=reuse):
        x = tflearn.fully_connected(z, nsamples*tsamples, activation='linear')
        x = tflearn.activations.leaky_relu(x, alpha=0.01)
        x = tf.reshape(x, shape=[-1, tsamples, nsamples])
        y = tflearn.lstm(x, nsamples, return_seq=True, name="lstm1", activation="linear")
        #x = tf.reshape(x, shape=[-1, tsamples, nsamples,1])
        #x = tflearn.conv_2d(x, 128, [2,9], activation='linear')
        #x = tflearn.activations.leaky_relu(x, alpha=0.01)
        #x = tflearn.conv_2d(x, 1, [1,3], activation='sigmoid')
        #x = tf.sigmoid(y)
        x = y
        x = tflearn.reshape(x ,[-1,tsamples,nsamples], name="reshaped")
    return x,y 

# Discriminator
def discriminator(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("discriminator", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.relu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.relu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.matmul(h1, wo) + bo

    return tf.sigmoid(y), y

# Cat Discriminator
def discriminator_cat(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("discriminator_cat", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.relu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.relu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.matmul(h1, wo) + bo

    return tf.sigmoid(y), y

# zs Discriminator
def discriminator_zs(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("discriminator_zs", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.relu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.relu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.matmul(h1, wo) + bo

    return tf.sigmoid(y), y

# Linear transformation 
def linear_transform(x, vdim, reuse=False):
    with tf.variable_scope("linear_transform", reuse=reuse):
        out = tflearn.fully_connected(x, vdim, activation="linear", bias=False)
    return out


#Semisupervised aae for cat 
def adversarial_autoencoder_semsup_cat_nodimred(x_hat, x, x_id, z_sample, cat_sample, dim_img, dim_z, n_hidden, keep_prob, nlabels=4, vdim=2):
    tsamples = dim_img[0]
    nsamples = dim_img[1]
    ## Reconstruction Loss
    # encoding
    z,cat = CNN_encoder_cat(x_hat, tsamples, nsamples, dim_z, nlabels=cat_sample.get_shape()[1])
    #z,cat = LSTM_encoder_cat(x_hat, tsamples, nsamples, dim_z, nlabels=cat_sample.get_shape()[1])

    # decoding
    print(z.get_shape())
    print(cat.get_shape())
    decin = tf.concat([z,cat],axis=1, name="concat")
    y,y_ = CNN_decoder(decin, tsamples, nsamples)
    #y,y_ = LSTM_decoder(decin, tsamples, nsamples)
    # loss
    #marginal_likelihood = -tf.reduce_mean(tf.reduce_mean(tf.squared_difference(x,y)))
    marginal_likelihood = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x,y), [1,2]))
    #marginal_likelihood = -tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_,labels=x), [1,2]))

    ## Style GAN 
    #----------------------
    z_real = z_sample
    z_fake = z
    D_real, D_real_logits = discriminator(z_real, (int)(n_hidden), 1, keep_prob)
    D_fake, D_fake_logits = discriminator(z_fake, (int)(n_hidden), 1, keep_prob, reuse=True)

    # discriminator loss
    D_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
    D_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
    D_loss = D_loss_real+D_loss_fake

    # generator loss
    G_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

    ## Cat GAN Loss
    #----------------------
    cat_real = cat_sample
    cat_fake = cat 
    D_real_cat, D_real_logits_cat = discriminator_cat(cat_real, (int)(n_hidden), 1, keep_prob)
    D_fake_cat, D_fake_logits_cat = discriminator_cat(cat_fake, (int)(n_hidden), 1, keep_prob, reuse=True)
    
    # discriminator loss
    D_loss_real_cat = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits_cat, labels=tf.ones_like(D_real_logits_cat)))
    D_loss_fake_cat = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_cat, labels=tf.zeros_like(D_fake_logits_cat)))
    D_loss_cat = D_loss_real_cat+D_loss_fake_cat

    # generator loss
    G_loss_cat = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_cat, labels=tf.ones_like(D_fake_logits_cat)))
    
    #marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    D_loss = tf.reduce_mean(D_loss+D_loss_cat)
    G_loss = tf.reduce_mean(G_loss+G_loss_cat)

    cat_gen_loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cat, labels=x_id))
    return y, z, marginal_likelihood, D_loss, G_loss, cat_gen_loss, cat 


def decoder(z, dim_img, n_hidden):
    tsamples = dim_img[0]
    nsamples = dim_img[1]
    y = CNN_decoder(z, tsamples, nsamples, reuse=True)
    return y
