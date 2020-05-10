import argparse
import glob
import os
import errno
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import tensorflow as tf

#import spec_data
#import esense_seqload
import hackrf_data
#import rawdata 
#import synthetic_data 
import prior_factory as prior
import spec_aae
import plot_utils

IMAGE_SIZE_MNIST = 28

def save_subimages(res,name):
    fig, ax = plt.subplots(nrows=len(res[0]) , ncols=len(res), sharex=True, sharey=True, figsize=(10, 10))
    #fig.text(0.45, 0.04, 'Continuous Features', ha='center')
    #fig.text(0.1, 0.45, 'Feature range [-1,1]', va='center', rotation='vertical')
    for i in range(len(res)):
        for j in range(len(res[0])):
            im = ax[j,i].imshow(res[i][j],interpolation='none', aspect='auto')
            ax[j,i].yaxis.set_major_formatter(FormatStrFormatter('%g'))
            #ax[j,i].yaxis.set_ticks(np.arange(0, 3, 1))
    fig.colorbar(im,  ax=ax.ravel().tolist(), orientation='vertical')
    fig.savefig(name)
    plt.close(fig)

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_path', type=str, default='results',
                        help='File path of output images')

    parser.add_argument('--prior_type', type=str, default='mixGaussian',
                        choices=['mixGaussian', 'swiss_roll', 'normal'],
                        help='The type of prior', required = True)

    parser.add_argument('--n_hidden', type=int, default=256, help='Number of hidden units in MLP')
    parser.add_argument('--dimz', type=int, default=20, help='Feature dimension')

    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    parser.add_argument('--PRR', type=bool, default=True,
                        help='Boolean for plot-reproduce-result')

    parser.add_argument('--PRR_n_img_x', type=int, default=5,
                        help='Number of images along x-axis')

    parser.add_argument('--PRR_n_img_y', type=int, default=5,
                        help='Number of images along y-axis')

    parser.add_argument('--PRR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR', type=bool, default=True,
                        help='Boolean for plot-manifold-learning-result')

    parser.add_argument('--PMLR_n_img_x', type=int, default=10,
                        help='Number of images along x-axis')

    parser.add_argument('--PMLR_n_img_y', type=int, default=10,
                        help='Number of images along y-axis')

    parser.add_argument('--PMLR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR_z_range', type=float, default=3.0,
                        help='Range for unifomly distributed latent vector')

    parser.add_argument('--PMLR_n_samples', type=int, default=10000,
                        help='Number of samples in order to get distribution of labeled data')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):

    # --results_path
    try:
        os.mkdir(args.results_path)
    except OSError as e:
        if (e ==errno.EEXIST):
            print('Removing files in the results directory')
    # delete all existing files
    files = glob.glob(args.results_path+'/*')
    for f in files:
        os.remove(f)

    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than one')

    # --learn_rate
    try:
        assert args.learn_rate > 0
    except:
        print('learning rate must be positive')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --PRR
    try:
        assert args.PRR == True or args.PRR == False
    except:
        print('PRR must be boolean type')
        return None

    if args.PRR == True:
        # --PRR_n_img_x, --PRR_n_img_y
        try:
            assert args.PRR_n_img_x >= 1 and args.PRR_n_img_y >= 1
        except:
            print('PRR : number of images along each axis must be larger than or equal to one')

        # --PRR_resize_factor
        try:
            assert args.PRR_resize_factor > 0
        except:
            print('PRR : resize factor for each displayed image must be positive')

    # --PMLR
    try:
        assert args.PMLR == True or args.PMLR == False
    except:
        print('PMLR must be boolean type')
        return None

    if args.PMLR == True:

        # --PMLR_n_img_x, --PMLR_n_img_y
        try:
            assert args.PMLR_n_img_x >= 1 and args.PMLR_n_img_y >= 1
        except:
            print('PMLR : number of images along each axis must be larger than or equal to one')

        # --PMLR_resize_factor
        try:
            assert args.PMLR_resize_factor > 0
        except:
            print('PMLR : resize factor for each displayed image must be positive')

        # --PMLR_z_range
        try:
            assert args.PMLR_z_range > 0
        except:
            print('PMLR : range for unifomly distributed latent vector must be positive')

        # --PMLR_n_samples
        try:
            assert args.PMLR_n_samples > 100
        except:
            print('PMLR : Number of samples in order to get distribution of labeled data must be large enough')

    return args

def shuffle_in_unison_inplace(a, b, c=[]): 
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    if len(c):
        assert len(c) == len(b)
        return a[p], b[p], c[p]
    else:
        return a[p], b[p]

"""main function"""
def main(args):

    np.random.seed(1337)
    """ parameters """
    RESULTS_DIR = args.results_path

    # network architecture
    n_hidden = args.n_hidden

    # train
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate

    # Plot
    PRR = args.PRR                              # Plot Reproduce Result
    PRR_n_img_x = args.PRR_n_img_x              # number of images along x-axis in a canvas
    PRR_n_img_y = args.PRR_n_img_y              # number of images along y-axis in a canvas
    PRR_resize_factor = args.PRR_resize_factor  # resize factor for each image in a canvas

    PMLR = args.PMLR                            # Plot Manifold Learning Result
    PMLR_n_img_x = args.PMLR_n_img_x            # number of images along x-axis in a canvas
    PMLR_n_img_y = args.PMLR_n_img_y            # number of images along y-axis in a canvas
    PMLR_resize_factor = args.PMLR_resize_factor# resize factor for each image in a canvas
    PMLR_z_range = args.PMLR_z_range            # range for random latent vector
    PMLR_n_samples = args.PMLR_n_samples        # number of labeled samples to plot a map from input data space to the latent space

    """ prepare MNIST data """

    '''

    esense_files = [
                    "AAU_livingLab4_202481591532165_1541682359",
                    "fabio_1-202481588431654_1541691060", 
                    "alemino_ZRH_202481601716927_1541691041",
                    "IMDEA_wideband_202481598624002_1541682492"
                    ]
                    b
    esense_folder = "./datadumps/esense_data_jan2019/"
    #train_data, train_labels, test_data, test_labels, bw_labels, pos_labels = spec_data.gendata()
    for ei,efile in enumerate(esense_files):
        print efile
        if ei==0:
            train_data, train_labels,_ = esense_seqload.gendata(esense_folder+efile)
        else:
            dtrain_data, dtrain_labels,_ = esense_seqload.gendata(esense_folder+efile)
            train_data = np.vstack((train_data,dtrain_data))
            train_labels = np.vstack((train_labels,dtrain_labels))
    '''
    #train_data, train_labels, _,_,_,_,_ = synthetic_data.gendata()
    train_data, train_labels,_,_,_ = hackrf_data.gendata("./datadumps/sample_hackrf_data.csv")
    #train_data, train_labels = rawdata.gendata()
    #Split the data
    train_data, train_labels = shuffle_in_unison_inplace(train_data, train_labels)
    splitval = int(train_data.shape[0] *0.5)
    test_data = train_data[:splitval] 
    test_labels = train_labels[:splitval] 
    train_data = train_data[splitval:]
    train_labels = train_labels[splitval:]
    #Semsup splitting
    splitval = int(train_data.shape[0] *0.2)
    train_data_sup = train_data[:splitval]
    train_data = train_data[splitval:]
    train_labels_sup = train_labels[:splitval]
    train_labels = train_labels[splitval:]
    n_samples = train_data.shape[0]
    tsamples = train_data.shape[1]
    fsamples = train_data.shape[2]
    dim_img = [tsamples,fsamples] 
    nlabels = train_labels.shape[1]
    print(nlabels)

    encoder="CNN"
    #encoder="LSTM"
    dim_z = args.dimz # to visualize learned manifold
    enable_sel =False
    """ build graph """

    # input placeholders
    x_hat = tf.placeholder(tf.float32, shape=[None, tsamples, fsamples], name='input_img')
    x = tf.placeholder(tf.float32, shape=[None, tsamples, fsamples], name='target_img')
    x_id = tf.placeholder(tf.float32, shape=[None, nlabels], name='input_img_label')

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # input for PMLR
    z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')
    

    # samples drawn from prior distribution
    z_sample = tf.placeholder(tf.float32, shape=[None, dim_z], name='prior_sample')
    cat_sample = tf.placeholder(tf.float32, shape=[None, nlabels], name='prior_sample_label')

    # network architecture
    #y, z, neg_marginal_likelihood, D_loss, G_loss = aae.adversarial_autoencoder(x_hat, x, x_id, z_sample, z_id, dim_img,
    #                                                                            dim_z, n_hidden, keep_prob)
    y, z, neg_marginal_likelihood, D_loss, G_loss, cat_gen_loss, cat = spec_aae.adversarial_autoencoder_semsup_cat_nodimred(x_hat, x, x_id, z_sample, cat_sample, dim_img,dim_z, n_hidden, keep_prob, nlabels=nlabels, vdim=2)

    # optimization
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if "discriminator" or "discriminator_cat" in var.name]
    g_vars = [var for var in t_vars if encoder+"_encoder_cat" in var.name]
    ae_vars = [var for var in t_vars if encoder+"_encoder_cat" or "CNN_decoder" in var.name]

    train_op_ae = tf.train.AdamOptimizer(learn_rate).minimize(neg_marginal_likelihood, var_list=ae_vars)
    train_op_d = tf.train.AdamOptimizer(learn_rate/2.0).minimize(D_loss, var_list=d_vars)
    train_op_g = tf.train.AdamOptimizer(learn_rate).minimize(G_loss, var_list=g_vars)
    train_op_cat = tf.train.AdamOptimizer(learn_rate).minimize(cat_gen_loss, var_list=g_vars)

    """ training """

    # Plot for reproduce performance
    if PRR:
        PRR = plot_utils.Plot_Reproduce_Performance(RESULTS_DIR, PRR_n_img_x, PRR_n_img_y, tsamples, fsamples, PRR_resize_factor)

        x_PRR = test_data[0:PRR.n_tot_imgs, :]

        x_PRR_img = x_PRR.reshape(PRR.n_tot_imgs, tsamples , fsamples)
        PRR.save_images(x_PRR_img, name='input.jpg')

    # Plot for manifold learning result
    if PMLR and dim_z == 2:

        PMLR = plot_utils.Plot_Manifold_Learning_Result(RESULTS_DIR, PMLR_n_img_x, PMLR_n_img_y, tsamples, fsamples, PMLR_resize_factor, PMLR_z_range)

        x_PMLR = test_data[0:PMLR_n_samples, :]
        id_PMLR = test_labels[0:PMLR_n_samples, :]

        decoded = spec_aae.decoder(z_in, dim_img, n_hidden)
    else:
        x_PMLR = test_data[0:PMLR_n_samples, :]
        id_PMLR = test_labels[0:PMLR_n_samples, :]
        z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')

    # train
    total_batch = int(n_samples / batch_size)
    min_tot_loss = 1e99
    prev_loss = 1e99

    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})

        for epoch in range(n_epochs):

            # Random shuffling
            train_data_, train_label_ = shuffle_in_unison_inplace(train_data, train_labels)
            train_data_sup_, train_labels_sup_ = shuffle_in_unison_inplace(train_data_sup, train_labels_sup)

            # Loop over all batches
            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (n_samples)
                offset_sup = (i * batch_size) % (train_data_sup.shape[0])
                batch_xs_input = train_data_[offset:(offset + batch_size), :]
                batch_ids_input = train_label_[offset:(offset + batch_size), :]
                batch_xs_sup_input = train_data_sup_[offset_sup:(offset_sup + batch_size), :]
                batch_ids_sup_input= train_labels_sup_[offset_sup:(offset_sup + batch_size), :]
                batch_xs_target = batch_xs_input
                batch_xs_sup_target = batch_xs_sup_input

                # draw samples from prior distribution
                if dim_z > 2:
                    if enable_sel:
                        if args.prior_type == 'mixGaussian':
                            z_id_ = np.random.randint(0, nlabels, size=[batch_size])
                            samples=np.zeros((batch_size, dim_z))
                            for el in range(dim_z/2):
                                samples_ = prior.gaussian_mixture(batch_size, 2 , n_labels=nlabels, label_indices=z_id_, y_var=(1.0/nlabels))
                                samples[:,el*2:(el+1)*2] = samples_ 
                        elif args.prior_type == 'swiss_roll':
                            z_id_ = np.random.randint(0, nlabels, size=[batch_size])
                            samples=np.zeros((batch_size, dim_z))
                            for el in range(dim_z/2):
                                samples_ = prior.swiss_roll(batch_size, 2, label_indices=z_id_)
                                samples[:,el*2:(el+1)*2] = samples_ 
                        elif args.prior_type == 'normal':
                            samples, z_id_ = prior.gaussian(batch_size, dim_z, n_labels=nlabels, use_label_info=True)
                        else:
                            raise Exception("[!] There is no option for " + args.prior_type)
                    else:
                        z_id_ = np.random.randint(0, nlabels, size=[batch_size])
                        samples = np.random.normal(0.0, 1, (batch_size, dim_z)).astype(np.float32) 
                else:
                    if args.prior_type == 'mixGaussian':
                        z_id_ = np.random.randint(0, nlabels, size=[batch_size])
                        samples = prior.gaussian_mixture(batch_size, dim_z, n_labels=nlabels, label_indices=z_id_, y_var=(1.0/nlabels))
                    elif args.prior_type == 'swiss_roll':
                        z_id_ = np.random.randint(0, nlabels, size=[batch_size])
                        samples = prior.swiss_roll(batch_size, dim_z, label_indices=z_id_)
                    elif args.prior_type == 'normal':
                        samples, z_id_ = prior.gaussian(batch_size, dim_z, n_labels=nlabels, use_label_info=True)
                    else:
                        raise Exception("[!] There is no option for " + args.prior_type)

                z_id_one_hot_vector = np.zeros((batch_size, nlabels))
                z_id_one_hot_vector[np.arange(batch_size), z_id_] = 1

                # reconstruction loss
                _, loss_likelihood0 = sess.run(
                    (train_op_ae, neg_marginal_likelihood),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, z_sample: samples,
                               cat_sample: z_id_one_hot_vector, keep_prob: 0.9})

                _, loss_likelihood1 = sess.run(
                    (train_op_ae, neg_marginal_likelihood),
                    feed_dict={x_hat: batch_xs_sup_input, x: batch_xs_sup_target, z_sample: samples,
                               cat_sample: batch_ids_sup_input, keep_prob: 0.9})
                loss_likelihood = loss_likelihood0 + loss_likelihood1
                # discriminator loss
                _, d_loss = sess.run(
                    (train_op_d, D_loss),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, z_sample: samples,
                               cat_sample: z_id_one_hot_vector, keep_prob: 0.9})

                # generator loss
                for _ in range(2):
                    _, g_loss = sess.run(
                        (train_op_g, G_loss),
                        feed_dict={x_hat: batch_xs_input, x: batch_xs_target, z_sample: samples,
                                   cat_sample: z_id_one_hot_vector,keep_prob: 0.9})

                # supervised phase
                    _, cat_loss = sess.run(
                        (train_op_cat, cat_gen_loss),
                        feed_dict={x_hat: batch_xs_sup_input, x: batch_xs_sup_target, x_id: batch_ids_sup_input, keep_prob: 0.9})
                

            tot_loss = loss_likelihood + d_loss + g_loss + cat_loss 

            # print cost every epoch
            print("epoch %d: L_tot %03.2f L_likelihood %03.4f d_loss %03.2f g_loss %03.2f " % (epoch, tot_loss, loss_likelihood, d_loss, g_loss))

            #for v in sess.graph.get_operations():
            #    print(v.name)
            # if minimum loss is updated or final epoch, plot results
            if epoch%2==0 or min_tot_loss > tot_loss or epoch+1 == n_epochs:
                min_tot_loss = tot_loss
                # Plot for reproduce performance
                if PRR:
                    y_PRR = sess.run(y, feed_dict={x_hat: x_PRR, keep_prob : 1})
                    save_subimages([x_PRR[:10],y_PRR[:10]],"./results/Reco_%02d" %(epoch))
                    #y_PRR_img = y_PRR.reshape(PRR.n_tot_imgs, tsamples, fsamples)
                    #PRR.save_images(y_PRR_img, name="/PRR_epoch_%02d" %(epoch) + ".jpg")

                # Plot for manifold learning result
                if PMLR and dim_z == 2:
                    y_PMLR = sess.run(decoded, feed_dict={z_in: PMLR.z, keep_prob : 1})
                    y_PMLR_img = y_PMLR.reshape(PMLR_n_img_x,PMLR_n_img_x,tsamples,fsamples)
                    save_subimages(y_PMLR_img,"./results/Mani_%02d" %(epoch))
                    #y_PMLR_img = y_PMLR.reshape(PMLR.n_tot_imgs, fsamples, tsamples)
                    #PMLR.save_images(y_PMLR_img, name="/PMLR_epoch_%02d" % (epoch) + ".jpg")

                    # plot distribution of labeled images
                    z_PMLR = sess.run(z, feed_dict={x_hat: x_PMLR, keep_prob : 1})
                    PMLR.save_scattered_image(z_PMLR,id_PMLR, name="/PMLR_map_epoch_%02d" % (epoch) + ".jpg", N=nlabels)
                else:
                    retcat,test_cat_loss, test_ll = sess.run((cat,cat_gen_loss,neg_marginal_likelihood), feed_dict={x_hat: x_PMLR, x_id: id_PMLR, x:x_PMLR,keep_prob : 1})
                    print("Accuracy: ",100.0 * np.sum(np.argmax(retcat, 1) == np.argmax(id_PMLR, 1))/retcat.shape[0], test_cat_loss, test_ll)
                    save_loss = test_cat_loss + test_ll 
                    if prev_loss > save_loss  and (epoch%100==0):# and epoch!=0:
                        prev_loss = save_loss
                        #save_graph(sess,"./savedmodels/","saved_checkpoint","checkpoint_state","input_graph.pb","output_graph.pb",encoder+"_encoder_cat/zout/BiasAdd,"+encoder+"_encoder_cat/catout/Softmax,CNN_decoder/reshaped/Reshape,discriminator_cat_1/add_2,discriminator_1/add_2")
                        save_path = saver.save(sess, "./savedmodels_allsensors/allsensors.ckpt")
                        tf.train.write_graph( sess.graph_def, "./savedmodels_allsensors/", "allsensors.pb", as_text=False )
                    #for i in range(dim_z):
                    #    print i, np.min(z_PMLR[:, i]), np.max(z_PMLR[:, i])


if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # main
    main(args)
