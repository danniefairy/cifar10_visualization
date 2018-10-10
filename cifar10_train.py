import tensorflow as tf
import pickle
import numpy as np
import os
from PIL import Image
import sys


TRAIN = False if len(sys.argv)<2 else sys.argv[1]
MIN_DEQUE = 1024*2 if TRAIN else 64
BATCH_SIZE = 64 if TRAIN else 512
EP = 20000


tfrecords_filename1 = '.\cifar10_train.tfrecords'
tfrecords_filename2 = '.\cifar10_test.tfrecords'

def read_and_decode(filename_queue):
    # 建立 TFRecordReader
    reader = tf.TFRecordReader()

    # 讀取 TFRecords 的資料
    _, serialized_example = reader.read(filename_queue)

    # 讀取一筆 Example
    features = tf.parse_single_example(
    serialized_example,
    features={
      'image_string': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64)
      })

    # 將序列化的圖片轉為 uint8 的 tensor
    image = tf.decode_raw(features['image_string'], tf.uint8)

    # 將 label 的資料轉為 float32 的 tensor
    label = tf.cast(features['label'], tf.int32)
    
    image = tf.reshape(image, [32*32*3])


    # 打散資料順序
    images, labels = tf.train.shuffle_batch(
    [image, label],
    batch_size=BATCH_SIZE,
    capacity=MIN_DEQUE+30*BATCH_SIZE,
    num_threads=2,
    min_after_dequeue=MIN_DEQUE)

    return images, labels

# 建立檔名佇列
filename_queue1 = tf.train.string_input_producer(
  [tfrecords_filename1], num_epochs=30)
# 讀取並解析 TFRecords 的資料
images, labels = read_and_decode(filename_queue1)

# 建立檔名佇列
filename_queue2 = tf.train.string_input_producer(
  [tfrecords_filename2], num_epochs=30)
# 讀取並解析 TFRecords 的資料
val_img, val_lab = read_and_decode(filename_queue2)

# model setting
x = tf.placeholder(tf.float32,shape=[None,32,32,3],name='img')
y = tf.placeholder(tf.float32,shape=[None,10],name='label')
prob_dropout = tf.placeholder(tf.float32)

def CNN(inputs):
    n_channel = [3,96,256,256+128,256]
    length = 32
    for i in range(4):
        with tf.variable_scope('cnn_{}'.format(str(i+1))):
            kernel = tf.Variable(tf.truncated_normal([3, 3, n_channel[i], n_channel[i+1]], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[n_channel[i+1]], dtype=tf.float32), name='biases')
            bias = tf.nn.bias_add(conv, biases)
            tf.layers.batch_normalization(bias)
            conv_activated = tf.nn.relu(bias)
            if i<=2:
                pool = tf.nn.max_pool(conv_activated ,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME',
                         name='pool')
                inputs = pool
                length/=2
            else:
                inputs = conv_activated
            
    length = int(length)
    return tf.reshape(inputs ,[-1,length*length*n_channel[-1]])

def DNN(inputs,prob_dropout):
    n_node = [inputs.get_shape()[1],4096,4096,10]
    for i in range(3):
        with tf.variable_scope('dnn_{}'.format(str(i+1))):
            weights = tf.get_variable("weights", [n_node[i], n_node[i+1]], initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases= tf.Variable(tf.constant(0.0, shape=[n_node[i+1]], dtype=tf.float32),  name='biases')
            fc=tf.add(tf.matmul(inputs,weights),biases)            
            if i!=len(n_node)-2:
                fc_activated = tf.nn.relu(fc)
                fc_activated=tf.nn.dropout(fc_activated,prob_dropout)
            else:
                fc_activated = fc

            inputs = fc_activated
    return inputs

features = CNN(x)
output = DNN(features,prob_dropout)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

pred = tf.nn.softmax(output)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# 初始化變數
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())


def img_pre(img):
    arr = []
    for m in img:        
        # 照片前處理
        m=np.true_divide(m, 255)
        arr.append(np.rot90(
                            np.transpose([
                                np.reshape(m[:1024],[32,32]),
                                np.reshape(m[1024:1024*2],[32,32]),
                                np.reshape(m[-1024:],[32,32])
                            ]),-1))
    return arr

def lab_pre(lab):
    arr = []
    for l in lab:
        tmp = [0]*10
        tmp[l-1]=1
        arr.append(tmp)
    return arr

with tf.Session()  as sess:
    # 初始化
    sess.run(init_op)


    save_path = './Save/model.ckpt'
    saver = tf.train.Saver()
    if tf.train.get_checkpoint_state(os.path.dirname(save_path)):            
        saver.restore(sess,save_path)
        print("Model restored")
    else:
        #sess.run(tf.global_variables_initializer())
        sess.run(init_op)
        print("No model is found")


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)




    # 示範用的簡單迴圈
    if TRAIN:
        for i in range(EP):

            img, lab = sess.run([images, labels])

            train_data = img_pre(img)
            train_label = lab_pre(lab)
            

            

            # batch data 資訊
            #print(np.shape(train_data),np.shape(train_label))
            #print(train_data[0],train_label[0])
            #break


            batch_loss,batch_acc,_ = sess.run([loss,accuracy,optimizer],feed_dict={x:train_data,y:train_label,prob_dropout:0.8})
            if i % 50 == 0:
                vimg, vlab= sess.run([val_img, val_lab])
                vimg = img_pre(vimg)
                vlab = lab_pre(vlab)
                val_loss,val_acc,vpred = sess.run([loss,accuracy,pred],feed_dict={x:vimg,y:vlab,prob_dropout:1.0})
                print("ep: {} , loss: {:.2f}, acc: {:.2f}, val_loss: {:.2f}, val_acc: {:.2f}".format(i,batch_loss,batch_acc,val_loss,val_acc))

                saver.save(sess,save_path,write_meta_graph = False)
    else:
        vimg, vlab = sess.run([val_img, val_lab])
        vimg = img_pre(vimg)
        vlab = lab_pre(vlab)
        val_loss,val_acc = sess.run([loss,accuracy],feed_dict={x:vimg,y:vlab,prob_dropout:1.0})
        print("val_loss: {:.2f}, val_acc: {:.2f}".format(val_loss,val_acc))

        # filter visualization
        folder_name = "filter/"
        layer_name = "cnn_1/weights:0"
        layer_weight = [v for v in tf.trainable_variables() if v.name==layer_name]
        layers = sess.run(layer_weight)[0]
        for i in range(len(layers[0,0,0,:])):
            tmp = layers[:,:,:,i]
            lys = (tmp-np.min(tmp))/(np.max(tmp)-np.min(tmp))*255.0      
            im = Image.fromarray(lys.astype('uint8'))
            if not os.path.exists(folder_name+layer_name.split(':')[0]):
                os.makedirs(folder_name+layer_name.split(':')[0])
            im.save("{}/{}.png".format(folder_name+layer_name.split(':')[0],i+1))

        # saliency map
        correct_scores = tf.gather_nd(pred,tf.stack((tf.range(BATCH_SIZE), tf.cast(tf.argmax(y,axis=1),tf.int32)), axis=1))
        #saliency_loss = correct_scores # right score
        saliency_loss = tf.reduce_max(pred,1) # max value => saliency map
        #target_scores = tf.gather_nd(pred,tf.stack((tf.range(BATCH_SIZE), tf.cast(tf.argmax(target,axis=1),tf.int32)), axis=1))
        #saliency_loss = correct_scores # target score => image fool,deep dream

        grad_img = tf.gradients(saliency_loss ,x)
        grad_saliency,score = sess.run([grad_img,correct_scores]  ,feed_dict={x:vimg,y:vlab,prob_dropout:1.0})

        map_saliency = np.sum(np.maximum(grad_saliency[0],0),axis=3) # avg impact
        #map_saliency = np.max(grad_saliency[0],axis=3) # max impact

        if not os.path.exists("saliency"):
                os.makedirs("saliency")

        count = 0
        for i in np.where(score>0.8)[0]:
            
            #print('score:',score[i])
            im = Image.fromarray((vimg[i]*255.0).astype('uint8'))
            im.save('saliency/ori_{}.png'.format(count+1))

            tmp = (map_saliency[i]-np.min(map_saliency[i]))/np.max(map_saliency[i])
            tmp = np.where(tmp<np.max(tmp)*0.5,0,255)
            im = Image.fromarray(tmp.astype('uint8'))
            im.save('saliency/sal_{}.png'.format(count+1))

            count +=1

        # features
        fts = sess.run(features  ,feed_dict={x:vimg,y:vlab,prob_dropout:1.0})
        if not os.path.exists("features"):
            os.makedirs("features")
        file = open('features/fts.pkl', 'wb')
        pickle.dump([fts,np.argmax(vlab,axis=1)], file)


    coord.request_stop()
    coord.join(threads)