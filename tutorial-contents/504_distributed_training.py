"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.4.0
"""

import tensorflow as tf
import multiprocessing as mp
import numpy as np
import os, shutil


TRAINING = True

# training data
x = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise


def work(job_name, task_index, step, lock):
    # set work's ip:port, parameter server and worker are the same steps
    cluster = tf.train.ClusterSpec({
        "ps": ['localhost:2221', ],
        "worker": ['localhost:2222', 'localhost:2223', 'localhost:2224',]
    })
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    if job_name == 'ps':
        # join parameter server
        print('Start Parameter Server: ', task_index)
        server.join()
    else:
        print('Start Worker: ', task_index, 'pid: ', mp.current_process().pid)
        # worker job
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):
            # build network
            tf_x = tf.placeholder(tf.float32, x.shape)
            tf_y = tf.placeholder(tf.float32, y.shape)
            l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)
            output = tf.layers.dense(l1, 1)
            loss = tf.losses.mean_squared_error(tf_y, output)
            global_step = tf.train.get_or_create_global_step()
            train_op = tf.train.GradientDescentOptimizer(
                learning_rate=0.001).minimize(loss, global_step=global_step)

        # set training steps
        hooks = [tf.train.StopAtStepHook(last_step=100000)]

        # get session
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_index == 0),
                                               checkpoint_dir='./tmp',
                                               hooks=hooks) as mon_sess:
            print("Start Worker Session: ", task_index)
            while not mon_sess.should_stop():
                # train
                _, loss_ = mon_sess.run([train_op, loss], {tf_x: x, tf_y: y})
                with lock:
                    step.value += 1
                if step.value % 500 == 0:
                    print("Task: ", task_index, "| Step: ", step.value, "| Loss: ", loss_)
        print('Worker Done: ', task_index)


def parallel_train():
    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')
    # use multiprocessing to create a local cluster with 2 parameter servers and 4 workers
    jobs = [('ps', 0), ('worker', 0), ('worker', 1), ('worker', 2)]
    step = mp.Value('i', 0)
    lock = mp.Lock()
    ps = [mp.Process(target=work, args=(j, i, step, lock), ) for j, i in jobs]
    [p.start() for p in ps]
    [p.join() for p in ps]


def eval():
    tf_x = tf.placeholder(tf.float32, [None, 1])
    l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)
    output = tf.layers.dense(l1, 1)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint('./tmp'))
    result = sess.run(output, {tf_x: x})
    # plot
    import matplotlib.pyplot as plt
    plt.scatter(x.ravel(), y, c='b')
    plt.plot(x.ravel(), result.ravel(), c='r')
    plt.show()


if __name__ == "__main__":
    if TRAINING:
        parallel_train()
    else:
        eval()
