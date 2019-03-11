import tensorflow as tf
import tflearn
import sys
import time
from ActorCriticLSTM import *
from Board import *
import os

from collections import deque
import pickle

import atexit
import random
from Pieces import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

actiondim = {_ for _ in range(81)}
minibatchSize = 25


def ex(buf, act, crit):
    crit.save()
    act.save()
    buf.save(('./rBuffer.pickle'))


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


def train(sess, env, actor, critic, actor_noise):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    total_moves = 0

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('summary', sess.graph)  # summary_dir

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(3000000)  # bufferSize, random_seed
    # atexit.register(ex, replay_buffer, actor, critic)
    if (os.path.isfile('./rBufferLSTM.pickle')):
        print('Loading buffer')
        replay_buffer.load('./rBufferLSTM.pickle')
    # Needed to enable BatchNorm.
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    tflearn.is_training(True)

    iterations = 0
    for i in range(500):  # max episodes
        b = Board()
        s = b.boardCode
        ep_reward = 0
        ep_ave_max_q = 0
        last_time = time.time()
        actor.save()
        critic.save()
        if i != 0:
            print('----- LAST BOARD --- ')
            lastboard.printBoard()
        lastboard = b
        j = 0
        mem = deque(maxlen=5)
        moves = 0
        for _ in range(5):
            mem.append(s)
        while not b.gameOver():  # max episode len
            moves = b.moves
            mem2 = deque(maxlen=5)
            for _ in mem:
                mem2.append(_)

            j += 1
            iterations += 1
            if iterations % 5000 == 0 and iterations != 0:
                actor.save()
                critic.save()
                replay_buffer.save('./rBufferLSTM.pickle')
            # try:
            if iterations % 3000 == 0 and iterations != 0:

                xc = b.randomMove()
                x = (b.c2n[xc[0], xc[1]], b.c2n[xc[2], xc[3]])
                try:
                    terminal, s2, r = b.takeTurn((xc[0], xc[1]), (xc[2], xc[3]), False)
                    mem2.append(s2)
                except Exception as e:
                    print('------errrrrr')
                    b.printBoard()
                    print(e)
                    j -= 1
                    exit()
            else:
                an = actor_noise()
                a = actor.predict(np.reshape(mem, (1, 5, 82))) + an
                a = list(map(abs, a))
                x = list(map(lambda x: int(round(x)), a[0]))
                try:
                    xc1 = b.n2c[x[0]]
                    xc2 = b.n2c[x[1]]
                except:
                    xc1 = 8
                    xc2 = 8
                try:
                    terminal, s2, r = b.takeTurn(xc1, xc2, True)
                    mem2.append(s2)
                except Exception as e:
                    print('------errrrrr')
                    b.printBoard()
                    print(e)
                    j -= 1
                    exit()
                if j > 4:
                    rep = True
                    for i in range(82):
                        if mem2[-1][i] == mem2[-2][i]:
                            continue
                        else:
                            rep = False
                            break
                    if rep:
                        r -= 4
            replay_buffer.add(np.reshape(mem, (5, 82)), np.reshape(x, (2)), r, terminal, np.reshape(mem2, (5, 82)))
            if replay_buffer.size() > minibatchSize:
                mem_batch, a_batch, r_batch, t_batch, mem2_batch = \
                    replay_buffer.sample_batch(minibatchSize)

                # Calculate targets
                target_q = critic.predict_target(
                    mem2_batch,
                    actor.predict_target(mem2_batch))

                y_i = []
                for k in range(minibatchSize):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    mem_batch, a_batch, np.reshape(y_i, (minibatchSize, 1)))  # (minibatchSize, 1)

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(mem_batch)
                grads = critic.action_gradients(mem_batch, a_outs)
                actor.train(mem_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

                for _ in mem2:
                    mem.append(_)
                ep_reward += r

                lastboard = b

                if j == 2501:
                    total_moves += moves
                    break
                if terminal:
                    total_moves += moves
                    summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: ep_reward,
                        summary_vars[1]: ep_ave_max_q / float(j + 1)
                    })

                    writer.add_summary(summary_str, i)
                    writer.flush()

                    print('\033[92m' + '| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} \033[0m'.format(int(ep_reward), \
                                                                                                      i, (
                                                                                                                  ep_ave_max_q / float(
                                                                                                              j + 1))))
                    break
        print('\033[92m' + 'Saving buffer \033[0m')
        print('----- \033[92m EPISODE {}, TOTAL MOVES: {}, LAST EPISODE: {} \033[0m -----'.format(i + 1, total_moves,
                                                                                                  moves))
        replay_buffer.save('./rBuffer.pickle')

    return actor, critic


with tf.Session() as sess:
    aenv = Board()
    aactor = ActorNetwork(sess, [None, 5, 82], actiondim, 81, 0.0005, 0.05, 25)
    acritic = CriticNetwork(sess, [None, 5, 82], actiondim, 0.001, 0.125, 0.9, aactor.get_num_trainable_vars())
    if (os.path.isfile('./actorLSTM/actorLSTM.ckpt.index')):
        aactor.load()
    if (os.path.isfile('./criticLSTM/criticLSTM.ckpt.index')):
        acritic.load()
    aactor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(2))
    asses = tf.Session()
    aactor, acritic = train(sess, aenv, aactor, acritic, aactor_noise)
    aactor.save()
    acritic.save()
