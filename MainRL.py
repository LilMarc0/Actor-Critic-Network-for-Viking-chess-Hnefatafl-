import tensorflow as tf
import tflearn
import sys
import time
from ActorCritic import *
from Board import *
import os

import atexit
import random
from Pieces import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

actiondim = {_ for _ in range(81)}
minibatchSize = 100


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
    replay_buffer = ReplayBuffer(1000000)  # bufferSize, random_seed
    # atexit.register(ex, replay_buffer, actor, critic)
    if (os.path.isfile('./rBuffer.pickle')):
        print('Loading buffer')
        replay_buffer.load('./rBuffer.pickle')
    # Needed to enable BatchNorm.
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    tflearn.is_training(True)

    for i in range(500):  # max episodes
        b = Board()
        s = b.boardCode
        ep_reward = 0
        ep_ave_max_q = 0
        last_time = time.time()
        print('-------------- NEXT EPISODE -----------')
        actor.save()
        critic.save()
        if i != 0:
            print('----- LAST BOARD --- ')
            lastboard.printBoard()
        lastboard = b
        j = 0
        while not b.gameOver():  # max episode len
            # print('||||')

            j += 1
            if j % 25000 == 0 and j != 0:
                actor.save()
                critic.save()
                replay_buffer.save('./rBuffer.pickle')
            x = []
            # try:
            if replay_buffer.count % 1000 == 0 and replay_buffer.count != 0:

                xc = b.randomMove()
                # print('Random---xc=', xc)
                x = (b.c2n[xc[0], xc[1]], b.c2n[xc[2], xc[3]])
                try:
                    terminal, s2, r = b.takeTurn((xc[0], xc[1]), (xc[2], xc[3]), False)
                except Exception as e:
                    b.printBoard()
                    print(e)
                    j -= 1
                    exit()
            else:
                an = actor_noise()
                a = actor.predict(np.reshape(s, (1, 82))) + an
                a = list(map(abs, a))
                x = list(map(lambda x: int(round(x)), a[0]))
                try:
                    xc1 = b.n2c[x[0]]
                    xc2 = b.n2c[x[1]]
                except:
                    xc1 = (8,8)
                    xc2 = (8,8)
                # print('Pred-----xc12', xc1, xc2)
                try:
                    terminal, s2, r = b.takeTurn(xc1, xc2, True)
                    
                except Exception as e:
                    b.printBoard()
                    print(e)
                    j -= 1
                    exit()
            print(xc1, xc2)
            replay_buffer.add(np.reshape(s, (82)),
     np.reshape(x, (2)), r, terminal,
     np.reshape(s2, (82)))
            if replay_buffer.count % 250 == 0:
                print(replay_buffer.count, x, time.time() - last_time)
                last_time = time.time()

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > minibatchSize:  # minibatchSize
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(minibatchSize)  # minibatchSize

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(minibatchSize):  # minibatchSize
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (minibatchSize, 1)))  # (minibatchSize, 1)

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            lastboard = b
            if j == 50002:
                break
            if terminal:
                total_moves += b.moves
                print(total_moves)
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
            # print('FPS: {}'.format(1/(time.time()-last_time)))
            # except Exception as e:
            #     print(e)
            #     print(
            #         '----------- ERRRRRRR --------------'
            #     )
            #     lastboard.printBoard()
            #     print('Saving buffer')
            #     replay_buffer.save('./rBuffer.pickle')
            #     actor.save()
            #     critic.save()

        print('Saving buffer')
        replay_buffer.save('./rBuffer.pickle')
    return actor, critic


with tf.Session() as sess:
    aenv = Board()
    aactor = ActorNetwork(sess, [None, 82], actiondim, 121, 0.0005, 0.05, 100)
    acritic = CriticNetwork(sess, [None, 82], actiondim, 0.001, 0.15, 0.75, aactor.get_num_trainable_vars())
    if (os.path.isfile('./actor/actor.ckpt.index')):
        aactor.load()
    if (os.path.isfile('./critic/critic.ckpt.index')):
        acritic.load()
    aactor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(2))
    asses = tf.Session()
    aactor, acritic = train(sess, aenv, aactor, acritic, aactor_noise)
    aactor.save()
    acritic.save()
