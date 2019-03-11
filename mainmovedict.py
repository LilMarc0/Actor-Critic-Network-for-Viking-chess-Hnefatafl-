import tensorflow as tf
import tflearn
import sys
import time
from actorcriticmoves import *
from Board import *
import os

from collections import deque
import pickle
from Pieces import *
from random import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

actiondim = {0,1}
minibatchSize = 25


move_dict = pickle.load(open('moves7x7.pickle', 'rb'))



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
    writer = tf.summary.FileWriter('summary', sess.graph)         # summary_dir

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(750)        # bufferSize, random_seed
    #atexit.register(ex, replay_buffer, actor, critic)
    if(os.path.isfile('./rBufferF.pickle')):
        print('Loading buffer')
        replay_buffer.load('./rBufferF.pickle')

    tflearn.is_training(True)

    iterations = 0
    movhist = deque(maxlen=5)
    rmov = 50
    for i in range(500):  # max episodes
        b = Board()
        s = b.boardMatrix
        ep_reward = 0
        ep_ave_max_q = 0
        last_time = time.time()
        actor.save()
        critic.save()
        replay_buffer.save('./rBufferF.pickle')
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


            # Guess the move
            an = actor_noise()
            a = actor.predict(np.reshape(mem, (1, 5, 7, 7)))[0] + an
            move = move_dict[np.argmax(a)]
            moveFrom = b.n2c[move[0]]
            moveTo = b.n2c[move[1]]

            # Make the move
            ######
            target = b.board.get(b.n2c[move[0]], None)
            if target:
                if target.isValid(moveFrom, moveTo, target.color, b.board):
                    terminal, s2, r = b.takeTurn(moveFrom,
                                                 moveTo,
                                                 True)
                    movhist.append(move)
                    mem2.append(s2)
                else:
                    if random() > 0.95:
                        terminal, s2, r = b.takeTurn(moveFrom,
                                                     moveTo,
                                                     True)
                        movhist.append(move)
                        mem2.append(s2)
                    else:
                        j-=1
                        continue
            else:
                if random() > 0.95:
                    terminal, s2, r = b.takeTurn(moveFrom,
                                                 moveTo,
                                                 True)
                    movhist.append(move)
                    mem2.append(s2)
                else:
                    j -= 1
                    continue
            ######

            if len(movhist) >= 3:
                if movhist[-1] == movhist[-2] == movhist[-3]:
                    r-=4
                    #print('--rep punish')

            if iterations % 500 == 0 and iterations != 0:
                print(move)
            #if iterations % 250 == 0 and iterations != 0:
            #    print(time() - last_time)
            #print(x)
            #print(replay_buffer.size())
            replay_buffer.add(np.reshape(mem, (5, 7, 7)), np.reshape(a, (588)), r, terminal, np.reshape(mem2, (5, 7, 7)))
            rot, rot2, fx = b.genFlips(mem, mem2, move)

            # rotatii
            for _ in range(3):
                replay_buffer.add(np.reshape(rot[_], (5, 7, 7)),
                                  np.reshape(fx[_], (588)),
                                  r,
                                  terminal,
                                  np.reshape(rot2[_], (5, 7, 7)))

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

                # reset
                if j == 7500 or terminal:
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
    return actor, critic

with tf.Session() as sess:
    aenv = Board()
    aactor = ActorNetwork(sess, [None, 5, 7, 7], actiondim, 1, 0.003, 0.125, 25) # lr, tau, batchSize
    acritic = CriticNetwork(sess, [None, 5, 7, 7], actiondim, 0.03, 0.125, 0.85, aactor.get_num_trainable_vars()) # lr, tau, gamma, trVars
    if(os.path.isfile('./actorF/actorF.ckpt.index')):
        aactor.load()
    if (os.path.isfile('./criticF/criticF.ckpt.index')):
        acritic.load()
    aactor_noise = OrnsteinUhlenbeckActionNoise(mu = np.zeros(588))
    asses = tf.Session()
    aactor, acritic = train(sess, aenv, aactor, acritic, aactor_noise)
    aactor.save()
    acritic.save()

