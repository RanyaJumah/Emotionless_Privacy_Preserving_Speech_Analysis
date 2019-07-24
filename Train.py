import os
import time
import numpy as np

from Models.cyclegan_vc2 import CycleGAN2
from Preprocess.speech_tools import load_pickle, sample_train_data

np.random.seed(300)

dataset = 'RAVDESS'
src_speaker = 'Emotion'
trg_speaker = 'Neutral'
model_name = 'cyclegan_vc2_two_step'
os.makedirs(os.path.join('experiments', dataset, model_name, 'checkpoints'), exist_ok=True)
log_dir = os.path.join('logs', '{}_{}'.format(dataset, model_name))
os.makedirs(log_dir, exist_ok=True)

data_dir = os.path.join('datasets', dataset)
exp_dir = os.path.join('experiments', dataset)

train_A_dir = os.path.join(data_dir, 'training', src_speaker)
train_B_dir = os.path.join(data_dir, 'training', trg_speaker)
exp_A_dir = os.path.join(exp_dir, src_speaker)
exp_B_dir = os.path.join(exp_dir, trg_speaker)

# Data parameters
sampling_rate = 22050
num_mcep = 36
frame_period = 5.0
n_frames = 128

# Training parameters
num_iterations = 200000
mini_batch_size = 1
generator_learning_rate = 0.0002
discriminator_learning_rate = 0.0001
lambda_cycle = 10
lambda_identity = 5

print('Loading cached data...')
coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A = load_pickle(
    os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)))
coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B = load_pickle(
    os.path.join(exp_B_dir, 'cache{}.p'.format(num_mcep)))

model = CycleGAN2(num_features=num_mcep, batch_size=mini_batch_size, log_dir=log_dir)

model.load(
    filepath=os.path.join('experiments', dataset, model_name, 'checkpoints', '{}_7500.ckpt'.format(model_name)))



iteration = 1
while iteration <= num_iterations:
    start_time = time.time()

    dataset_A, dataset_B = sample_train_data(dataset_A=coded_sps_A_norm, dataset_B=coded_sps_B_norm, n_frames=n_frames)
    n_samples = dataset_A.shape[0]

    for i in range(n_samples // mini_batch_size):
        if iteration > 10000:
            lambda_identity = 0

        start = i * mini_batch_size
        end = (i + 1) * mini_batch_size

        generator_loss, discriminator_loss = model.train(input_A=dataset_A[start:end], input_B=dataset_B[start:end],
                                                         lambda_cycle=lambda_cycle, lambda_identity=lambda_identity,
                                                         generator_learning_rate=generator_learning_rate,
                                                         discriminator_learning_rate=discriminator_learning_rate)

        if iteration % 10 == 0:
            print('Iteration: {:07d}, Generator Loss : {:.3f}, Discriminator Loss : {:.3f}'.format(iteration,
                                                                                                   generator_loss,
                                                                                                   discriminator_loss))
        if iteration % 2500 == 0:
            print('Checkpointing...')
            model.save(directory=os.path.join('experiments', dataset, model_name, 'checkpoints'),
                       filename='{}_{}.ckpt'.format(model_name, iteration))

            end_time = time.time()
            time_elapsed = end_time - start_time

            print('Time Elapsed for 2500: %02d:%02d:%02d' % (
                time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

        iteration += 1


