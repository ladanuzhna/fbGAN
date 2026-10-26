import numpy as np
import tensorflow as tf

from gan import GAN
from globals import *
from utils.protein_utilities import DNA_to_protein
from utils.data_utilities import get_data_for_feedback, transform_sequence, OneHot_Seq


class GAN_FBNet():

    def __init__(
        self,
        generator_path=None,
        discriminator_path=None,
        fbnet_path=PATH_FB,
        features=DESIRED_FEATURES,
        tokenizer=None
    ):
        self.GAN = GAN(generator_weights_path=generator_path, discriminator_weights_path=discriminator_path)
        self.FBNet = tf.keras.models.load_model(fbnet_path)

        if tokenizer is None:
            _, _, _, _, tokenizer = get_data_for_feedback(PATH_DATA)
        self.tokenizer = tokenizer
        self.label_order = np.array(['B', 'C', 'E', 'G', 'H', 'I', 'S',
                                     'T'])  # order of labels as output by Multilabel binarizer - don't change!
        if not set(features).issubset(self.label_order):
            raise ValueError("Unknown secondary-structure feature")
        self.desired_features = list(features)
        self.data = None
        self.checkpoint_dir = './'

    def get_scores(self, inputs):
        # convert the DNA sequences to protein sequences
        protein_sequence = DNA_to_protein(inputs)
        transformed, _ = transform_sequence(protein_sequence, self.tokenizer, max_len=self.FBNet.input_shape[1])
        # use FBNet to grade the sequences
        scores = self.FBNet.predict(transformed)
        return scores

    def get_score_per_feature(self, scores):
        scores = np.array(scores)
        if scores.size == 0:
            return []
        avg_scores = np.rint(100 * np.mean(scores, axis=0))
        score_per_feature = []
        for feature in self.desired_features:
            i = int(np.where(feature == self.label_order)[0][0])
            score_i = int(avg_scores[i])
            fscore = (feature, score_i)
            score_per_feature.append(fscore)

        return score_per_feature

    def add_samples(self, generated, scores, score_threshold=0.75, replace=False):
        if replace:
            raise NotImplementedError('Replacing training samples is not implemented')
        generated = np.asarray(generated)
        scores = np.asarray(scores)
        desired_indices = [np.where(self.label_order == feature)[0][0] for feature in self.desired_features]
        accepted = np.all(scores[:, desired_indices] > score_threshold, axis=1)
        best_samples = generated[accepted]
        best_scores = scores[accepted]
        if len(best_samples):
            self.data = (
                best_samples.copy() if self.data is None else np.concatenate((self.data, best_samples), axis=0)
            )
        return best_samples, best_scores

    def train(self, inputs, epochs, step_log=50, steps_per_epoch=100, batch_size=BATCH_SIZE):
        self.data = inputs
        if batch_size <= 0 or steps_per_epoch <= 0 or step_log <= 0:
            raise ValueError("Batch size, steps per epoch, and log interval must be positive")
        self.batch_size = batch_size
        self.GAN.batch_size = batch_size

        for epoch in range(epochs):
            dataset = self.create_dataset(self.data)

            print(f'Epoch {epoch} / {epochs}')

            step = 0

            for sample_batch in dataset.take(steps_per_epoch):
                G_loss = self.GAN.G_train_step()
                D_loss, GP = self.GAN.D_train_step(sample_batch)

                generated = self.GAN.generate_samples(number=self.batch_size, decoded=False)
                OneHot = OneHot_Seq(letter_type=TASK_TYPE)
                decoded_generated = OneHot.onehot_to_seq(generated)
                scores = self.get_scores(decoded_generated)
                generated = tf.cast(generated, tf.float32)
                best_samples, best_scores = self.add_samples(generated, scores)

                if step % step_log == 0:
                    print(
                        f'\tStep {step}:   Generator: {G_loss.numpy()}   Discriminator: {D_loss.numpy()}   Samples: {len(self.data)}'
                    )

                    print('\tBest scores per feature: ', end=' ')
                    sc = self.get_score_per_feature(best_scores)
                    pprint = [f'{sc[0]}: {sc[1]}%' for sc in sc]
                    print(*pprint, sep=' ')

                    print('Average scores per feature: ', end=' ')
                    sc = self.get_score_per_feature(scores)
                    pprint = [f'{sc[0]}: {sc[1]}%' for sc in sc]
                    print(*pprint, sep=' ')

                step += 1

            percent_fake = int((len(self.data) - len(inputs)) / len(self.data) * 100)
            print(f'\tPercent of the fake samples in the discriminator: {percent_fake}%.')

    def create_dataset(self, inputs):
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.shuffle(inputs.shape[0], seed=0).batch(self.batch_size, drop_remainder=True)
        return dataset
