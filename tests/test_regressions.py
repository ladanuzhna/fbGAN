"""Regression checks with synthetic data; no historical checkpoints are required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import tensorflow as tf

import models
from gan import GAN
from gan_fbnet import GAN_FBNet
from globals import MAX_LEN, MAX_LEN_PROTEIN
from utils.data_utilities import (
    OneHot_Seq,
    get_data_for_feedback,
    prepare_dataset,
    transform_sequence,
)
from utils.protein_utilities import DNA_to_protein, protein_to_DNA, translate


class DataTests(unittest.TestCase):

    def test_proline_and_stop_codons(self):
        protein = 'ACDEFGHIKLMNPQRSTVWY'
        encoded = protein_to_DNA(protein)
        self.assertEqual(DNA_to_protein(encoded), ['M' + protein])
        np.testing.assert_array_equal(encoded, protein_to_DNA([protein]))
        self.assertEqual(translate('ATGCCTTAGGGT'), 'MP')
        self.assertEqual(translate(['ATGCCTPPPP']), 'MP')

    def test_onehot_roundtrip(self):
        codec = OneHot_Seq(max_length=12)
        encoded = codec.seq_to_onehot(['ATGC', 'CCTA'])
        self.assertEqual(encoded.shape, (2, 12, 5))
        self.assertEqual(codec.onehot_to_seq(encoded), [['ATGCPPPPPPPP'], ['CCTAPPPPPPPP']])

    def test_reuse_tokenizer_and_variable_lengths(self):
        data, tokenizer = transform_sequence(['ACDE', 'AC'], max_len=8)
        reused, same = transform_sequence(['AC'], tokenizer, max_len=8)
        self.assertIs(same, tokenizer)
        np.testing.assert_array_equal(data[1], reused[0])

    def test_data_length_filter_and_fixed_feedback_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            lengths = [MAX_LEN_PROTEIN] * 10 + [MAX_LEN_PROTEIN - 1] * 10
            pd.DataFrame({
                'seq': ['P' * n for n in lengths],
                'sst8': ['C' * n for n in lengths],
                'len': lengths,
                'has_nonstd_aa': [False] * len(lengths),
            }).to_csv(path, index=False)
            prepared = prepare_dataset(str(path), split=0.2)
            self.assertEqual(prepared.shape, (8, MAX_LEN, 5))
            decoded = OneHot_Seq().onehot_to_seq(prepared)
            self.assertTrue(all(len(row[0]) == MAX_LEN for row in decoded))
            X_train, _, y_train, _, _ = get_data_for_feedback(str(path), max_len=100)
            self.assertEqual(X_train.shape[1], 100)
            self.assertEqual(y_train.shape[1], 8)
            self.assertTrue(np.all(y_train[:, 1] == 1))


class ModelTests(unittest.TestCase):

    def setUp(self):
        self.dim_patch = patch.object(models, 'DIM', 2)
        self.dim_patch.start()

    def tearDown(self):
        self.dim_patch.stop()
        tf.keras.backend.clear_session()

    def test_residual_preserves_input_with_zero_weights(self):
        block = models.ResidualBlock()
        inputs = tf.ones((2, 9, 2))
        block(inputs)
        for weight in block.weights:
            weight.assign(tf.zeros_like(weight))
        np.testing.assert_allclose(block(inputs).numpy(), inputs.numpy())
        self.assertTrue(block.trainable_variables)

    def test_shapes_variable_batches_and_gradient_updates(self):
        gan = GAN(batch_size=2)
        for batch in [1, 3]:
            generated = gan.generate_samples(batch)
            self.assertEqual(generated.shape, (batch, MAX_LEN, 5))
            np.testing.assert_allclose(tf.reduce_sum(generated, axis=-1), 1, atol=1e-6)
            self.assertEqual(gan.D(generated).shape, (batch, 1))
        self.assertTrue(np.isfinite(gan.G_train_step().numpy()))
        # The real batch size may differ from the configured sample batch size.
        loss, penalty = gan.D_train_step(tf.one_hot(tf.zeros((3, MAX_LEN), dtype=tf.int32), 5))
        self.assertTrue(np.isfinite(loss.numpy()))
        self.assertTrue(np.isfinite(penalty.numpy()))

    def test_feedback_has_explicit_configuration(self):
        feedback = models.Feedback(n_words=10, max_len=8)
        output = feedback.model(np.zeros((2, 8), dtype=np.int32))
        self.assertEqual(output.shape, (2, 8))
        self.assertTrue(np.all((output.numpy() >= 0) & (output.numpy() <= 1)))


class FeedbackTests(unittest.TestCase):

    def setUp(self):
        self.fb = GAN_FBNet.__new__(GAN_FBNet)
        self.fb.label_order = np.array(list('BCEGHIST'))
        self.fb.desired_features = ['C', 'H', 'E']
        self.fb.data = np.zeros((2, MAX_LEN, 5), dtype=np.float32)

    def test_empty_selection_keeps_data_shape(self):
        selected, scores = self.fb.add_samples(np.ones((3, MAX_LEN, 5)), np.zeros((3, 8)))
        self.assertEqual(selected.shape, (0, MAX_LEN, 5))
        self.assertEqual(scores.shape, (0, 8))
        self.assertEqual(len(self.fb.data), 2)
        self.assertEqual(self.fb.get_score_per_feature(scores), [])

    def test_only_all_desired_features_are_accepted_once(self):
        scores = np.zeros((3, 8))
        scores[0, [1, 2, 4]] = 0.9
        scores[1, [1, 2]] = 0.9
        scores[2, [1, 2, 4]] = 0.75
        selected, _ = self.fb.add_samples(np.ones((3, MAX_LEN, 5)), scores)
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(self.fb.data), 3)

    def test_feedback_training_honors_batch_and_step_limits(self):

        class FakeGAN:

            def __init__(self):
                self.steps = 0

            def G_train_step(self):
                self.steps += 1
                return tf.constant(0.)

            def D_train_step(self, samples):
                return tf.constant(0.), tf.constant(0.)

            def generate_samples(self, number, decoded):
                return tf.one_hot(tf.zeros((number, MAX_LEN), dtype=tf.int32), 5)

        self.fb.GAN = FakeGAN()
        self.fb.get_scores = lambda _: np.ones((2, 8))
        self.fb.train(np.zeros((8, MAX_LEN, 5)), epochs=1, batch_size=2, steps_per_epoch=2)
        self.assertEqual(self.fb.GAN.batch_size, 2)
        self.assertEqual(self.fb.GAN.steps, 2)
        self.assertEqual(len(self.fb.data), 12)


if __name__ == '__main__':
    unittest.main()
