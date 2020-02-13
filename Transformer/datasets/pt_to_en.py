from __future__ import print_function
import tensorflow_datasets as tfds
import tensorflow as tf

class pt_to_en_dataset():
    def __init__(self, data_path='D:/Automatic/SRTP/GAN/datasets'):
        self.MAX_LENGTH = 40
        self.BUFFER_SIZE=20000
        self.BATCH_SIZE=64

        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,data_dir=data_path+'/ted_hrlr_translate', as_supervised=True)
        train_examples, val_examples = examples['train'], examples['validation']
        self.tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

        self.tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)
        self.input_vocab_size = self.tokenizer_pt.vocab_size + 2
        self.target_vocab_size = self.tokenizer_en.vocab_size + 2

        self.train_examples = train_examples
        self.val_examples = val_examples
        self.test_examples = examples['test']

    def encode(self, lang1, lang2):
        tokenizer_pt=self.tokenizer_pt
        tokenizer_en=self.tokenizer_en
        lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(lang1.numpy()) + [tokenizer_pt.vocab_size + 1]
        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang2.numpy()) + [tokenizer_en.vocab_size + 1]
        return lang1, lang2

    def filter_max_length(self, x, y):
      return tf.logical_and(tf.size(x) <= self.MAX_LENGTH,tf.size(y) <= self.MAX_LENGTH)
    def tf_encode(self, pt, en):
      result_pt, result_en = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])
      result_pt.set_shape([None])
      result_en.set_shape([None])

      return result_pt, result_en
    def train_dataset(self):
        train_dataset = self.train_examples.map(self.tf_encode)
        train_dataset = train_dataset.filter(self.filter_max_length)
        # 将数据集缓存到内存中以加快读取速度。
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(self.BUFFER_SIZE).padded_batch(self.BATCH_SIZE,  padded_shapes=([None],[None]))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset
    def valid_dataset(self):
        val_dataset = self.val_examples.map(self.tf_encode)
        val_dataset = val_dataset.filter(self.filter_max_length).padded_batch(self.BATCH_SIZE,  padded_shapes=([None],[None]))
        return val_dataset
    def test_dataset(self):
        test_dataset = self.test_examples.map(self.tf_encode)
        test_dataset = test_dataset.filter(self.filter_max_length).batch(1)
        return test_dataset


dataset = pt_to_en_dataset()
train_dataset = dataset.train_dataset()


