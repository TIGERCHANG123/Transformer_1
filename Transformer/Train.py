# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from models.Transformer import Transformer, create_masks
from datasets.pt_to_en import pt_to_en_dataset
from show_pic import draw
from evaluate import test

model_dataset = 'translate_pt_to_en'
root = 'D:/Automatic/SRTP/GAN/temp'

def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

class train_one_epoch():
    def __init__(self, transformer, train_dataset, val_dataset, optimizer, metrics):
        self.transformer = transformer
        self.optimizer = optimizer
        self.train_loss, self.train_accuracy, self.val_loss, self.val_accuracy = metrics
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ])
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ])
    def val_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        predictions, _ = self.transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

        self.val_loss(loss)
        self.val_accuracy(tar_real, predictions)

    def train(self, epoch,  pic):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.val_loss.reset_states()
        self.val_accuracy.reset_states()
        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(self.train_dataset):
            self.train_step(inp=inp, tar=tar)
            pic.add([self.train_loss.result(), self.train_accuracy.result()])
            pic.save(root + '/temp_pic_save/' + model_dataset)
        accuracy = 0.0
        loss = 0.0
        t=0
        for (val_batch, (inp, tar)) in enumerate(self.val_dataset):
            self.val_step(inp, tar)
            loss += self.val_loss.result()
            accuracy += self.val_accuracy.result()
            t+=1
        print('epoch: {}, loss: {}, accuracy: {}'.format(epoch, self.val_loss.result()/t, self.val_accuracy.result()/t))

#optimizer
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    if not (os.path.exists(root + '/temp_pic/' + model_dataset)):
        os.makedirs(root + '/temp_pic/' + model_dataset)
    if not (os.path.exists(root + '/temp_pic_save/' + model_dataset)):
        os.makedirs(root + '/temp_pic_save/' + model_dataset)
    if not(os.path.exists(root + '/temp_txt_save/'+model_dataset)):
        os.makedirs(root + '/temp_txt_save/'+model_dataset)
    pic = draw(10)
    if not(os.path.exists(root+'/temp_txt_save/'+model_dataset+'/validation.txt')):
        txt = open(root+'/temp_txt_save/'+model_dataset+'/validation.txt','w')
    else:
        txt = open(root+'/temp_txt_save/'+model_dataset+'/validation.txt', 'a')
    dataset = pt_to_en_dataset()
    train_dataset = dataset.train_dataset()
    val_dataset = dataset.valid_dataset()
    test_dataset = dataset.test_dataset()

    d_model = 128
    learning_rate = CustomSchedule(d_model=d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,epsilon=1e-9)

    transformer = Transformer(num_layers=4, d_model=d_model, num_heads=8, dff=512,
                              input_vocab_size=dataset.input_vocab_size, target_vocab_size=dataset.target_vocab_size,
                              pe_input=dataset.input_vocab_size,
                              pe_target=dataset.target_vocab_size,
                              rate=0.1)

    checkpoint_path = root + '/temp_model_save/' + model_dataset
    ckpt = tf.train.Checkpoint(optimizer=optimizer,transformer=transformer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='train_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    train = train_one_epoch(transformer=transformer, train_dataset=train_dataset, val_dataset=val_dataset,
              optimizer=optimizer, metrics=[train_loss, train_accuracy, val_loss, val_accuracy])
    for epoch in range(10):
        train.train(epoch=epoch, pic=pic)
        pic.show(root+'/temp_pic/' + model_dataset + '/pic')
        if (epoch + 1) % 5 == 0:
            ckpt_manager.save()
    test(transformer, dataset, test_dataset, txt)
    txt.close()
    return

if __name__ == '__main__':
    main()