import tensorflow as tf
from tensorflow import keras
# import numpy as np
from tensorflow.keras.layers import Dropout, LSTM, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation, Embedding, Conv1D, MaxPooling1D, Flatten, \
    GlobalMaxPooling1D
from tensorflow.keras.optimizers.legacy import Adam
# from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.losses import MeanAbsolutePercentageError, Huber, MeanAbsoluteError, MeanSquaredError, \
    CosineSimilarity, BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Accuracy
import random as rn
import os
import numpy as np
from tensorflow.keras import regularizers
sd=89
np.random.seed(sd)
rn.seed(sd)
os.environ['PYTHONHASHSEED'] = str(sd)

from keras import backend as K

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.compat.v1.set_random_seed(sd)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
K.set_session(sess)
os.environ['TF_DETERMINISTIC_OPS'] = '1'



class Linear(keras.layers.Layer):
    def __init__(self, d=32, l=32, unit=32, name='l1'):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(l, l), initializer="random_normal", trainable=True, name=name,
            regularizer=tf.keras.regularizers.l1_l2()
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w)  # ,transpose_b=True)


class SumLinear(keras.layers.Layer):
    def __init__(self):
        super(SumLinear, self).__init__()

    def call(self, input1, input2):
        avg = tf.math.reduce_mean(input2, 0)
        avg = avg[0]
        avg = avg[0]
        return tf.math.add(input1, avg)  # ,transpose_b=True)


##========================== RevIN Normalization layer ===================##
'''
Adaptively learn the variance and mean of shifted and transformed data
'''


class Denormalization(keras.layers.Layer):
    def __init__(self):
        super(Denormalization, self).__init__()

    def call(self, input1, avg, std):
        # input1 = tf.reshape(input1, (1, 1))
        avg = avg[0]
        std = std[0]
        # return tf.math.add(input1 * tf.reduce_mean(std[0]), tf.reduce_mean(avg))
        # todo: testing which one is better
        #output = tf.math.add(input1, tf.reduce_mean(avg))
        output = tf.math.add(input1* tf.reduce_mean(std[0]) , tf.reduce_mean(avg))#* tf.reduce_mean(std[0])

        return tf.math.reduce_mean(output, axis=1)


class DainNormalization(keras.layers.Layer):

    def __init__(self, d=7, l=14):  # , mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001
        super(DainNormalization, self).__init__()

        self.eps = 1e-8

        self.marketD = d
        self.marketL = l

        def get_config(self):
            output = {'market_Delay': self.marketD,
                      'market_features': self.marketL}
            return output

        @classmethod
        def from_config(cls, config):
            return cls(**config)

        self.mean_layer = Linear(d, l, name="LinearLayer1")

        self.scaling_layer = Linear(d, l, name="LinearLayer2")

        self.gating_layer = Linear(d, l, name="LinearLayer3")

    def call(self, inputs):
        # adaptive_avg =  keras_tensor.KerasTensor(type_spec=tf.TensorSpec(shape=(1,1), dtype=tf.float32))
        # adaptive_std =  keras_tensor.KerasTensor(type_spec=tf.TensorSpec(shape=(1,1), dtype=tf.float32))

        # step 1 : adaptive shifting layer
        avg = tf.math.reduce_mean(inputs, 0, keepdims=True)  # mean around y axis

        # print(avg.numpy())
        # print(avg.shape)
        adaptive_avg = self.mean_layer(avg)

        # print(adaptive_avg.numpy())
        # print(inputs.numpy())
        inputs = tf.math.subtract(inputs, adaptive_avg)
        # inputs = inputs - adaptive_avg
        # print(inputs.numpy())
        # Step 2: Normalization by division to STD

        std = tf.math.reduce_mean(inputs ** 2, axis=0, keepdims=True)
        std = tf.math.sqrt(std + self.eps)

        adaptive_std = self.scaling_layer(std)
        # adaptive_std[adaptive_std <= self.eps] = 1
        # may be need to reshape adaptive_avg into (adaptive_std.size(0),adaptive_std.size(1),1)
        inputs = inputs / adaptive_std
        # print(inputs.numpy())

        # Step 3: Supression outliers
        avg = tf.math.reduce_mean(inputs, axis=0, keepdims=True)
        gate = tf.math.sigmoid(self.gating_layer(avg))
        # may be need to reshape adaptive_avg into (gate.size(0), gate.size(1),1)
        inputs = inputs * gate

        # ToDO: test avg or adaptive_avg ? which one is Better?

        # ToDO: How can we use std or adaptive_std?

        # AVG =tf.reduce_mean(adaptive_avg[0])
        # STD =tf.reduce_mean(adaptive_std[0])
        return inputs, adaptive_avg[0], adaptive_std[0]


##========================== End Dain Normalization layer ===================##

##==========================Bahadanu attention layer ===================##
'''
attention_weighits   = Softmax(tanh(W1 * input))
output = attention_weighits * input 
'''


class BahdanauAttentionConcate(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttentionConcate, self).__init__()
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.V = tf.keras.layers.Dense(1)

    def call(self, values_with_time_axis):
        score = self.V(tf.nn.tanh(self.W1(values_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values_with_time_axis
        # weighted summation of input values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


##==========================End of Bahadanu attention layer ===================##


##==========================Hourly news Based Feature Extraction===================##
'''
This layer set to extract temporal features from sequence of news in a day. 
We split day into 4 hours steps and attention layer learn to assign weight to news in each batch.
The output of this layer is a temporal order of extracted features from weighted news in each 4 hour batch.
'''


class newsContextVectorLayer(keras.layers.Layer):
    def __init__(self, embeddingDim, maxNewsPer_4Hour):
        super(newsContextVectorLayer, self).__init__()
        self.maxNewsPer_4Hour = maxNewsPer_4Hour
        self.embeddingDim = embeddingDim

        def get_config(self):
            output = {'maxNewsPer_4Hour': self.maxNewsPer_4Hour,
                      'embeddingDim': self.embeddingDim, }
            return output

        @classmethod
        def from_config(cls, config):
            return cls(**config)

        self.attentionLayer = BahdanauAttentionConcate(self.embeddingDim)

    def call(self, newsEmbeddingSequence):
        '''
        The length of newsEmbeddingSequence matrix should be 6* maxNewsPer_4Hour. since we consider maxNewsPer_4Hour pereach 4 hour and also we have 6 * 4_hour in a day

        :param newsEmbeddingSequence:
        :return:
        '''

        attention_result1, attention_weights1 = self.attentionLayer(
            tf.transpose(newsEmbeddingSequence[:, :, 0:15], perm=[0, 2, 1]))
        attention_result2, attention_weights2 = self.attentionLayer(
            tf.transpose(newsEmbeddingSequence[:, :, 15:30], perm=[0, 2, 1]))
        attention_result3, attention_weights3 = self.attentionLayer(
            tf.transpose(newsEmbeddingSequence[:, :, 30:45], perm=[0, 2, 1]))
        attention_result4, attention_weights4 = self.attentionLayer(
            tf.transpose(newsEmbeddingSequence[:, :, 45:60], perm=[0, 2, 1]))
        attention_result5, attention_weights5 = self.attentionLayer(
            tf.transpose(newsEmbeddingSequence[:, :, 60:75], perm=[0, 2, 1]))
        attention_result6, attention_weights6 = self.attentionLayer(
            tf.transpose(newsEmbeddingSequence[:, :, 75:90], perm=[0, 2, 1]))
        # we have a temporal order of weighted news vectors ,
        # todo: maybe I use Conv1D or LSTM after this layer
        news_features = tf.reshape(tf.concat(
            [attention_result1, attention_result2, attention_result3,
             attention_result4, attention_result5, attention_result6], axis=1),
            shape=(-1, 6, self.embeddingDim))

        news_Attention = tf.reshape(tf.concat(
            [attention_weights1, attention_weights2, attention_weights3, attention_weights4, attention_weights5,
             attention_weights6], axis=1), shape=(-1, 6, self.embeddingDim,))
        return news_features, news_Attention


##========================== End of hourly news  ===================##


##==========================Daily Market Based Feature Extraction===================##
'''
This layer set to extract features from  a sequence of  input features. 
AT FIRST LAYER WE USE ADAPTIVE NORMALIZATION, then feature passed to LSTM that followed by Bahadanu attention 
'''


class AttentionBasedRNN(keras.layers.Layer):
    def __init__(self, d, l, lname, unit=128):
        super(AttentionBasedRNN, self).__init__()
        self.unit = unit
        self.marketD = d
        self.marketL = l
        self.layername = lname

        def get_config(self):
            output = {'market_Delay': self.marketD,
                      'market_features': self.marketL}
            return output

        @classmethod
        def from_config(cls, config):
            return cls(**config)

        self.RnnLayer = LSTM(self.unit, name=self.layername,
                             kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                             bias_regularizer=regularizers.L2(1e-4),
                             activity_regularizer=regularizers.L2(1e-5), return_sequences=True,
                             return_state=True)
        self.dropOutLayer = Dropout(0.2, name="dropoutLayer3")
        self.attentionLayer = BahdanauAttentionConcate(self.unit)

    def call(self, inputs):
        # output, avg, std = self.normalizationLayer(inputs)
        # tf.print(tf.shape(inputs))
        whole_seq_output, final_memory_state, final_carry_state = self.RnnLayer(inputs)
        # tf.print (tf.shape(whole_seq_output))
        whole_seq_output, final_ATTENTION = self.attentionLayer(whole_seq_output)
        # tf.print(tf.shape(whole_seq_output))
        # tf.print('---------------------')
        return whole_seq_output, final_ATTENTION


##==========================End Daily Market Based Feature Extraction===================##
'''
##==========================Hourly Market Based Feature Extraction===================##
def xavier_init(shape):
    # Computes the xavier initialization values for a weight matrix
    in_dim, out_dim = shape
    xavier_lim = tf.sqrt(6.) / tf.sqrt(tf.cast(in_dim + out_dim, tf.float32))
    weight_vals = tf.random.uniform(shape=(in_dim, out_dim),
                                    minval=-xavier_lim, maxval=xavier_lim, seed=22)
    return weight_vals


class DenseLayer(keras.layers.Layer):

    def __init__(self, out_dim, weight_init=xavier_init, activation=tf.identity):
        # Initialize the dimensions and activation functions
        self.out_dim = out_dim
        #self.weight_init = weight_init
        self.activation = activation
        self.built = False

    def __call__(self, x):
        if not self.built:
            # Infer the input dimension based on first call
            self.in_dim = x.shape[2]
            # Initialize the weights and biases
            # initializer="random_normal", trainable=True,
            self.w = self.add_weight(shape=(self.in_dim, self.out_dim), initializer="random_normal",
                                     trainable=True, regularizer=tf.keras.regularizers.l1_l2())
            # self.w = tf.Variable(self.weight_init(shape=(self.in_dim, self.out_dim)))
            self.b = tf.Variable(tf.zeros(shape=(self.out_dim,)))
            self.built = True
        # Compute the forward pass
        z = tf.add(tf.matmul(x, self.w), self.b)
        return self.activation(z)
'''


class highResolutionMarket(keras.layers.Layer):
    def __init__(self, lag, feature_number):
        super(highResolutionMarket, self).__init__()

        self.marketLag = lag
        self.marketFeature = feature_number

        # self.denseLayer = DenseLayer(out_dim=lag)

        def get_config(self):
            output = {'market_Delay': self.marketLag,
                      'market_features': self.marketFeature}
            return output

        @classmethod
        def from_config(cls, config):
            return cls(**config)

        # self.normalizationLayer = DainNormalization(self.marketLag, self.marketFeature)
        self.denseLayer = Dense(units=1, activation='relu')

    def call(self, inputs):
        # output, avg, std = self.normalizationLayer(inputs)

        output1 = self.denseLayer(tf.reshape(inputs[:, 0:4, :], (-1, 4 * self.marketFeature)))
        output2 = self.denseLayer(tf.reshape(inputs[:, 4:8, :], (-1, 4 * self.marketFeature)))
        output3 = self.denseLayer(tf.reshape(inputs[:, 8:12, :], (-1, 4 * self.marketFeature)))
        output4 = self.denseLayer(tf.reshape(inputs[:, 12:16, :], (-1, 4 * self.marketFeature)))
        output5 = self.denseLayer(tf.reshape(inputs[:, 16:20, :], (-1, 4 * self.marketFeature)))
        output6 = self.denseLayer(tf.reshape(inputs[:, 20:24, :], (-1, 4 * self.marketFeature)))
        return tf.reshape(tf.concat([output1, output2, output3, output4, output5, output6], axis=1), shape=(-1, 6, 1))


##==========================End hourly Market Based Feature Extraction===================##

##==========================BERT_BOEC_Model ABM_BCSIM===================##
class AngleFusionRegression(keras.layers.Layer):
    def __init__(self, marketD, marketL, newsD, newsL, marketD_low, marketL_low):
        super(AngleFusionRegression, self).__init__()
        self.newsEmbeddingDim = newsL
        self.marketD = marketD
        self.marketL = marketL
        self.newsD = newsD
        self.newsL = newsL
        self.marketD_low = marketD_low
        self.marketL_low = marketL_low
        self.normalizationLayer1 = DainNormalization(marketD, marketL)
        self.normalizationLayer2 = DainNormalization(marketD_low, marketL_low)

        self.newsContextVectorLayer = newsContextVectorLayer(newsD, newsL)
        self.highResolutionMarket = highResolutionMarket(marketD, marketL)

        self.newsAttentionRnn = AttentionBasedRNN(newsD, newsL, 'newsRNN')
        self.highResolutionMarketRnn = AttentionBasedRNN(marketD, marketL, 'highResMarketRNN')
        self.lowResolutionMarketRnn = AttentionBasedRNN(marketD_low, marketL_low, 'lowResMarketRNN')

        self.decisionLayer = Dense(units=1, activation='linear')
        self.denormalizationLayer = Denormalization()
        # self.FusionLayer = FusionLayer()

    def get_config(self):
        output = {'market_Delay': self.marketD,
                  'market_features': self.marketL,
                  'market_delay_lowReslutiuon': self.marketD_low,
                  'market_feature_lowReslutiuon': self.marketL_low,
                  'news_Delay': self.newsD,
                  'news_features': self.newsL}
        return output

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, marketInputs, newsInputs, market_lowInputs):
        # input : batchSize, 210, 15* 6 = 210*90
        temporalNewsFeature, newsAttention1 = self.newsContextVectorLayer(newsInputs)
        # output: batchSize, 6 ,embeddingDim

        # input : batchSize,25,4
        normalizedMarketInput, avg, std = self.normalizationLayer1(marketInputs)
        temporalMarketHighFeature = self.highResolutionMarket(normalizedMarketInput)
        # output: batchSize,6 ,1

        # input: batchSize, 6 ,embeddingDim
        newsFeature, newsAttention2 = self.newsAttentionRnn(temporalNewsFeature)
        # output

        marketHighResolutionFeature, marketHighResolutionAttention = self.highResolutionMarketRnn(
            temporalMarketHighFeature)

        normalizedmarketLowResolutionFeature, avg2, std2 = self.normalizationLayer2(market_lowInputs)
        marketLowResolutionFeature, marketLowResolutionAttention = self.lowResolutionMarketRnn(
            normalizedmarketLowResolutionFeature)
        output = tf.concat([newsFeature, marketHighResolutionFeature, marketLowResolutionFeature], axis=1)

        output = self.decisionLayer(output)

        output = self.denormalizationLayer(output, avg2, std2)
        return output

class ablation1AngleFusionRegression(keras.layers.Layer):
    def __init__(self, marketD, marketL, newsD, newsL,  ):
        super(ablation1AngleFusionRegression, self).__init__()
        self.newsEmbeddingDim = newsL
        self.marketD = marketD
        self.marketL = marketL
        self.newsD = newsD
        self.newsL = newsL

        self.normalizationLayer2 = DainNormalization(marketD, marketL)

        self.newsContextVectorLayer = newsContextVectorLayer(newsD, newsL)


        self.newsAttentionRnn = AttentionBasedRNN(newsD, newsL, 'newsRNN')
        self.highResolutionMarketRnn = AttentionBasedRNN(marketD, marketL, 'highResMarketRNN')

        self.decisionLayer = Dense(units=1, activation='linear')
        self.denormalizationLayer = Denormalization()
        # self.FusionLayer = FusionLayer()

    def get_config(self):
        output = {'market_Delay': self.marketD,
                  'market_features': self.marketL,
                  'news_Delay': self.newsD,
                  'news_features': self.newsL}
        return output

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, marketInputs, newsInputs ):
        # input : batchSize, 210, 15* 6 = 210*90
        temporalNewsFeature, newsAttention1 = self.newsContextVectorLayer(newsInputs)
        # output: batchSize, 6 ,embeddingDim

        newsFeature, newsAttention2 = self.newsAttentionRnn(temporalNewsFeature)
        # output
        normalizedmarketLowResolutionFeature, avg2, std2 = self.normalizationLayer2(marketInputs)
        marketLowResolutionFeature, marketLowResolutionAttention = self.highResolutionMarketRnn(
            normalizedmarketLowResolutionFeature)
        output = tf.concat([newsFeature,   marketLowResolutionFeature], axis=1)
        output = self.decisionLayer(output)
        output = self.denormalizationLayer(output, avg2, std2)
        return output

class AngleFusionClassification(keras.layers.Layer):
    def __init__(self, marketD, marketL, newsD, newsL, marketD_low, marketL_low):
        super(AngleFusionClassification, self).__init__()
        self.newsEmbeddingDim = newsL
        self.marketD = marketD
        self.marketL = marketL
        self.newsD = newsD
        self.newsL = newsL
        self.marketD_low = marketD_low
        self.marketL_low = marketL_low
        self.normalizationLayer1 = DainNormalization(marketD, marketL)
        self.normalizationLayer2 = DainNormalization(marketD_low, marketL_low)

        self.newsContextVectorLayer = newsContextVectorLayer(newsD, newsL)
        self.highResolutionMarket = highResolutionMarket(marketD, marketL)

        self.newsAttentionRnn = AttentionBasedRNN(newsD, newsL, 'newsRNN')
        self.highResolutionMarketRnn = AttentionBasedRNN(marketD, marketL, 'highResMarketRNN')
        self.lowResolutionMarketRnn = AttentionBasedRNN(marketD_low, marketL_low, 'lowResMarketRNN')

        self.decisionLayer = Dense(units=1, activation='sigmoid')
        # self.denormalizationLayer = Denormalization()
        # self.denormalizationLayer = Denormalization()
        # self.FusionLayer = FusionLayer()

    def get_config(self):
        output = {'market_Delay': self.marketD,
                  'market_features': self.marketL,
                  'market_delay_lowReslutiuon': self.marketD_low,
                  'market_feature_lowReslutiuon': self.marketL_low,
                  'news_Delay': self.newsD,
                  'news_features': self.newsL}
        return output

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, marketInputs, newsInputs, market_lowInputs):
        # input : batchSize, 210, 15* 6 = 210*90
        temporalNewsFeature, newsAttention1 = self.newsContextVectorLayer(newsInputs)
        # output: batchSize, 6 ,embeddingDim

        # input : batchSize,25,4
        normalizedMarketInput, avg, std = self.normalizationLayer1(marketInputs)
        temporalMarketHighFeature = self.highResolutionMarket(normalizedMarketInput)
        # output: batchSize,6 ,1

        # input: batchSize, 6 ,embeddingDim
        newsFeature, newsAttention2 = self.newsAttentionRnn(temporalNewsFeature)
        # output

        marketHighResolutionFeature, marketHighResolutionAttention = self.highResolutionMarketRnn(
            temporalMarketHighFeature)

        normalizedmarketLowResolutionFeature, avg2, std2 = self.normalizationLayer2(market_lowInputs)
        marketLowResolutionFeature, marketLowResolutionAttention = self.lowResolutionMarketRnn(
            normalizedmarketLowResolutionFeature)
        output = tf.concat([newsFeature, marketHighResolutionFeature, marketLowResolutionFeature], axis=1)

        output = self.decisionLayer(output)

        # output = self.denormalizationLayer(output, avg2, std2)
        return output


class baseline2Predictive_model(keras.layers.Layer):
    def __init__(self, marketD, marketL, newsD, newsL):
        super(baseline2Predictive_model, self).__init__()
        self.normalizationLayer = DainNormalization(marketD, marketL)
        self.conv1_1 = keras.layers.Conv1D(filters=300, kernel_size=3, activation="relu")
        self.conv1_2 = keras.layers.Conv1D(filters=10, kernel_size=3, activation="relu")
        self.flattenLayer = Flatten()
        self.dense1 = Dense(units=64, activation="relu")
        self.decisionLayer = Dense(units=1, activation="linear")
        self.denormLayer = Denormalization()

    def call(self, marketInputs, newsInputs):
        normalizedMarketData, avg, std = self.normalizationLayer(marketInputs)
        output1 = self.conv1_1(normalizedMarketData)
        output2 = self.conv1_2(newsInputs)

        # Flatten the convolutional outputs
        flattened1 = self.flattenLayer(output1)
        flattened2 = self.flattenLayer(output2)

        # Concatenate the flattened features
        merged = keras.layers.concatenate([flattened1, flattened2])
        self.unit = 128
        # Dense layers (replace units as needed)
        dense1 = self.dense1(merged)
        output = self.decisionLayer(dense1)  # Adjust units for your output
        output = self.denormLayer(output, avg, std)
        return output


class baseline3Predictive_model(keras.layers.Layer):
    def __init__(self, marketD, marketL, newsD, newsL):
        super(baseline3Predictive_model, self).__init__()
        self.unit = 128
        self.normalizationLayer = DainNormalization(marketD, marketL)
        self.RNN_1 = keras.layers.LSTM(self.unit,
                                       kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                       bias_regularizer=regularizers.L2(1e-4),
                                       activity_regularizer=regularizers.L2(
                                           1e-5), )  # return_sequences=True, return_state=True
        self.RNN_2 = keras.layers.LSTM(self.unit,
                                       kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                       bias_regularizer=regularizers.L2(1e-4),
                                       activity_regularizer=regularizers.L2(1e-5))

        self.dense1 = Dense(units=64, activation="relu")
        self.decisionLayer = Dense(units=1, activation="linear")
        self.denormLayer = Denormalization()

    def call(self, marketInputs, newsInputs):
        normalizedMarketData, avg, std = self.normalizationLayer(marketInputs)
        output1 = self.RNN_1(normalizedMarketData)
        output2 = self.RNN_2(newsInputs)


        # Concatenate the flattened features
        merged = keras.layers.concatenate([output1, output2])

        # Dense layers (replace units as needed)
        dense1 = self.dense1(merged)
        output = self.decisionLayer(dense1)  # Adjust units for your output
        output = self.denormLayer(output, avg, std)
        return output

class ablation2AngleFusionRegression(keras.layers.Layer):
    def __init__(self, marketD, marketL,   marketD_low, marketL_low):
        super(ablation2AngleFusionRegression, self).__init__()
        self.marketD = marketD
        self.marketL = marketL

        self.marketD_low = marketD_low
        self.marketL_low = marketL_low
        self.normalizationLayer1 = DainNormalization(marketD, marketL)
        self.normalizationLayer2 = DainNormalization(marketD_low, marketL_low)


        self.highResolutionMarket = highResolutionMarket(marketD, marketL)

        self.highResolutionMarketRnn = AttentionBasedRNN(marketD, marketL, 'highResMarketRNN')
        self.lowResolutionMarketRnn = AttentionBasedRNN(marketD_low, marketL_low, 'lowResMarketRNN')

        self.decisionLayer = Dense(units=1, activation='linear')
        self.denormalizationLayer = Denormalization()
        # self.FusionLayer = FusionLayer()

    def get_config(self):
        output = {'market_Delay': self.marketD,
                  'market_features': self.marketL,
                  'market_delay_lowReslutiuon': self.marketD_low,
                  'market_feature_lowReslutiuon': self.marketL_low,
                 }
        return output

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, marketInputs,   market_lowInputs):
        # input : batchSize, 210, 15* 6 = 210*90
        # output: batchSize, 6 ,embeddingDim

        # input : batchSize,25,4
        normalizedMarketInput, avg, std = self.normalizationLayer1(marketInputs)
        temporalMarketHighFeature = self.highResolutionMarket(normalizedMarketInput)
        # output: batchSize,6 ,1

        # input: batchSize, 6 ,embeddingDim
         # output

        marketHighResolutionFeature, marketHighResolutionAttention = self.highResolutionMarketRnn(
            temporalMarketHighFeature)

        normalizedmarketLowResolutionFeature, avg2, std2 = self.normalizationLayer2(market_lowInputs)
        marketLowResolutionFeature, marketLowResolutionAttention = self.lowResolutionMarketRnn(
            normalizedmarketLowResolutionFeature)
        output = tf.concat([ marketHighResolutionFeature, marketLowResolutionFeature], axis=1)

        output = self.decisionLayer(output)

        output = self.denormalizationLayer(output, avg2, std2)
        return output

class predictiveModel:
    def __init__(self, ):
        return

    @staticmethod
    def modelTrain(train_x, train_news_x, train_market_lowRes_x, train_y
                   , validation_x, validation_news_x, validation_market_highRes_x, validation_y, problem='reg'):
        marketD = train_x.shape[1]
        marketL = train_x.shape[2]
        newsD = train_news_x.shape[1]
        newsL = train_news_x.shape[2]
        marketD_low = train_market_lowRes_x.shape[1]
        marketL_low = train_market_lowRes_x.shape[2]
        input1 = Input(shape=(marketD, marketL))
        input2 = Input(shape=(newsD, newsL))
        input3 = Input(shape=(marketD_low, marketL_low))

        if problem.strip().lower() == 'reg':
            output = AngleFusionRegression(marketD, marketL, newsD, newsL,
                                           marketD_low, marketL_low)(input1, input2, input3)

            model = Model(inputs=[input1, input2, input3], outputs=output)
            model.compile(loss=MeanAbsolutePercentageError(), optimizer=Adam(lr=0.001, decay=1e-6),
                          metrics=[tf.keras.metrics.MeanSquaredError(),  tf.keras.metrics.MeanAbsolutePercentageError()])
            model.summary()
            # tf.keras.utils.plot_model(model,show_shapes=True)
            # callback = EarlyStopping(monitor='val_loss', patience=30 , min_delta=0.00001,)

            history = model.fit([train_x, train_news_x, train_market_lowRes_x], train_y,
                                validation_data=(
                                    [validation_x, validation_news_x, validation_market_highRes_x], validation_y),
                                epochs=300, batch_size=32)  # callbacks=[callback])
            # model.save(model_name)
            return history, model
        elif problem.lower().strip() == 'classifier':
            output = AngleFusionClassification(marketD, marketL, newsD, newsL,
                                               marketD_low, marketL_low)(input1, input2, input3)

            model = Model(inputs=[input1, input2, input3], outputs=output)
            model.compile(loss=BinaryCrossentropy(), optimizer=Adam(lr=0.001, decay=1e-6),
                          metrics=[Accuracy()])
            model.summary()
            # tf.keras.utils.plot_model(model,show_shapes=True)
            # callback = EarlyStopping(monitor='val_loss', patience=30 , min_delta=0.00001,)

            history = model.fit([train_x, train_news_x, train_market_lowRes_x], train_y,
                                validation_data=(
                                    [validation_x, validation_news_x, validation_market_highRes_x], validation_y),
                                epochs=100, batch_size=32)  # callbacks=[callback])
            # model.save(model_name)
            return history, model
        else:
            return None, None

    @staticmethod
    def baselineModelTrain(train_x, train_news_x, train_y
                           , validation_x, validation_news_x, validation_y, problem='baseline2'):
        marketD = train_x.shape[1]
        marketL = train_x.shape[2]
        newsD = train_news_x.shape[1]
        newsL = train_news_x.shape[2]
        input1 = Input(shape=(marketD, marketL))
        input2 = Input(shape=(newsD, newsL))
        if problem.lower().strip() == 'baseline2':
            output = baseline2Predictive_model(marketD, marketL, newsD, newsL)(input1, input2)

            model = Model(inputs=[input1, input2], outputs=output)
            model.compile(loss=MeanAbsolutePercentageError(), optimizer=Adam(lr=0.001, decay=1e-6),
                          metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsolutePercentageError()])
            model.summary()
            # tf.keras.utils.plot_model(model,show_shapes=True)
            # callback = EarlyStopping(monitor='val_loss', patience=30 , min_delta=0.00001,)

            history = model.fit([train_x, train_news_x], train_y,
                                validation_data=(
                                    [validation_x, validation_news_x], validation_y),
                                epochs=300, batch_size=32)  # callbacks=[callback])
            # model.save(model_name)
            return history, model
        if problem.lower().strip() == 'baseline3':
            output = baseline3Predictive_model(marketD, marketL, newsD, newsL)(input1, input2)

            model = Model(inputs=[input1, input2], outputs=output)
            model.compile(loss=MeanAbsolutePercentageError(), optimizer=Adam(lr=0.001, decay=1e-6),
                          metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsolutePercentageError()])
            model.summary()
            # tf.keras.utils.plot_model(model,show_shapes=True)
            # callback = EarlyStopping(monitor='val_loss', patience=30 , min_delta=0.00001,)

            history = model.fit([train_x, train_news_x], train_y,
                                validation_data=(
                                    [validation_x, validation_news_x], validation_y),
                                epochs=300, batch_size=32)  # callbacks=[callback])
            # model.save(model_name)
            return history, model

    @staticmethod
    def ablationWithoutHourlyModelTrain(train_x, train_news_x, train_y
                           , validation_x, validation_news_x, validation_y, problem='ablation1'):
        marketD = train_x.shape[1]
        marketL = train_x.shape[2]
        newsD = train_news_x.shape[1]
        newsL = train_news_x.shape[2]
        input1 = Input(shape=(marketD, marketL))
        input2 = Input(shape=(newsD, newsL))
        if problem.lower().strip() == 'ablation1':
            output = ablation1AngleFusionRegression(marketD, marketL, newsD, newsL)(input1, input2)

            model = Model(inputs=[input1, input2], outputs=output)
            model.compile(loss=MeanAbsolutePercentageError(), optimizer=Adam(lr=0.001, decay=1e-6),
                          metrics=[tf.keras.metrics.MeanSquaredError(),
                                   tf.keras.metrics.MeanAbsolutePercentageError()])
            model.summary()
            # tf.keras.utils.plot_model(model,show_shapes=True)
            # callback = EarlyStopping(monitor='val_loss', patience=30 , min_delta=0.00001,)

            history = model.fit([train_x, train_news_x], train_y,
                                validation_data=(
                                    [validation_x, validation_news_x], validation_y),
                                epochs=300 , batch_size=32)  # callbacks=[callback])
            # model.save(model_name)
            return history, model

        return None, None
    @staticmethod
    def ablationWithoutNews(train_x,   train_market_lowRes_x, train_y
                   , validation_x,   validation_market_highRes_x, validation_y, problem='ablation3'):
        marketD = train_x.shape[1]
        marketL = train_x.shape[2]

        marketD_low = train_market_lowRes_x.shape[1]
        marketL_low = train_market_lowRes_x.shape[2]
        input1 = Input(shape=(marketD, marketL))

        input2 = Input(shape=(marketD_low, marketL_low))

        if problem.strip().lower() == 'ablation3':
            output = ablation2AngleFusionRegression(marketD, marketL,
                                           marketD_low, marketL_low)(input1,  input2)

            model = Model(inputs=[input1,  input2], outputs=output)
            model.compile(loss=MeanAbsolutePercentageError(), optimizer=Adam(lr=0.001, decay=1e-6),
                          metrics=[tf.keras.metrics.MeanSquaredError(),  tf.keras.metrics.MeanAbsolutePercentageError()])
            model.summary()
            # tf.keras.utils.plot_model(model,show_shapes=True)
            # callback = EarlyStopping(monitor='val_loss', patience=30 , min_delta=0.00001,)

            history = model.fit([train_x,   train_market_lowRes_x], train_y,
                                validation_data=(
                                    [validation_x,   validation_market_highRes_x], validation_y),
                                epochs=300, batch_size=32)  # callbacks=[callback])
            # model.save(model_name)
            return history, model
        return None, None


##==========================End BERT_BOEC_Model ABM_BCSIM===================##
