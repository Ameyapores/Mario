from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Lambda, Concatenate
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import LSTM
from keras import backend as K

def build_network(input_shape, output_shape):
    state = Input(shape=input_shape)
    h = Conv2D(32, (3, 3) , strides=(2, 2), padding='same', data_format='channels_first')(state)
    h = ELU(alpha=1.0)(h)
    h = Conv2D(32, (3, 3) , strides=(2, 2), padding='same', data_format='channels_first')(h)
    h = ELU(alpha=1.0)(h)
    h = Conv2D(32, (3, 3) , strides=(2, 2), padding='same', data_format='channels_first')(h)
    h = ELU(alpha=1.0)(h)
    h = Conv2D(32, (3, 3) , strides=(2, 2), padding='same', data_format='channels_first')(h)
    h = ELU(alpha=1.0)(h)
    h = Flatten()(h)
    
    value = Dense(256, activation='relu')(h)
    value = Dense(1, activation='linear', name='value')(value)
    #policy = LSTM(output_shape, activation='sigmoid', name='policy')(h)
    policy = Dense(output_shape, activation='sigmoid', name='policy')(h)
    
    value_network = Model(input=state, output=value)
    policy_network = Model(input=state, output=policy)

    advantage = Input(shape=(1,))
    train_network = Model(input=[state, advantage], output=[value, policy])

    return value_network, policy_network, train_network, advantage

def build_feature_map(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3) , strides=(2, 2), padding='same', data_format='channels_first'))
    model.add(ELU(alpha=1.0))
    model.add(Conv2D(32, (3, 3) , strides=(2, 2), padding='same', data_format='channels_first'))
    model.add(ELU(alpha=1.0))
    model.add(Conv2D(32, (3, 3) , strides=(2, 2), padding='same', data_format='channels_first'))
    model.add(ELU(alpha=1.0))
    model.add(Conv2D(32, (3, 3) , strides=(2, 2), padding='same', data_format='channels_first'))
    model.add(ELU(alpha=1.0))
    model.add(Flatten(name="feature"))
    model.name = "FM"
    
    return model

def inverse_model(output_dim=6):
    """
    s_t, s_t+1 -> a_t
    """
    def func(ft0, ft1):
        h = Concatenate(axis=1)([ft0, ft1])
        h = Dense(256, activation='relu', name="im1")(h)
        h = Dense(output_dim, activation='sigmoid', name = "im2")(h)
        return h
    return func

def forward_model(output_dim=288):
    """
    s_t, a_t -> s_t+1
    """
    def func(ft, at):
        h = Concatenate(axis=-1)([ft, at])
        h = Dense(256, activation='relu', name="fm1")(h)
        h = Dense(output_dim, activation='linear', name="fm2")(h)
        return h
    return func

def build_icm_model(observation_shape, action_shape, lmd=1.0, beta=0.01):
    s_t0 = Input(shape=observation_shape, name="state0")
    print (s_t0)
    s_t1 = Input(shape=observation_shape, name="state1")
    a_t = Input(shape=action_shape, name="action")
    
    
    reshape = Reshape(target_shape=(1,) + observation_shape)
    fmap = build_feature_map((1,) + observation_shape)
    #print(fmap.__dict__)
    f_t0 = fmap(reshape(s_t0))
    f_t1 = fmap(reshape(s_t1))
    act_hat = inverse_model(action_shape[0])(f_t0, f_t1)
    f_t1_hat = forward_model(output_dim = fmap.outputs[0]._keras_shape[1])(f_t0, a_t)
    r_in = Lambda(lambda x: 0.5 * K.sum(K.square(x[0] - x[1]), axis=-1), output_shape=(1,), name="reward_intrinsic")([f_t1, f_t1_hat])
    l_i = Lambda(lambda x: -K.sum(x[0] * K.log(x[1] + K.epsilon()), axis=-1), output_shape=(1,), name = "Li")([a_t, act_hat])
    loss0 = Lambda(lambda x: beta * x[0] + (1.0 - beta) * x[1], output_shape=(1,), name = "loss0")([r_in, l_i])
    rwd = Input(shape=(1,), name = "reward")
    loss = Lambda(lambda x: (-lmd * x[0] + x[1]), output_shape=(1,), name="loss")([rwd, loss0])
    return Model([s_t0, s_t1, a_t, rwd], loss)

def get_reward_intrinsic(model, x):
    return K.function([model.get_layer("state0").input,
                       model.get_layer("state1").input,
                       model.get_layer("action").input],
                      [model.get_layer("reward_intrinsic").output])(x)[0]

if __name__ == "__main__":
    from setup_env import setup_env
    env= setup_env('SuperMarioBros-v0')
    icm= build_icm_model((env.observation_space.shape[:2]), (env.action_space.n,))
    icm.summary()
