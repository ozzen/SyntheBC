import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

num_states = 10
print("Size of State Space ->  {}".format(num_states))
num_actions = 1
print("Size of Action Space ->  {}".format(num_actions))

# rtds constraints
high_d = 0.69
low_d = 0.68
high_q = 1e-5
low_q = -1e-5
high_md = 0.39
low_md = 0.38
high_mq = 0.055
low_mq = 0.052
high_v_oq = 0.008
low_v_oq = 0.007
v_ref = 0.48
v_init = 0.01*v_ref
v_fsc = 0.03*v_ref

w = 100
n = 1

# action value bounds
upper_bound = 1
lower_bound = -1

# contain values within a range
def bound(n, minn, maxn):
    return max(min(maxn, n), minn)

# generates initial configuration
def D1(flag):
    i_d = np.random.uniform(low=low_d, high=high_d)
    i_q = np.random.uniform(low=low_q, high=high_q)
    i_od = np.random.uniform(low=low_d, high=high_d)
    i_oq = np.random.uniform(low=low_q, high=high_q)
    if flag % 2 == 0:
        v_od = np.random.uniform(low=v_ref-v_fsc-v_init, high=v_ref-v_fsc)
    else:
        v_od = np.random.uniform(low=v_ref+v_fsc, high=v_ref+v_fsc+v_init)
    v_oq = np.random.uniform(low=low_v_oq, high=high_v_oq)
    i_ld = np.random.uniform(low=low_d, high=high_d)
    i_lq = np.random.uniform(low=0, high=0)
    m_d = np.random.uniform(low=low_md, high=high_md)
    m_q = np.random.uniform(low=low_mq, high=high_mq)

    state = np.array([i_d, i_q, i_od, i_oq, v_od, v_oq, i_ld, i_lq, m_d, m_q])
    dataset = 1
    return state, dataset

def D2():
    i_d = np.random.uniform(low=low_d, high=high_d)
    i_q = np.random.uniform(low=low_q, high=high_q)
    i_od = np.random.uniform(low=low_d, high=high_d)
    i_oq = np.random.uniform(low=low_q, high=high_q)
    v_od = np.random.uniform(low=v_ref-v_init, high=v_ref+v_init)
    v_oq = np.random.uniform(low=low_v_oq, high=high_v_oq)
    i_ld = np.random.uniform(low=low_d, high=high_d)
    i_lq = np.random.uniform(low=0, high=0)
    m_d = np.random.uniform(low=low_md, high=high_md)
    m_q = np.random.uniform(low=low_mq, high=high_mq)

    state = np.array([i_d, i_q, i_od, i_oq, v_od, v_oq, i_ld, i_lq, m_d, m_q])
    dataset = 2
    return state, dataset

def D3():
    i_d = np.random.uniform(low=low_d, high=high_d)
    i_q = np.random.uniform(low=low_q, high=high_q)
    i_od = np.random.uniform(low=low_d, high=high_d)
    i_oq = np.random.uniform(low=low_q, high=high_q)
    v_od = np.random.uniform(low=v_ref-v_fsc, high=v_ref+v_fsc)
    v_oq = np.random.uniform(low=low_v_oq, high=high_v_oq)
    i_ld = np.random.uniform(low=low_d, high=high_d)
    i_lq = np.random.uniform(low=0, high=0)
    m_d = np.random.uniform(low=low_md, high=high_md)
    m_q = np.random.uniform(low=low_mq, high=high_mq)

    state = np.array([i_d, i_q, i_od, i_oq, v_od, v_oq, i_ld, i_lq, m_d, m_q])
    dataset = 3
    return state, dataset

def plant_dyn(state):
    p_star = 1
    q_star = 1e-6
    w = 60
    r_n = 1e3
    r_c = 0.0384
    c_f = 2500
    r_f = 2e-3
    l_f = 100e-6
    i_ref_d = 2.08
    i_ref_q = 1e-6
    kp = 0.5
    v_bd = 0.48
    v_bq = 1e-6

    x1 = float(state[0])
    x2 = float(state[1])
    x3 = float(state[2])
    x4 = float(state[3])
    x5 = float(state[4])
    x6 = float(state[5])
    x7 = float(state[6])
    x8 = float(state[7])
    x9 = float(state[8])
    x10 = float(state[9])

    X1 = -p_star * x1 + w * x2 + v_bd
    X2 = -q_star * x2 - w * x1 + v_bq
    X3 = -r_c * x3 + w * x4 + x5 - v_bd
    X4 = -r_c * x4 - w * x3 + x6 - v_bq
    X5 = w * x6 + (x7 - x3) / c_f
    X6 = -w * x5 + (x8 - x4) / c_f
    X7 = -(r_f / l_f) * x7 + w * x8 + (x9 - x5) / l_f
    X8 = -(r_f / l_f) * x8 - w * x7 + (x10 - x6) / l_f
    X9 = -w * x8 + kp * (i_ref_d - x7)
    X10 = -w * x7 + kp * (i_ref_q - x8)

    X = X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10
    return X

def get_reward(action, state, dataset):
    if dataset == 1:
        e = 0.0001
        reward = min(action-e, 0)
    if dataset == 2:
        reward = -max(action, 0)
    if dataset == 3:
        if action <= 0:
            reward = 0
        else:
            fx = plant_dyn(state)
            reward = -max(fx, 0)
    return reward

"""
To implement better exploration by the Actor network, we use noisy perturbations,
specifically an **Ornstein-Uhlenbeck process** for generating noise, as described 
in the paper. It samples noise from a correlated normal distribution.
"""

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

"""
**Critic loss** - Mean Squared Error of `y - Q(s, a)`
where `y` is the expected return as seen by the Target network,
and `Q(s, a)` is action value predicted by the Critic network. `y` is a moving target
that the critic model tries to achieve; we make this target
stable by updating the Target model slowly.

**Actor loss** - This is computed using the mean of the value given by the Critic network
for the actions taken by the Actor network. We seek to maximize this quantity.

Hence we update the Actor network so that it produces actions that get
the maximum predicted value as seen by the Critic, for a given state.
"""

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=128):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

"""
Here we define the Actor and Critic networks. These are basic Dense models
with `ReLU` activation.

Note: We need the initialization for last layer of the Actor to be between
`-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
the initial stages, which would squash our gradients to zero,
as we use the `tanh` activation.
"""

def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(50, activation="relu")(inputs)
    out = layers.Dense(50, activation="relu")(out)
    outputs = layers.Dense(1, activation=None, kernel_initializer=last_init)(out)

    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(50, activation="relu")(state_input)
    state_out = layers.Dense(50, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(50, activation="relu")(action_input)

    # Both are passed through separate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(50, activation="relu")(concat)
    out = layers.Dense(50, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)
    return model

"""
`policy()` returns an action sampled from our Actor network plus some noise for exploration.
"""

def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise
    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return [np.squeeze(legal_action)]

"""
## Training hyperparameters
"""

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
actor_lr = 0.0001
critic_lr = 0.0001

actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

actor_model.compile(optimizer=actor_optimizer)
critic_model.compile(optimizer=critic_optimizer)
target_actor.compile(optimizer=actor_optimizer)
target_critic.compile(optimizer=critic_optimizer)

total_episodes = 2000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.001

buffer = Buffer(50000, 128)

"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""

ep_reward_list = []
avg_reward_list = []
dataset3 = []

for ep in range(total_episodes):
    iter = 0
    count = 0
    episodic_reward = 0

    if (ep+1) % 200 == 0 and iter < 10000:
        state, dataset = D3()
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = actor_model.predict(tf_state)
        u = float(action[0])
        if u <= 0 and u >= -0.001:
            dataset3.append(state)
        iter = iter+1

    while count < 500:
        if count % 2 == 0:
            state, dataset = D1(total_episodes)
        else:
            state, dataset = D2()
        # Converting to tensor format
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        # Generating action from policy
        action = policy(tf_state, ou_noise)
        # Computing reward function
        u = float(action[0])
        reward = get_reward(u, state, dataset)
        # Generating next state
        next_state, next_dataset = D3()
        # Updating the replay buffer
        buffer.record((state, action, reward, next_state))
        # Aggregating the reward values
        episodic_reward += reward
        # Actor network learning using replay experience
        buffer.learn()
        # Updating taget networks
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)
        # Updating state information
        count = count + 1

    ep_reward_list.append(episodic_reward)
    avg_reward = np.mean(ep_reward_list[-500:])
    if ep % 50 == 0:
        print("Episode {} : Avg Reward ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

print(len(dataset3))

parent_dir = "/Users/admin/Desktop/SyntheBC"
result_dir = datetime.datetime.now().strftime('%m-%d-%Y_%H-%M')
result_path = os.path.join(parent_dir, result_dir)
os.mkdir(result_path)
data_dir = "data"
# plot_dir = "plots"
data_path = os.path.join(result_path,data_dir)
# plot_path = os.path.join(result_path,plot_dir)
os.mkdir(data_path)
# os.mkdir(plot_path)

# Collecting data
df1 = pd.DataFrame(avg_reward_list)
df1.to_csv(os.path.join(data_path,'Avg_Reward.csv'),index=False)
df2 = pd.DataFrame(ep_reward_list)
df2.to_csv(os.path.join(data_path,'Ep_Reward.csv'),index=False)

# Plotting graphs
# Episodes versus Average Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Reward")
plt.savefig(os.path.join(data_path,"Avg_Reward_plot.png"))
plt.show()

# Episodes versus Episodic Rewards
plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel("Ep. Reward")
plt.savefig(os.path.join(data_path,"Ep_Reward_plot.png"))
plt.show()

"""
If training proceeds correctly, the average episodic reward will increase with time.
"""

# Save the model
actor_model.save(os.path.join(result_path,"candidateBC.h5"))
actor_model.save_weights(os.path.join(result_path,"candidateBC_weights.h5"))
