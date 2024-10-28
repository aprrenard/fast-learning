import numpy as np
import matplotlib.pyplot as plt

class ReinforcementLearningModel:
    def __init__(self, learning_rate, multiplicative_factor, noise_std, initial_weights):
        self.learning_rate = learning_rate
        self.multiplicative_factor = multiplicative_factor
        self.noise_std = noise_std
        self.weights = np.array(initial_weights)

    def decision(self, stimulus):
        """Makes a decision based on current weights and stimulus input."""
        activation = np.dot(self.weights, stimulus) + np.random.normal(0, self.noise_std)
        return 1 if activation > 0 else 0

    def multiplicative_update(self, stimulus, reward_signal):
        """Updates weights using a multiplicative learning rule."""
        error = reward_signal - np.dot(self.weights, stimulus)
        learning_step = self.multiplicative_factor * self.learning_rate * self.weights
        self.weights += learning_step * error * stimulus

    def run_trial(self, stimulus, reward_signal):
        """Runs a single trial, returning decision and updating weights."""
        lick = self.decision(stimulus)
        self.multiplicative_update(stimulus, reward_signal)
        return lick

class PretrainingPhase:
    def __init__(self, model, auditory_stimulus, N_pretrain_trials, reward_signal):
        self.model = model
        self.auditory_stimulus = auditory_stimulus
        self.N_pretrain_trials = N_pretrain_trials
        self.reward_signal = reward_signal

    def run_pretraining(self):
        """Simulates the auditory pretraining phase to modify initial weights."""
        for _ in range(self.N_pretrain_trials):
            self.model.run_trial(self.auditory_stimulus, self.reward_signal)
        return self.model.weights  # Return the updated weights after pretraining

class WhiskerTaskPhase:
    def __init__(self, model, whisker_stimulus, blank_stimulus, N_trials, reward_signal, no_reward_signal):
        self.model = model
        self.whisker_stimulus = whisker_stimulus
        self.blank_stimulus = blank_stimulus
        self.N_trials = N_trials
        self.reward_signal = reward_signal
        self.no_reward_signal = no_reward_signal
        self.performance_whisker = np.zeros(N_trials)
        self.performance_blank = np.zeros(N_trials)
        self.weights_track = np.zeros((N_trials, 2))

    def run_task(self):
        """Simulates the whisker detection task after pretraining."""
        for trial in range(self.N_trials):
            if trial % 2 == 0:  # Whisker stimulus trial
                stimulus = self.whisker_stimulus
                target_reward = self.reward_signal
            else:  # Blank stimulus trial
                stimulus = self.blank_stimulus
                target_reward = self.no_reward_signal

            # Run trial and get the decision
            lick = self.model.run_trial(stimulus, target_reward)
            
            # Track performance
            if np.array_equal(stimulus, self.whisker_stimulus):
                self.performance_whisker[trial] = 1 if lick == target_reward else 0
            else:
                self.performance_blank[trial] = 1 if lick == target_reward else 0
            
            # Track weight changes
            self.weights_track[trial] = self.model.weights

        return self.performance_whisker, self.performance_blank, self.weights_track

# Parameters
learning_rate = 0.005
multiplicative_factor = 1.2
noise_std = 0.1
initial_weights = [0.01, 0.01]  # Initial synaptic weights
N_pretrain_trials = 100  # Number of pretraining trials
N_whisker_trials = 500  # Number of whisker task trials
reward_signal = 1  # Reward signal for lick decision
no_reward_signal = -1  # No reward signal for blank trials

# Stimuli
S_auditory = np.array([1, 0])  # Auditory stimulus for pretraining
S_whisker = np.array([1, 0])  # Whisker stimulation for task
S_blank = np.array([0, 1])  # Blank trial for task

# Initialize model
model = ReinforcementLearningModel(learning_rate, multiplicative_factor, noise_std, initial_weights)

# Pretraining phase on auditory stimulus
pretraining = PretrainingPhase(model, S_auditory, N_pretrain_trials, reward_signal)
pretrained_weights = pretraining.run_pretraining()

print(f"Weights after pretraining: {pretrained_weights}")

# Run whisker detection task after pretraining
whisker_task = WhiskerTaskPhase(model, S_whisker, S_blank, N_whisker_trials, reward_signal, no_reward_signal)
performance_whisker, performance_blank, weights_track = whisker_task.run_task()

# Plot results
plt.figure(figsize=(12, 5))

# Plot performance over trials
plt.subplot(1, 2, 1)
plt.plot(np.convolve(performance_whisker, np.ones(20)/20, mode='valid'), label='Whisker Stimulus')
plt.plot(np.convolve(performance_blank, np.ones(20)/20, mode='valid'), label='Blank Stimulus')
plt.title('Learning Performance Over Trials')
plt.xlabel('Trial')
plt.ylabel('Correct Decisions (%)')
plt.legend()

# Plot weight changes over trials
plt.subplot(1, 2, 2)
plt.plot(weights_track[:, 0], label='Weight: Whisker Stimulus')
plt.plot(weights_track[:, 1], label='Weight: Blank Stimulus')
plt.title('Weight Updates Over Trials')
plt.xlabel('Trial')
plt.ylabel('Synaptic Weight')
plt.legend()

plt.tight_layout()
plt.show()