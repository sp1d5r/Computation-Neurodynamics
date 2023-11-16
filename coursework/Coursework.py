from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt

class ModulularNetwork:
    def __init__(self, p, simulation_time, delta_t):
        self.p = p
        self.delta_t = delta_t
        self.simulation_time = simulation_time

        self.recordings_v = [[] for _ in range(1000)]  # using 1,000 neurones for this
        self.recordings_u = [[] for _ in range(1000)]

        # DEPRECATED - self.neurones = []
        # New Neurone Implementation
        self._max_neurone_potential = 30
        self.v_now = None
        self.u_now = None
        self.I_now = None
        self.firing = None
        self.a = None
        self.b = None
        self.c = None
        self.d = None

        print("Initialising Neurones")
        self.initialise_neurones()
        print("Neurones Initialised!")

        # DEPRECATED - self.synapses = []
        # New Synapse
        self.synaptic_weights = None
        self.scaling_factors = None
        self.conduction_delays = None
        self.conduction_queue = None
        self.pre_synaptic_neurones = None
        self.post_synaptic_neurones = None
        self.synaptic_efficacy = None
        self.adjacency_matrix = [[None for _ in range(1000)] for _ in range(1000)]
        self.decay_factor = 0.01

        print("Initialising Synapses")
        self.initialise_synapses()
        print("Synapses Initialised!")

    def initialize_neuron_parameters(self, num_excitatory, num_inhibitory):
        # Vectorized initialization of parameters
        r = np.random.random(num_excitatory + num_inhibitory)

        # a Parameter
        a_excitatory = 0.02 * np.ones(num_excitatory)
        a_inhibitory = 0.02 + 0.08 * np.square(r[num_excitatory:])
        a = np.concatenate([a_excitatory, a_inhibitory])

        # b parameter
        b_excitatory = 0.2 * np.ones(num_excitatory)
        b_inhibitory = 0.25 - 0.05 * r[num_excitatory:]
        b = np.concatenate([b_excitatory, b_inhibitory])

        # c parameter
        c_excitatory = -65 + 15 * np.square(r[:num_excitatory])
        c_inhibitory = -65 * np.ones(num_inhibitory)
        c = np.concatenate([c_excitatory, c_inhibitory])

        # d parameter
        d_excitatory = 8 - 6 * np.square(r[:num_excitatory])
        d_inhibitory = 2 * np.ones(num_inhibitory)
        d = np.concatenate([d_excitatory, d_inhibitory])

        return a, b, c, d

    def initialise_neurones(self):
        num_neurons = 1000
        num_excitatory = 800
        num_inhibitory = 200

        # Initialize neuron parameters
        self.v_now = np.random.uniform(-65, -60, num_neurons)
        self.u_now = -1 * np.ones(num_neurons)
        self.I_now = np.zeros(num_neurons)
        self.firing = np.zeros(num_neurons, dtype=bool)

        # Neuron type specific parameters
        a, b, c, d = self.initialize_neuron_parameters(num_excitatory, num_inhibitory)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        return None

    def update_neurons(self, external_currents, delta_t=0.01):
        # Number of updates within 1ms
        num_updates = int(1.0 / delta_t)

        # Update currents
        self.I_now += external_currents
        recordings_v = []
        recordings_u = []

        self.firing = np.array([False for _ in range(1000)])

        # Perform updates for each small timestep within 1ms
        for time_step in range(num_updates):
            not_fired = np.logical_not(self.firing)
            # Vectorized update
            dv_dt = 0.04 * self.v_now[not_fired] ** 2 + 5 * self.v_now[not_fired] + 140 - self.u_now[not_fired] + \
                    self.I_now[not_fired]
            du_dt = self.a[not_fired] * (self.b[not_fired] * self.v_now[not_fired] - self.u_now[not_fired])

            self.v_now[not_fired] += delta_t * dv_dt
            self.u_now[not_fired] += delta_t * du_dt

            # Check for firing neurons
            firing_neurons = self.v_now >= self._max_neurone_potential
            recordings_v.append(self.v_now.copy())
            recordings_u.append(self.u_now.copy())

            self.firing = np.logical_or(firing_neurons, self.firing)

        self.v_now[self.firing] = self.c[self.firing]
        self.u_now[self.firing] += self.d[self.firing]

        # Reset Currents
        self.I_now = np.zeros_like(self.I_now)

        # Extend the historical recordings with new data
        for neuron_id in range(1000):
            self.recordings_v[neuron_id].extend([v[neuron_id] for v in recordings_v])
            self.recordings_u[neuron_id].extend([u[neuron_id] for u in recordings_u])

        return self.v_now, self.u_now, self.firing

    def initialise_synapse_params(self, synapes):
        def get_synapse_params(pre_neurone_type, post_neurone_type):
            if pre_neurone_type == "EXCITATORY":
                if post_neurone_type == "INHIBITORY":
                    return random.random(), 50, 1
                else:
                    return 1, 17, random.randint(1, 20)
            else:
                if post_neurone_type == "INHIBITORY":
                    return -random.random(), 1, 1
                else:
                    return -random.random(), 2, 1

        weights = []
        scalings = []
        delays = []
        conduction_queue = []
        for index, synapse in enumerate(synapes):
            pre_synapse_type = "EXCITATORY" if synapse[0] < 800 else "INHIBITORY"
            post_synapse_type = "EXCITATORY" if synapse[1] < 800 else "INHIBITORY"
            weight, scaling, delay = get_synapse_params(pre_synapse_type, post_synapse_type)
            weights.append(weight)
            scalings.append(scaling)
            delays.append(delay)
            self.adjacency_matrix[synapse[0]][synapse[1]] = index
            conduction_queue.append(deque([0 for i in range(delay)], maxlen=delay))

        self.synaptic_weights = np.array(weights)
        self.scaling_factors = np.array(scalings)
        self.conduction_delays = np.array(delays)
        self.conduction_queue = conduction_queue

    def initialise_synapses(self):
        # To make this more efficient, let's define the connections first in terms of index -> index
        synapses = []

        for module_number in range(8):
            pre_synpatic_neurones = np.random.randint(0, 99, 1000) + (module_number * 100)
            for i in pre_synpatic_neurones:
                post_synaptic_neurone = random.randint(0, 99) + (module_number * 100)
                while i == post_synaptic_neurone:
                    post_synaptic_neurone = random.randint(0, 99) + (module_number * 100)
                synapses.append((i, post_synaptic_neurone))

        all_modules = set(range(0, 8))

        # Rewire these neurone connections
        for index, synapse in enumerate(synapses):
            if random.random() < self.p:
                [pre_neurone_id, post_neurone_id] = synapse
                other_modules = all_modules - {post_neurone_id // 100}
                new_module = random.choice(list(other_modules))
                new_post_neurone = new_module * 100 + random.randint(0, 99)
                synapses[index] = (pre_neurone_id, new_post_neurone)

        for i in range(800, 1000):
            for j in range(1000):
                if i == j:
                    continue
                synapses.append((i, j))

            module_number = random.randint(0, 7)
            selected_neurone_ids = [random.randint(module_number * 100, (module_number + 1) * 100) for _ in range(4)]
            for selected_pre_neurone in selected_neurone_ids:
                synapses.append((selected_pre_neurone, i))

        self.pre_synaptic_neurones = np.array([synapses[i][0] for i in range(len(synapses))])
        self.post_synaptic_neurones = np.array([synapses[i][1] for i in range(len(synapses))])
        self.synaptic_efficacy = np.zeros(len(synapses), dtype=np.float64)
        self.initialise_synapse_params(synapses)

    def update_synapse(self):
        # Apply decay to all synaptic efficacies
        self.synaptic_efficacy *= self.decay_factor

        # Handle the pre-synaptic neuron firing and update synaptic efficacy
        for i, queue in enumerate(self.conduction_queue):
            pre_neurone_id = self.pre_synaptic_neurones[i]
            post_neurone_id = self.post_synaptic_neurones[i]

            # Check if pre-synaptic neuron is firing
            if self.firing[pre_neurone_id]:
                queue.append(1)
            else:
                queue.append(0)

            # Check if a new spike has arrived (based on conduction delay)
            if queue.popleft():
                self.synaptic_efficacy[i] += self.synaptic_weights[i]

            # Apply the current synaptic efficacy to the post-synaptic neuron
            # Note: This will require accumulating the input from all synapses for each neuron
            self.I_now[post_neurone_id] += self.synaptic_efficacy[i] * self.scaling_factors[i]

    def run_simulation(self):
        for sim_time in range(self.simulation_time):
            # Update background current for neurons as a NumPy array
            external_currents = np.array([15 if np.random.poisson(0.01) > 0 else 0 for _ in range(1000)])
            # Update neurons and synapses
            self.update_neurons(external_currents)
            self.update_synapse()
            progress = (sim_time + 1) / self.simulation_time
            print(f'\rSimulation progress: {progress:.2%}', end='')

    def plot_adjacency_matrix(self, filename):
        plot_matrix = np.array([[0 if value is None else 1 for value in row] for row in self.adjacency_matrix])
        plt.figure(figsize=(10, 10))
        plt.imshow(plot_matrix, cmap='Greys', interpolation='none')
        plt.colorbar()
        plt.title('Neural Network Adjacency Matrix')
        plt.xlabel('Neuron j')
        plt.ylabel('Neuron i')
        # Turn off the axis ticks if you prefer a cleaner look
        plt.xticks([])
        plt.yticks([])

        # Show the plot
        plt.savefig(filename)
        plt.show()

    def plot_rassta_plots(self, filename):
        # Assuming 'data' is a 2D NumPy array where rows correspond to neurons and columns correspond to time points
        # Convert your recordings to a NumPy array if it isn't already
        data = np.array(self.recordings_v[:800])

        delta_t = self.delta_t  # Time step
        time_points = np.arange(data.shape[1]) * delta_t  # Calculate actual time points

        # Set up the figure
        plt.figure(figsize=(15, 10))

        # Use imshow to create the heatmap for the membrane potential
        plt.imshow(data, aspect='auto', origin='lower', cmap='viridis', interpolation='none',
                   extent=[0, time_points[-1], 0, data.shape[0]])
        plt.colorbar(label='Membrane Potential (mV)')  # Add a colorbar for reference

        # Overlay the spikes
        # Iterate over each neuron's data
        for neuron_idx, neuron_data in enumerate(data):
            # Find the time points where the potential is greater than 20
            spike_times = np.where(neuron_data > self._max_neurone_potential)[
                              0] * delta_t  # Adjust spike times for delta_t
            # Plot these as larger blue points
            plt.scatter(spike_times, np.full_like(spike_times, neuron_idx), color='blue',
                        s=10)  # s is the size of the point

        # Label the axes and add a title
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        plt.title('Membrane Potential Over Time for All Neurons')

        # Show the plot
        plt.savefig(filename)
        plt.show()

    def plot_mean_firing_rate(self, filename):
        def _calculate_mean_firing_rate(module_data, window_size_ms=50, shift_ms=20, total_time=1000):
            steps_per_ms = int(1 / self.delta_t)
            window_size = int(window_size_ms / self.delta_t)
            shift = int(shift_ms / self.delta_t)

            num_windows = int((total_time / self.delta_t - window_size) / shift + 1)

            mean_firing_rates = []
            for i in range(num_windows):
                start = i * shift
                end = start + window_size

                spikes_in_window = 0
                for j in range(start, end, steps_per_ms):
                    # Count the number of neurons that spiked in each 1ms interval
                    spikes_in_1ms = np.any(module_data[:, j:j + steps_per_ms] > self._max_neurone_potential, axis=1)
                    spikes_in_window += np.sum(spikes_in_1ms)
                # Calculate mean firing rate as spikes per millisecond
                mean_firing_rate = spikes_in_window / window_size_ms
                mean_firing_rates.append(mean_firing_rate)

            return mean_firing_rates

        all_mean_firing_rates = []
        for i in range(8):
            module_data = np.array(self.recordings_v[i * 100: (i + 1) * 100])
            mean_firing_rates = _calculate_mean_firing_rate(module_data)
            all_mean_firing_rates.append(mean_firing_rates)

        x_values = np.arange(0, 1000, 20)[:len(all_mean_firing_rates[0])]

        plt.figure(figsize=(10, 6))
        for i, rates in enumerate(all_mean_firing_rates):
            plt.plot(x_values, rates, label=f"Excitatory Module: {i + 1}")

        plt.xlabel('Time in ms')
        plt.ylabel('Mean Firing Rate')
        plt.title('Mean Firing Rate in Each Module')
        plt.legend()
        plt.savefig(filename)
        plt.show()


if __name__=="__main__":
    rewiring_ps = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    for index, rewiring_p in enumerate(rewiring_ps):
        module = ModulularNetwork(rewiring_p, 1000, 0.01)
        module.run_simulation()
        module.plot_adjacency_matrix(f"{index}-Adjacency-Matrix.png")
        module.plot_rassta_plots(f"{index}-Rassta-Plot.png")
        module.plot_mean_firing_rate(f"{index}-Mean-Firing-Rate.png")
