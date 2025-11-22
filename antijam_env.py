import gymnasium as gym
import numpy as np
import random as random
from gymnasium import spaces
from gymnasium.envs.registration import register     #name the env
from gymnasium.utils.env_checker import check_env   #check if the env follows the gym API   
import matplotlib.pyplot as plt
from IPython.display import display
import time

#make the name of the env
register(
    id='AntiJammingEnv-v0',                         #name of the env when we instatiate
    # entry_point="simplenv.ipynb:AntiJammingEnv",   #module name:class name
    entry_point= lambda **kwargs: AntiJammingEnv(**kwargs),  #if the class is in the same file, we can use lambda
)


class AntiJammingEnv(gym.Env):
    # used to render the env
    metadata = {"render_modes": ["human"], "render_fps": 4}


    def __init__(self, num_freqs=22, num_powers=50, num_of_jammers=7, jammer_type="sweeping", render_mode=None):
        super().__init__()
        self.num_freqs = num_freqs
        self.num_powers = num_powers
        self.jammer_type = jammer_type          #random in this env
        self.num_of_jammers = num_of_jammers    # 7 random jammers
        self._sweep_step = 1             # how many bins to move per step for sweeping jammer
        self.power_levels = np.linspace(1.58e-4, 1e-3, num_powers)  # power levels from -8dbm to 0dbm converted to watts , taken from the paper (50 levels)
        self.jammer_power = 3.162e-3 # Jammer power 3.162e-3 W = 5 dBm from Table 1 in the paper

        #modifications to state
        self.state = None                 # current observation (PSD vector)
        self.state_dim = num_freqs + 1  #PSD + prev_freq , no prev_crc
        self.prev_freq = None           #need for freq switching penalty
        # self.prev_crc = None            #new
    
        # moved from step() to be class variables
        self.power_noise  = 3.162e-16      # NEW (kTB*N; matches your comment)
        self.alpha_k_user   = 9.1e-12        # NEW (Rayleigh scale for user)
        self.alpha_k_jammer = 1.15e-12       # NEW (Rayleigh scale for jammer)
        self._gain_user   = np.zeros(self.num_freqs)   # NEW (slot user gains)
        self._gain_jammer = np.zeros(self.num_freqs)   # NEW (slot jammer gains)


        #all rendering related variables
        self.render_mode = render_mode
        self._fig = None
        self._ax = None
        self._jam_bars = None
        self._agent_line = None
        self._power_text = None
        self._last_action = None
        self._last_reward = None
        self._total_reward = 0.0
        self.agent_marker = None
        self._display_handle = None


        # Define action space (freq × power combinations)
        # 22 * 50 = 1100 discrete actions (0..1099)
        # too many actions for DQN, but OK for policy gradient methods
        self.action_space = spaces.Discrete(num_freqs * num_powers)
        # self.action_space = spaces.MultiDiscrete([self.num_freqs, self.num_powers])

        # State = PSD (vector of size num_freqs)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )

        #self.rng = random number generator
        self.rng = np.random.default_rng()

        # Internal jammer state & dynamics parameters
        self._jam_idxs = []             # indices of the currently jammed frequency bins


    def _build_state_from_jammer(self):
        """
        Construct the PSD observation vector given the *current* jammers indecies.
        - Background noise self.power_noise (kTB*NF from paper, e-16 range) plus 5% gaussian noise
        - Fixed scaling (stationary across steps) to fit in state space (0..1)
        
        This function is used in both `reset()` and `step()` so that
        the observation is always generated consistently.
        """
        s = np.full(self.num_freqs, self.power_noise)   # background noise = power noise (kTBN)

        if len(self._jam_idxs) == 0:       # choose jammer indices if not already chosen           
            self._jam_idxs = self.np_random.choice(range(self.num_freqs), size=self.num_of_jammers, replace=False).tolist()

        # set jammer powers
        jam_idxs_array = np.array(self._jam_idxs, dtype=int)  # helper array to avoid for loop to set each jammer power
        s[jam_idxs_array] += self.jammer_power * self._gain_jammer[jam_idxs_array] # psd = jammer power + noise 

        # add gaussian measurement noise and clip at low floor
        s = s * (1.0 + 0.05 * self.rng.standard_normal(self.num_freqs))
        s = np.clip(s, self.power_noise * 0.5, None)

        # FIXED scaling (stationary across steps)
        # ~95% jammed values lie within headroom when gains are exponential
        headroom = self.jammer_power * self.alpha_k_jammer * 3.0
        denom = max(headroom, 1e-30)
        s_normalized = (s - self.power_noise) / denom
        s_normalized = np.clip(s_normalized, 0.0, 1.0)

        # add prev_freq and prev_crc to the state
        prev_f_norm = 0.0 if self.prev_freq is None else self.prev_freq / (self.num_freqs - 1)
        full_state  = np.concatenate([s_normalized, [prev_f_norm]], axis=0)

        return full_state.astype(np.float32)
    
    

    def _advance_jammer(self):
        """
        Update the jammer index based on jammer_type dynamics:
        - fixed: jammer never moves
        - sweeping: jammer shifts to the next bin each step
        - random: jammer picks a new random bin every step
        """

        jt = self.jammer_type.lower()
        if jt == "fixed":
            pass  # jammer stays in the same bin forever

        elif jt == "sweeping":
            for i in range(len(self._jam_idxs)):
                #modulo to wrap around (23 -> 1 for example)
                self._jam_idxs[i] = (self._jam_idxs[i] + self._sweep_step) % self.num_freqs

        elif jt == "random":
            # for i in range(len(self._jam_idxs)):
            #     self._jam_idxs[i] = int(self.rng.integers(0, self.num_freqs)) May lead to duplicates
            self._jam_idxs = self.rng.choice(self.num_freqs, size=self.num_of_jammers, replace=False).tolist()  # NEW



    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize jammer index and build the first observation
        # used self.np_random that apparently got initted by super().reset(seed=seed)

        #new reset function
        self._jam_idxs = self.np_random.choice(self.num_freqs, size=self.num_of_jammers, replace=False).tolist() 
        self._gain_user   = self.alpha_k_user   * -1 * np.log(self.rng.random(self.num_freqs))  # NEW, needs to be initialized for s0
        self._gain_jammer = self.alpha_k_jammer * -1 * np.log(self.rng.random(self.num_freqs))  # NEW, needs to be initialized for s0

        self.prev_freq = None
        self.prev_crc  = 0.0
        self.state = self._build_state_from_jammer()

        info = {}

        #rendering related book-keeping
        self._last_state = self.state      # so first render works
        self._last_action = None
        self._last_reward = None

        return self.state.astype(np.float32), info
    



    def step(self, action):
        """
        One-slot delay logic:
        - Agent picked `action` using observation s_k (= self._last_state).
        - Environment first advances jammer to unseen slot k+1 and measures s_{k+1}.
        - Reward is computed against s_{k+1} (the actual slot you just transmitted in).
        - Next observation returned to agent is s_{k+1}.
        """
        # Decode (freq, power)
        freq = action // self.num_powers
        power = action % self.num_powers   # 1..num_powers
        # freq, power = map(int, action)  #power here is the index of the power level
        power = self.power_levels[power]

        self._gain_user   = self.alpha_k_user   * -1 * np.log(self.rng.random(self.num_freqs))  # NEW
        self._gain_jammer = self.alpha_k_jammer * -1 * np.log(self.rng.random(self.num_freqs))  # NEW

        #for simplicity and faithfulness to old code
        # alpha_k_user   = self.alpha_k_user   # from SNR_clean target (≈12 dB)
        # alpha_k_jammer = self.alpha_k_jammer  # from desired jammed SINR (≈0 dB)
        gain_vector_user = self._gain_user # channel gain vector for all frequencies sampled each step 
        gain_vector_jammer = self._gain_jammer # channel gain vector for all frequencies sampled each step
          


        # === Move to *new* (unseen) slot and measure PSD_{k+1} ===
        #uncomment this 2 lines for 1slot delay
        self._advance_jammer()
        curr_psd = self._build_state_from_jammer()    # s_{k+1}

        #uncomment this line for 0slot delay
        # curr_psd = self.state  # s_{k}

        # --- Compute reward using the *current* slot conditions (which were not observed before acting) ---
        jammed = (freq in self._jam_idxs)  # NEW
        powerNoise = self.power_noise     # NEW
        jamming_power = self.jammer_power * gain_vector_jammer[freq] if jammed else 0.0 # if jammer is present, its power is 3.162e-3 W (5 dBm) * gain of jammer as in the paper, else very small power 1e-10 W
        sinr = (power*gain_vector_user[freq]) / (powerNoise+jamming_power+1e-30)

        #crc calculation
        beta_th = 10**(6/10)  # SINR threshold for successful CRC, it is taken from GPT as a standard practice with QPSK modulation, but can be manipulated between 6-10 dbm , this shows 8dbm
        # right before crc_success line
        den = powerNoise + (self.jammer_power * gain_vector_jammer[freq] if jammed else 0.0)
        sinr = (power * gain_vector_user[freq]) / (den + 1e-30)
        sinr_db = 10*np.log10(sinr + 1e-30)
        # print(f"clean={not jammed} P={power:.2e} gU={gain_vector_user[freq]:.2e} den={den:.2e} SINR={sinr_db:.2f}dB pass={sinr>beta_th}")
        crc_success = 1.0 if sinr > beta_th else 0.0

        # Switching penalty (compare with last used freq)   
        switching_penalty = 0.0
        if self.prev_freq is not None and self.prev_freq != freq:
            switching_penalty = 1.0
        
    
        lambd = 600
        # subBandWidth = 25e3 # 25 kHz , taking from the paper that the base is 20 kHz and some roll-off margin
        # epsilon = 1e4 # taken from GPT, suggested to make the affect of the power in the same scale with swithcing and etc.. 
        alpha = 1.2# weight for switching penalty from the paper Table 1
        # reward =  crc_success * (subBandWidth * np.log2(1 + sinr)) / (epsilon * power) - alpha * switching_penalty
        reward = crc_success * ((np.log2(1 + sinr))) - lambd * power - alpha * switching_penalty



        # reward = (crc_success * (((subBandWidth * np.log2(1+sinr))/(epsilon*power))))-(alpha*switching_penalty)
#             print(
#     f"\n[Step Debug]"
#     f"\nFreq: {freq} | Power (W): {power:.3e} | PSD@Freq: {noise_at_chosen:.2f} "
#     f"| Jammed: {noise_at_chosen>0.9}"
#     f"\nUserGain: {gain_vector_user[freq]:.3e} | JammerGain: {gain_vector_jammer[freq]:.3e}"
#     f"\nSINR: {sinr:.3e} | SINR_dB: {10*np.log10(sinr+1e-12):.2f} dB"
#     f"\nCRC: {crc_success} | SwitchPenalty: {switching_penalty}"
#     f"\nReward: {reward:.6f} | TotalReward: {self._total_reward + reward:.6f}"
# )

        # Update previous freq
        self.prev_freq = freq
        # self.prev_crc = float(crc_success)

        # === One-slot-delay book-keeping ===
        # Next observation is the PSD we just measured in this slot.
        
        #uncomment this for 1slot delay
        self._last_state = curr_psd      # used for rendering
        self.state = curr_psd

        #uncomment this for 0slot delay
        # self._advance_jammer()
        # next_state = self._build_state_from_jammer()   # s_{k+1}
        # self.state = next_state

        terminated = False
        truncated = False
        info = {
            "crc_success": float(crc_success),      # 0 / 1
            "switched":    float(switching_penalty > 0),  # 0 / 1
            "power_w":     float(power)             # chosen power (W)
        }

        #rendering related book-keeping
        self._last_action = (freq, power)
        self._last_reward = float(reward)
        self._total_reward = self._total_reward + reward
        if self.render_mode == "human":
            self.render()
        
        return self.state.astype(np.float32), reward, terminated, truncated, info


    def render(self):
        if self.render_mode != "human":
            return

        # ── show the PSD from the slot where the last action was used ────────────
        psd_to_show = self._last_state[:self.num_freqs] if getattr(self, "_last_state", None) is not None else self.state[:self.num_freqs]
        # ─────────────────────────────────────────────────────────────────────────

        # Lazily build the figure once
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            plt.ioff()
            self._fig, self._ax = plt.subplots(figsize=(10, 4))
            self._ax.set_title("Anti-jamming spectrum (live)")
            self._ax.set_xlabel("Frequency bin")
            self._ax.set_ylabel("PSD / jammer power")
            self._ax.set_ylim(0.0, 1.2)
            self._ax.set_xlim(-0.5, self.num_freqs - 0.5)

            self._jam_bars = self._ax.bar(
                range(self.num_freqs),
                psd_to_show if psd_to_show is not None else np.zeros(self.num_freqs),
                color="#1f77b4",
                alpha=0.75,
            )
            self._agent_marker = self._ax.scatter(
                [], [], s=120, color="#ff7f0e", edgecolor="black", zorder=5
            )
            self._power_text = self._ax.text(
                0.01, 0.97, "Awaiting first action",
                transform=self._ax.transAxes,
                fontsize=11, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )
            self._fig.canvas.draw()
            self._display_handle = display(self._fig, display_id=True)

        # ── update jammer bars ────────────────────────────────────────────────────
        if psd_to_show is not None:
            for bar, height in zip(self._jam_bars, psd_to_show):
                bar.set_height(height)
                bar.set_color("#d62728" if height > 0.9 else "#1f77b4")

        # ── update agent marker & annotation ──────────────────────────────────────
        if self._last_action is not None:
            freq, power_w = self._last_action
            self._agent_marker.set_offsets([[freq, psd_to_show[freq]]])

            lines = [
                f"Agent freq = {freq}",
                f"Power = {power_w:.2e} W",
            ]
            if self._last_reward is not None:
                lines.append(f"Reward = {self._last_reward:.3f}")
            if self._total_reward is not None:
                lines.append(f"Total = {self._total_reward:.3f}")

            self._power_text.set_text("\n".join(lines))
        else:
            self._agent_marker.set_offsets([])
            self._power_text.set_text("Awaiting first action")

        # ── draw & throttle frame rate ────────────────────────────────────────────
        self._fig.canvas.draw()
        if self._display_handle is not None:
            self._display_handle.update(self._fig)
            time.sleep(0.8)
