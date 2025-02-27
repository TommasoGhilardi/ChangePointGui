import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from interactive_plot_module import interactive_plot

# %% Function

# Define a changepoint detection function.
def detect_changepoints(data, lower_yellow_max, upper_yellow_min, window_size=6, threshold=0.01):
    # Create a mask for the data within the specified range
    mask = np.arange(len(data))
    mask = mask[(mask >= lower_yellow_max) & (mask <= upper_yellow_min)]
    if len(mask) < window_size:
        return []
    region_data = data[mask]
    for i in range(len(region_data) - window_size + 1):
        window = region_data[i:i+window_size]
        is_increasing = True
        for j in range(1, len(window)):
            if window[j] <= window[j-1] + threshold:
                is_increasing = False
                break
        if is_increasing:
            return mask[i]
    return []


# %% Settings

# This should be the path to the EEG OneDrive folder
os.chdir(r'C:\Users\tomma\OneDrive - Birkbeck, University of London\ESC_StairCase_2023')

#### Find all the files
# TODO: Only Adults ("AD") for now
files = glob.glob(os.path.join('Data', 'Data', '*_AD', 'Motiontracking', '*_combined_new*.csv'))
#### Select Subject
file = files[7]
# Extract participant
participant = os.path.splitext(os.path.basename(file))[0][:7]

outtemporary = r'C:\Users\tomma\OneDrive - Birkbeck, University of London\ACT-UP_EEG_2025'
csv_output_path = os.path.join(outtemporary, 'Data', 'ProcessedData', 'MotionTracking', participant, "Changepoint.csv")
plot_output_path = os.path.join(outtemporary, 'Data', 'ProcessedData', 'MotionTracking', participant)
os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

#### Detection settings
initial_area = [40, 200]
initial_window_size = 6
initial_threshold = 0.01

# %% Computations

db = pd.read_csv(file, low_memory=False)
db = db.assign(X=0.0, Z=0.0, MotionStartFrame=0)

# Prepare a dictionary to store signals for each trial and a list for trial numbers.
signals_dict = {}
trial_list = []

Db = pd.DataFrame(columns=['id', 'trial', 'inital_detection', 'initial_frame', 'final_detection', 'final_frame'])

# Loop over all trials (e.g. 80 trials)
for trial in range(80):
    df = db.loc[db['Trial.ordinal'] == trial, :].copy().reset_index(drop=True)
    
    # Calculate the mean for each axis from three sources (only for X and Z)
    for axis in ("X", "Z"):
        mean = df[[f'DoubleTop_{axis}', f'DoubleLow_{axis}', f'Single_{axis}']].mean(axis=1)
        df[axis] = mean
    
    # Compute differences and then speed using only X and Z.
    df['dx'] = df['X'].diff()
    df['dz'] = df['Z'].diff()
    df['df'] = df['Frame'].diff()
    df['distance'] = np.sqrt(df['dx']**2 + df['dz']**2)
    df['speed'] = df['distance'] / df['df']
    df = df.drop(columns=['dx', 'dz']).reset_index(drop=True)
    
    # Compute baselines and correct the signals
    speed_total = df['speed'].fillna(0).values
    baseline_speed = np.mean(speed_total[:20])
    baseline_x = np.mean(df['X'].values[:20])
    baseline_z = np.mean(df['Z'].values[:20])
    
    corrected_speed = speed_total - baseline_speed
    corrected_x = np.abs(df['X'].values - baseline_x)
    corrected_z = np.abs(df['Z'].values - baseline_z)

    frames = df['Frame'].values

    #### Find the changepoint for the trial
    Initial_changepoint = detect_changepoints(corrected_speed,
                                              initial_area[0], initial_area[1],
                                              window_size=initial_window_size,
                                              threshold=initial_threshold)

    trial_list.append(trial)

    # Build signals dictionary with only two auxiliary signals: X and Z.
    signals_dict[trial] = {
         "y": corrected_speed,  # Main signal (speed)
         "y1": corrected_x,     # Auxiliary signal: X
         "y2": corrected_z,     # Auxiliary signal: Z
         "time": frames,
         "initial_changepoint": Initial_changepoint
    }

    # Append trial info to dataframe
    Db.loc[len(Db), :] = [participant, trial, Initial_changepoint, frames[Initial_changepoint], np.nan, np.nan]

Db.to_csv(csv_output_path)

# %%
# Call the GUI function with the first trial’s data.
first_trial = trial_list[0]
trial_data = signals_dict[first_trial]
x_initial = np.arange(len(trial_data["y"]))

interactive_plot(
    x=x_initial,
    y=trial_data["y"],
    y1=trial_data["y1"],
    y2=trial_data["y2"],
    time=trial_data["time"],
    main_title="Motion Tracking Data",
    subject=participant,
    trial=str(first_trial),
    trial_list=trial_list,
    signals_dict=signals_dict,
    yellow_left_end=initial_area[0],
    yellow_right_start=initial_area[1],
    csv_path=csv_output_path
)

# %% Save plots

Db = pd.read_csv(csv_output_path,
    dtype={'id': str, 'trial': int, 'inital_detection': float, 'initial_frame': float, 
           'final_detection': float, 'final_frame': float})

for i in range(len(Db)):
    if not np.isnan(Db["final_frame"][i]):
        initial_point = int(Db.loc[i, 'inital_detection'])
        final_point = int(Db.loc[i, 'final_detection'])
        info_plot = f'Subject{participant}_Trial{i}'

        plt.figure(figsize=(10, 8))
        plt.plot(signals_dict[i]['y'])
        plt.plot(initial_point, signals_dict[i]['y'][initial_point],
                 marker='o', markersize=8, markerfacecolor='none', 
                 markeredgecolor='orange', linestyle='None', label='Initial Detection')
        plt.plot(final_point, signals_dict[i]['y'][final_point],
                 marker='o', markersize=8, markerfacecolor='none', 
                 markeredgecolor='green', linestyle='None', label='Final Detection')
        plt.title(info_plot)
        plt.legend()

        plt.savefig(os.path.join(plot_output_path, info_plot + ".png"), dpi=300, bbox_inches='tight')
