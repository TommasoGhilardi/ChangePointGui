import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from interactive_plot_module import interactive_plot  # Replace 'your_module_name' with your actual module name

def generate_fake_signal(length=300, noise_level=0.2, changepoint_loc=150):
    """Generate a synthetic signal with a changepoint for testing."""
    # Create time points
    t = np.linspace(0, 10, length)
    
    # Base signal with noise
    signal = np.zeros(length)
    signal[:changepoint_loc] = 0.2  # Before changepoint
    
    # Create a gradual increase after the changepoint
    for i in range(changepoint_loc, min(changepoint_loc + 30, length)):
        # Gradually increase over 30 samples
        progress = (i - changepoint_loc) / 30
        signal[i] = 0.2 + progress * 0.8
    
    if changepoint_loc + 30 < length:
        signal[changepoint_loc + 30:] = 1.0  # After changepoint + transition
    
    # Add noise
    noise = np.random.normal(0, noise_level, length)
    signal += noise
    
    return signal

def create_test_signals(num_trials=5):
    """Create a set of test signals for multiple trials."""
    signals_dict = {}
    for trial in range(1, num_trials + 1):
        # Randomly place changepoint between 100-200
        changepoint_loc = np.random.randint(100, 200)
        
        # Main signal
        y = generate_fake_signal(length=300, changepoint_loc=changepoint_loc)
        
        # Create X and Z signals (different patterns for demonstration)
        y1 = generate_fake_signal(length=300, noise_level=0.1, changepoint_loc=changepoint_loc-10)  # X signal
        y2 = generate_fake_signal(length=300, noise_level=0.15, changepoint_loc=changepoint_loc+10)  # Z signal
        
        # Frames/time (just incrementing for demo)
        time = np.arange(300) * 0.1
        
        # Store with changepoint info
        signals_dict[trial] = {
            "y": y,
            "y1": y1,
            "y2": y2, 
            "time": time,
            "initial_changepoint": changepoint_loc
        }
    
    return signals_dict

def main():
    # Create a temporary CSV file path
    temp_csv_path = "temp_onset_data.csv"
    
    # Create fake data
    num_trials = 5
    signals_dict = create_test_signals(num_trials)
    trial_list = list(range(1, num_trials + 1))
    
    # Sample signals from first trial
    first_trial = signals_dict[1]
    y = first_trial["y"]
    y1 = first_trial["y1"]
    y2 = first_trial["y2"]
    time = first_trial["time"]
    x = np.arange(len(y))  # X is just the index
    
    # Create initial CSV file with empty data
    if not os.path.exists(temp_csv_path):
        df = pd.DataFrame(columns=['id', 'trial', 'inital_detection', 'initial_frame', 
                                  'final_detection', 'final_frame'])
        df.to_csv(temp_csv_path, index=False)
    
    # Call the interactive plot function
    interactive_plot(
        x=x, 
        y=y, 
        y1=y1, 
        y2=y2, 
        time=time,
        main_title="Signal Visualization Demo", 
        subject="TestSubject", 
        trial=1,
        trial_list=trial_list,
        signals_dict=signals_dict,
        csv_path=temp_csv_path,
        yellow_left_end=50,
        yellow_right_start=250
    )
    
    # Clean up the temporary file after we're done (uncomment if you want to remove the file)
    # if os.path.exists(temp_csv_path):
    #     os.remove(temp_csv_path)

if __name__ == "__main__":
    main()
