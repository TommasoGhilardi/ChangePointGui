import os
import numpy as np
import FreeSimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

def interactive_plot(x, y, y1, y2, time=None,
                     zoom_x_range=50, zoom_y_range=5, 
                     main_title="My Interactive Signal Plot", subject="Unknown", trial="Unknown",
                     trial_list=None,
                     signals_dict=None,
                     csv_path="onset_data.csv",
                     yellow_left_end=20,   # left yellow region end
                     yellow_right_start=200):  # right yellow region start
    """
    Interactive GUI that displays a main signal plot, a zoomed plot, and two top plots for X and Z.
    
    Parameters:
      - x, y, y1, y2: arrays of signal values.
      - time: A vector of Frame values corresponding to each sample.
      - zoom_x_range, zoom_y_range: Parameters for the zoomed view.
      - main_title, subject, trial: Strings used for display.
      - trial_list: A list of trial identifiers.
      - signals_dict: A dictionary containing signal data for each trial.
      - csv_path: File path where the CSV (with detection info) will be read/written.
      - yellow_left_end, yellow_right_start: Boundaries for the yellow shaded regions.
      
    The initial cursor position is determined using a precomputed changepoint if available,
    otherwise it defaults to sample index 100.
    """
    if trial_list is None:
        trial_list = [trial]
    current_trial_index = 0
    current_trial = trial_list[current_trial_index]
    if signals_dict is None:
        signals_dict = {tr: {"y": y, "y1": y1, "y2": y2, "time": time} for tr in trial_list}
    trial_signals = signals_dict[current_trial]
    segment_length = len(trial_signals["y"])
    x = np.arange(segment_length)

    def detect_changepoints(data, lower_yellow_max, upper_yellow_min, window_size=6, threshold=0.01):
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
                return [mask[i]]
        return []

    if "initial_changepoint" in trial_signals and trial_signals["initial_changepoint"]:
        cur_x = int(round(trial_signals["initial_changepoint"]))
    else:
        cur_x = 100

    yellow_left_end = max(0, min(yellow_left_end, segment_length - 10))
    yellow_right_start = max(yellow_left_end + 10, min(yellow_right_start, segment_length - 1))
    
    display_title = f"{main_title} - Subject: {subject} Trial: {current_trial}"

    def draw_figure(canvas, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg

    # Create initial figures.
    main_fig, main_ax = plt.subplots(figsize=(6, 4))
    zoom_fig, zoom_ax = plt.subplots(figsize=(3, 2))
    # Changed to two top plots (for X and Z) with adjusted figure size and tight layout
    fig_top, axs_top = plt.subplots(1, 2, figsize=(10, 2), sharex=True)
    # Apply tight layout to make better use of space
    fig_top.tight_layout(pad=2.0)

    main_vline = None
    main_hline = None
    zoom_vline = None
    zoom_hline = None
    top_vlines = []

    filter_size = 0

    def update_plots(new_x, slider_vals=None):
        nonlocal cur_x, main_vline, main_hline, zoom_vline, zoom_hline, top_vlines
        cur_x = int(np.clip(round(new_x), 0, segment_length - 1))
        cur_y = trial_signals["y"][cur_x]
        if filter_size <= 1:
            filtered_y = trial_signals["y"]
        else:
            kernel = np.ones(filter_size) / filter_size
            filtered_y = np.convolve(trial_signals["y"], kernel, mode='same')
        
        if slider_vals is None:
            try:
                lower_yellow_max = float(window['-LOWER_YELLOW_MAX-'].get())
            except Exception:
                lower_yellow_max = yellow_left_end
            try:
                upper_yellow_min = float(window['-UPPER_YELLOW_MIN-'].get())
            except Exception:
                upper_yellow_min = yellow_right_start
            try:
                window_size_val = int(window['-WINDOW_SIZE-'].get())
            except Exception:
                window_size_val = 6
            try:
                threshold_val = float(window['-THRESHOLD-'].get())
            except Exception:
                threshold_val = 0.01
        else:
            lower_yellow_max = float(slider_vals['-LOWER_YELLOW_MAX-'])
            upper_yellow_min = float(slider_vals['-UPPER_YELLOW_MIN-'])
            try:
                window_size_val = int(slider_vals['-WINDOW_SIZE-'])
            except:
                window_size_val = 6
            try:
                threshold_val = float(slider_vals['-THRESHOLD-'])
            except:
                threshold_val = 0.01

        cps = detect_changepoints(filtered_y, lower_yellow_max, upper_yellow_min,
                                  window_size=window_size_val, threshold=threshold_val)
        
        # Update main plot.
        main_ax.clear()
        main_ax.axvspan(0, lower_yellow_max, color='yellow', alpha=0.35, zorder=0)
        main_ax.axvspan(upper_yellow_min, segment_length - 1, color='yellow', alpha=0.35, zorder=0)
        main_ax.plot(x, filtered_y, label="Signal", color='blue', zorder=2)
        main_ax.axhline(y=0, color='black', linewidth=1, zorder=1)
        main_vline = main_ax.axvline(x=cur_x, color='red', linestyle='--', zorder=3)
        main_hline = main_ax.axhline(y=cur_y, color='green', linestyle='--', zorder=3)
        for cp in cps:
            main_ax.plot(cp, filtered_y[cp], 'o', markerfacecolor='none', markeredgecolor='orange',
                         markeredgewidth=2, markersize=10, alpha=0.9, zorder=4)
        main_ax.set_title(f"{main_title} - Subject: {subject} Trial: {current_trial}")
        main_ax.set_xlabel("Sample Index")
        main_ax.set_ylabel("Amplitude")
        main_ax.set_xlim(0, segment_length - 1)
        main_ax.relim()
        main_ax.autoscale_view()
        changepoints_text = str(cps[0]) if cps else "None"
        window['-CHANGEPOINT_VALUE-'].update(changepoints_text)
        main_fig_agg.draw()
        
        # Update zoom plot.
        zoom_ax.clear()
        zoom_ax.plot(x, filtered_y, label="Signal", color='blue')
        zoom_ax.axhline(y=0, color='black', linewidth=1, zorder=1)
        zoom_vline = zoom_ax.axvline(x=cur_x, color='red', linestyle='--', zorder=2)
        zoom_hline = zoom_ax.axhline(y=cur_y, color='green', linestyle='--', zorder=2)
        for cp in cps:
            if lower_yellow_max <= cp <= upper_yellow_min:
                zoom_ax.plot(cp, filtered_y[cp], 'o', markerfacecolor='none', markeredgecolor='orange',
                             markeredgewidth=2, markersize=10, alpha=0.9, zorder=4)
        zoom_ax.set_title("Zoomed Signal")
        zoom_ax.set_xlabel("Sample Index")
        zoom_ax.set_ylabel("Amplitude")
        zoom_ax.set_xlim(lower_yellow_max, upper_yellow_min)
        ymin = max(cur_y - zoom_y_range, -1)
        ymax = cur_y + zoom_y_range
        zoom_ax.set_ylim(ymin, ymax)
        zoom_fig_agg.draw()
        
        # Update top plots for X and Z.
        top_vlines = []
        labelplots = ['X', 'Z']
        colors = ['orange', 'cyan']
        for i, ax in enumerate(axs_top):
            ax.clear()
            key = "y1" if i == 0 else "y2"
            ax.plot(x, trial_signals[key], label=labelplots[i], color=colors[i])
            ax.axhline(y=0, color='black', linewidth=1, zorder=1)
            vline = ax.axvline(x=cur_x, color='red', linestyle='--', zorder=2)
            top_vlines.append(vline)
            ax.set_xticks(np.linspace(0, segment_length - 1, 5))
            ax.set_yticks([])
            ax.set_xlim(lower_yellow_max, upper_yellow_min)
            ax.set_title(labelplots[i])
        # Apply tight layout to adjust spacing
        fig_top.tight_layout()
        fig_top_agg.draw()
        window["-CURRENT_SAMPLE-"].update(f"{cur_x}")

    def update_cursor_position(new_x):
        nonlocal cur_x
        cur_x = int(np.clip(round(new_x), 0, segment_length - 1))
        cur_y = trial_signals["y"][cur_x]
        try:
            main_vline.set_xdata([cur_x, cur_x])
            main_hline.set_ydata([cur_y, cur_y])
            zoom_vline.set_xdata([cur_x, cur_x])
            zoom_hline.set_ydata([cur_y, cur_y])
            for line in top_vlines:
                line.set_xdata([cur_x, cur_x])
        except Exception as e:
            sg.popup_no_buttons(f"Error updating cursor: {e}", auto_close=True, auto_close_duration=0.75)
        window["-CURRENT_SAMPLE-"].update(f"{cur_x}")
        main_fig_agg.draw()
        zoom_fig_agg.draw()
        fig_top_agg.draw()
        
    def load_onset_for_trial():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, 
                             dtype={'id': str, 'trial': int, 'inital_detection': float, 'initial_frame': float, 
                                    'final_detection': float, 'final_frame': float})
            df['id'] = df['id'].str.strip().str.lower()
            subj_str = str(subject).strip().lower()
            mask = (df['id'] == subj_str) & (df['trial'] == current_trial)
            if mask.any():
                row = df.loc[mask].iloc[0]
                if not pd.isna(row['final_detection']):
                    return int(row['final_detection'])
        if "initial_changepoint" in trial_signals and trial_signals["initial_changepoint"]:
            return int(round(trial_signals["initial_changepoint"]))
        else:
            return 100

    def update_top_plots_xrange(xmin, xmax):
        try:
            xmin, xmax = float(xmin), float(xmax)
            if xmin < xmax:
                for ax in axs_top:
                    ax.set_xlim(xmin, xmax)
                fig_top_agg.draw()
            else:
                sg.popup_no_buttons("X-Min must be less than X-Max.", auto_close=True, auto_close_duration=0.75)
        except ValueError:
            sg.popup_no_buttons("Please enter valid numbers for X-Min and X-Max.", auto_close=True, auto_close_duration=0.75)
    
    def on_click(event):
        if event.inaxes == main_ax:
            update_cursor_position(event.xdata)
    
    def on_scroll(event):
        if event.inaxes == main_ax and event.key == 'control':
            xlim = main_ax.get_xlim()
            ylim = main_ax.get_ylim()
            zoom_factor = 1.1 if event.button == 'up' else 1 / 1.1
            x_center = event.xdata
            y_center = event.ydata
            new_xlim = ((xlim[0] - x_center) * zoom_factor + x_center,
                        (xlim[1] - x_center) * zoom_factor + x_center)
            new_ylim = ((ylim[0] - y_center) * zoom_factor + y_center,
                        (ylim[1] - y_center) * zoom_factor + y_center)
            main_ax.set_xlim(new_xlim)
            main_ax.set_ylim(new_ylim)
            main_fig_agg.draw()
    
    def reset_zoom():
        nonlocal zoom_x_range, zoom_y_range
        zoom_x_range = 50
        zoom_y_range = 5
        window['-ZOOM_Y-'].update(value=zoom_y_range)
        main_ax.set_xlim(0, segment_length - 1)
        main_ax.relim()
        main_ax.autoscale_view()
        for ax in axs_top:
            ax.set_xlim(0, segment_length - 1)
        update_plots(cur_x)
    
    def reset_default_settings():
        window["-LOWER_YELLOW_MAX-"].update(value=yellow_left_end)
        window["-UPPER_YELLOW_MIN-"].update(value=yellow_right_start)
        window["-WINDOW_SIZE-"].update(value=6)
        window["-THRESHOLD-"].update(value=0.01)
    
    left_column = sg.Column([
        [sg.Canvas(key='-ZOOM_CANVAS-')],
        [sg.Text("Zoom Y:"), sg.Slider(range=(0.1, 5), default_value=zoom_y_range, resolution=0.01, 
                                       orientation='h', key='-ZOOM_Y-', enable_events=True, size=(20, 10))],
        [sg.Text("Sliding Window Detection", font=("Helvetica", 12, "bold"), pad=((0,0), (10, 5)))],
        [sg.Text("Window Size:"), sg.Slider(range=(2, 20), default_value=6, resolution=1, 
                                            orientation='h', key='-WINDOW_SIZE-', enable_events=True, size=(20, 8))],
        [sg.Text("Threshold:"), sg.Slider(range=(0, 1), default_value=0.01, resolution=0.01, 
                                          orientation='h', key='-THRESHOLD-', enable_events=True, size=(20, 8))],
        [sg.Text("Changepoint:", size=(15, 1)), sg.Text("None", key="-CHANGEPOINT_VALUE-", size=(20, 1))],
        [sg.Button("Go to Changepoint", size=(15, 1), pad=((0,0),(5,10)))]
    ], element_justification='left')
    
    center_column = sg.Column([
        [sg.Canvas(key='-MAIN_CANVAS-')],
        [sg.Slider(range=(0, segment_length - 1), orientation='h', key='-LOWER_YELLOW_MAX-', default_value=yellow_left_end,
                   enable_events=True, expand_x=True)],
        [sg.Slider(range=(0, segment_length - 1), orientation='h', key='-UPPER_YELLOW_MIN-', default_value=yellow_right_start,
                   enable_events=True, expand_x=True)]
    ])
    
    right_column = sg.Column([
        [sg.Text("Top Plot X Range", font=("Helvetica", 12, "bold"), pad=((0,0), (10, 5)))],
        [sg.Text("X-Min:"), sg.InputText("0", key="-TOP_XMIN-", size=(10, 1))],
        [sg.Text("X-Max:"), sg.InputText(f"{segment_length - 1}", key="-TOP_XMAX-", size=(10, 1))],
        [sg.Button("Set Top Plot Range", size=(15, 1), pad=((0, 0), (10, 20)))],
        [sg.Text("Sample Control", font=("Helvetica", 12, "bold"), pad=((0,0), (10, 5)))],
        [sg.Text("Current Sample:", size=(15, 1)), sg.Text(f"{cur_x}", key="-CURRENT_SAMPLE-", size=(20, 1))],
        [sg.Text("Move to sample:"), sg.InputText(f"{cur_x}", key="-NEW_SAMPLE-", size=(10, 1))],
        [sg.Button("Go", size=(7, 2), pad=((0,0),(5,10)))],
        [sg.Text("Filter Settings", font=("Helvetica", 12, "bold"), pad=((0,0), (10, 5)))],
        [sg.Radio("0", "FILTER", key="-FILTER_0-", default=True, enable_events=True),
         sg.Radio("2", "FILTER", key="-FILTER_2-", enable_events=True)],
        [sg.Radio("5", "FILTER", key="-FILTER_5-", enable_events=True),
         sg.Radio("10", "FILTER", key="-FILTER_10-", enable_events=True)],
        [sg.Radio("20", "FILTER", key="-FILTER_20-", enable_events=True),
         sg.Radio("30", "FILTER", key="-FILTER_30-", enable_events=True)],
        [sg.Button("Reset", size=(10,2), pad=((0,0),(10,0)))]
    ], justification='left', pad=(10, 10))

    # Create a container for the top canvas with expand_x=True to center it
    top_container = sg.Column([
        [sg.Canvas(key='-TOP_CANVAS-', size=(1200, 120))]
    ], expand_x=True, element_justification='center')
    
    bottom_controls = [
        sg.Column([
            [sg.Button("<<", size=(10,2)),
             sg.Button("-5", size=(10,2)),
             sg.Button("-1", size=(10,2)),
             sg.Button("Ok", size=(10,2)),
             sg.Button("+1", size=(10,2)),
             sg.Button("+5", size=(10,2)),
             sg.Button(">>", size=(10,2))]
        ], expand_x=True, element_justification='center')
    ]
    
    layout = [
        [sg.Text(display_title, key="-DISPLAY_TITLE-", font=("Helvetica", 16, "bold"), justification="center", expand_x=True)],
        [top_container],  # Use the container to center the top plots
        [left_column, center_column, right_column],
        [bottom_controls]
    ]   
    window = sg.Window("Interactive Signal Plot with Zoom", layout, size=(1300, 800),
                       finalize=True, return_keyboard_events=True)
    
    window.bind('<Left>', '-1')
    window.bind('<Right>', '+1')
    
    main_canvas = window['-MAIN_CANVAS-'].TKCanvas
    zoom_canvas = window['-ZOOM_CANVAS-'].TKCanvas
    top_canvas = window['-TOP_CANVAS-'].TKCanvas
    main_fig_agg = draw_figure(main_canvas, main_fig)
    zoom_fig_agg = draw_figure(zoom_canvas, zoom_fig)
    fig_top_agg = draw_figure(top_canvas, fig_top)
    
    update_plots(cur_x)
    main_fig.canvas.mpl_connect("button_press_event", on_click)
    main_fig.canvas.mpl_connect("scroll_event", on_scroll)
    
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        elif event in ['-1', 'Left:37', 'Left:']:
            update_cursor_position(cur_x - 1)
        elif event in ['+1', 'Right:39', 'Right:']:
            update_cursor_position(cur_x + 1)
        elif event == '-5':
            update_cursor_position(cur_x - 5)
        elif event == '+5':
            update_cursor_position(cur_x + 5)
        elif event == "Ok":
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, 
                                 dtype={'id': str, 'trial': int, 'inital_detection': float, 'initial_frame': float, 
                                        'final_detection': float, 'final_frame': float})
                df['id'] = df['id'].str.strip().str.lower()
            else:
                df = pd.DataFrame(columns=['id', 'trial', 'inital_detection','initial_frame','final_detection','final_frame'])
            subj_str = str(subject).strip().lower()
            mask = (df['id'] == subj_str) & (df['trial'] == current_trial)
            final_sample = cur_x
            final_frame = time[cur_x] if time is not None else np.nan
            if mask.any():
                df.loc[mask, 'final_detection'] = final_sample
                df.loc[mask, 'final_frame'] = final_frame
            else:
                new_row = {'id': subj_str, 'trial': current_trial, 'inital_detection': np.nan, 'initial_frame': np.nan,
                           'final_detection': final_sample, 'final_frame': final_frame}
                df = df.append(new_row, ignore_index=True)
            df.to_csv(csv_path, index=False)
            sg.popup_no_buttons(f"Final detection saved for Subject {subject}, Trial {current_trial}!", 
                                title="Info", auto_close=True, auto_close_duration=0.75)
            if current_trial_index < len(trial_list) - 1:
                current_trial_index += 1
                current_trial = trial_list[current_trial_index]
                display_title = f"{main_title} - Subject: {subject} Trial: {current_trial}"
                window["-DISPLAY_TITLE-"].update(display_title)
                trial_signals = signals_dict[current_trial]
                segment_length = len(trial_signals["y"])
                x = np.arange(segment_length)
                cur_x = load_onset_for_trial()
                reset_default_settings()
                update_plots(cur_x)
            else:
                sg.popup_no_buttons("Already at the last trial.", auto_close=True, auto_close_duration=0.75)
        elif event == 'Go':
            try:
                new_val = float(values["-NEW_SAMPLE-"])
                update_cursor_position(new_val)
            except ValueError:
                sg.popup_no_buttons("Please enter a valid number.", auto_close=True, auto_close_duration=0.75)
        elif event == "Set Top Plot Range":
            xmin = values["-TOP_XMIN-"]
            xmax = values["-TOP_XMAX-"]
            update_top_plots_xrange(xmin, xmax)
        elif event == '-ZOOM_Y-':
            zoom_y_range = float(values['-ZOOM_Y-'])
            update_plots(cur_x, values)
        elif event in ['-WINDOW_SIZE-', '-THRESHOLD-']:
            update_plots(cur_x, values)
        elif event in ['-LOWER_YELLOW_MAX-', '-UPPER_YELLOW_MIN-']:
            update_plots(cur_x, values)
        elif event in ["-FILTER_0-", "-FILTER_2-", "-FILTER_5-", "-FILTER_10-", "-FILTER_20-", "-FILTER_30-"]:
            if event == "-FILTER_0-":
                filter_size = 0
            elif event == "-FILTER_2-":
                filter_size = 2
            elif event == "-FILTER_5-":
                filter_size = 5
            elif event == "-FILTER_10-":
                filter_size = 10
            elif event == "-FILTER_20-":
                filter_size = 20
            elif event == "-FILTER_30-":
                filter_size = 30
            update_plots(cur_x, values)
        elif event == "Reset":
            reset_zoom()
        elif event == "Go to Changepoint":
            try:
                changepoint_text = window["-CHANGEPOINT_VALUE-"].get()
                if changepoint_text != "None":
                    changepoint = int(changepoint_text)
                    update_cursor_position(changepoint)
            except ValueError:
                sg.popup_no_buttons("No valid changepoint detected.", auto_close=True, auto_close_duration=0.75)
        elif event == '<<':
            if current_trial_index > 0:
                current_trial_index -= 1
                current_trial = trial_list[current_trial_index]
                display_title = f"{main_title} - Subject: {subject} Trial: {current_trial}"
                window["-DISPLAY_TITLE-"].update(display_title)
                trial_signals = signals_dict[current_trial]
                segment_length = len(trial_signals["y"])
                x = np.arange(segment_length)
                cur_x = load_onset_for_trial()
                reset_default_settings()
                update_plots(cur_x)
            else:
                sg.popup_no_buttons("Already at the first trial.", auto_close=True, auto_close_duration=0.75)
        elif event == '>>':
            if current_trial_index < len(trial_list) - 1:
                current_trial_index += 1
                current_trial = trial_list[current_trial_index]
                display_title = f"{main_title} - Subject: {subject} Trial: {current_trial}"
                window["-DISPLAY_TITLE-"].update(display_title)
                trial_signals = signals_dict[current_trial]
                segment_length = len(trial_signals["y"])
                x = np.arange(segment_length)
                cur_x = load_onset_for_trial()
                reset_default_settings()
                update_plots(cur_x)
            else:
                sg.popup_no_buttons("Already at the last trial.", auto_close=True, auto_close_duration=0.75)
    
    window.close()