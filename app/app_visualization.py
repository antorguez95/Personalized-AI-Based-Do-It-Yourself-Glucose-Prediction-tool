import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from typing import List
import tensorflow as tf

def cgm_data_summary_figure(id : str, sensor : str, num_blocks : int, cgm_data : np.array, cgm_timestamps : np.array, sensor_sampling_period : int,
                            pred_steps : int, N : int, prediction : np.array, pred_rmse : np.array,
                            visualization_range : str = '2 weeks', unit : str = 'mg/dL', top_threshold : int = 180, bottom_threshold : int = 70, 
                            ): 
    
    """
    This function reads the user's CGM data ant plots a summary of the data. Depending on the 
    input parameters (chosen by the user in the application), a different personalized 
    visualization will be given. 

    Args
    ----
        id : Patient ID
        sensor : Sensor used by the patient
        num_blocks : Number of blocks without sensor reading interruptions
        cgm_data : CGM values of the patient (the timeseries itself)
        cgm_timestamps : Timestamps of the CGM values
        sensor_sampling_period : Sampling period of the sensor in minutes
        pred_steps : Number of steps predicted by the AI model (dependent on previous steps, architecture, sensor, etc.)
        N : Number of samples used to predict the next hour (dependent on previous steps, architecture, sensor, patient, etc.)
        prediction : Prediction of the AI model for the next hour resulting from the call of the Keras model 
        pred_rmse : RMSE of the prediction of the AI model for the next hour resulting from the validation phase of the Keras model 
        visualization_range : Range of the CGM values data that the user wants to visualize.
        unit : Unit of the CGM values. Default is mg/dL
        top_threshold : Top threshold of the CGM values. Default is 180 mg/dL
        botom_threshold : Bottom threshold of the CGM values. Default is 70 mg/dL

    """

    # Declare a subplot of 4 x 2
    fig, axs = plt.subplots(5, 2, figsize=(15,15))

    ####### FIGURE GENERAL PARAMS #########
    # Set San Serif font and Font size 
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['font.family'] = 'Arial'

    # Avoid title and subplots overlapping 
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Set fig title
    fig.suptitle('Know more about your CGM data', fontsize=16)

    ###### FIGURE 1: TIMESERIES ######
    # The visualization range will impact on final result. 
    match visualization_range: 
        
        # 2 weeks (default)
        case '2 weeks':

            # Exctract 2 weeks of data 
            timeseries_vis = cgm_data[-(14*24*4):]
            first_date = cgm_timestamps[-(14*24*4)]
        
        # 1 week
        case '1 week':

            # Extract the last week of data 
            timeseries_vis = cgm_data[-(7*24*4):]
            first_date = cgm_timestamps[-(7*24*4)]
            
        # 1 month 
        case '1 month':

            # Extract the last month of data
            timeseries_vis = cgm_data[-(30*24*4):]
            first_date = cgm_timestamps[-(30*24*4)]

    # Plot the CGM with colours depending on the range where it is 
    # Colour the line depending on its value 
    hyper = np.ma.masked_where(timeseries_vis < top_threshold, timeseries_vis)
    hypo = np.ma.masked_where(timeseries_vis > bottom_threshold, timeseries_vis)
    normal = np.ma.masked_where((timeseries_vis < bottom_threshold) | (timeseries_vis > top_threshold), timeseries_vis)

    x_vector = np.linspace(0, len(timeseries_vis), len(timeseries_vis))

    # Compute mean and std of the timeseries
    cgm_mean = np.mean(timeseries_vis)
    cgm_std = np.std(timeseries_vis)

    # Plot the three ranges 
    axs[0,0].plot(x_vector, normal, 'o', color='green', alpha = 0.5)
    axs[0,0].plot(x_vector, hypo, 'o', color='red', alpha = 0.5)
    axs[0,0].plot(x_vector, hyper, 'o', color='yellow', alpha = 0.5)

    # X and Y labels 
    axs[0,0].set_ylabel('CGM ' + '(' + unit + ')')

    match visualization_range: 

        case '1 week': 

            # Fill between each 3 hours (3 x pred_steps samples) with the background colour with 4 different ranges
            for i in range(0, len(timeseries_vis), pred_steps*24*4):
                axs[0,0].axvspan(i, i+pred_steps*24, facecolor='lightgrey', alpha=0.25)
                axs[0,0].axvspan(i+pred_steps*24, i+pred_steps*24*2, facecolor='darkgrey', alpha=0.25)
                axs[0,0].axvspan(i+pred_steps*24*2, i+pred_steps*24*3, facecolor='grey', alpha=0.25)
                axs[0,0].axvspan(i+pred_steps*24*3, i+pred_steps*24*4, facecolor='dimgray', alpha=0.25)
            
            # Place ticks only every 24*pred_steps samples (every day) 
            axs[0,0].set_xticks(np.linspace(0, len(timeseries_vis), num=round(len(timeseries_vis)/(24*pred_steps))+1))

            # Remove x labels 
            axs[0,0].set_xticklabels([])

            # Set labels with the days with space before 
            axs[0,0].set_xticklabels(["", "Mon, 26", "Tue, 27", "Wed, 28", "Thu, 29", "Fri, 30", "Sat, 1", "Sun, 2"])
        
        case '2 weeks':

            # Fill between every day with the background colour with 4 different ranges
            for i in range(0, len(timeseries_vis), pred_steps*24*4):
                axs[0,0].axvspan(i, i+pred_steps*24, facecolor='lightgrey', alpha=0.25)
                axs[0,0].axvspan(i+pred_steps*24, i+pred_steps*24*2, facecolor='darkgrey', alpha=0.25)
                axs[0,0].axvspan(i+pred_steps*24*2, i+pred_steps*24*3, facecolor='grey', alpha=0.25)
                axs[0,0].axvspan(i+pred_steps*24*3, i+pred_steps*24*4, facecolor='dimgray', alpha=0.25)
            
            # Place ticks only every 24*pred_steps samples (every day) 
            axs[0,0].set_xticks(np.linspace(0, len(timeseries_vis), num=round(len(timeseries_vis)/(24*pred_steps))+1))

            # Set all x labels empty 
            axs[0,0].set_xticklabels([])

            # Set labels only in the first and last day
            axs[0,0].set_xticklabels(["", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

        case '1 month':

            # Fill between every day with the background colour with 4 different ranges
            for i in range(0, len(timeseries_vis), pred_steps*24*4):
                axs[0,0].axvspan(i, i+pred_steps*24, facecolor='lightgrey', alpha=0.25)
                axs[0,0].axvspan(i+pred_steps*24, i+pred_steps*24*2, facecolor='darkgrey', alpha=0.25)
                axs[0,0].axvspan(i+pred_steps*24*2, i+pred_steps*24*3, facecolor='grey', alpha=0.25)
                axs[0,0].axvspan(i+pred_steps*24*3, i+pred_steps*24*4, facecolor='dimgray', alpha=0.25)
            
            # Place ticks only every 24*pred_steps samples (every day) 
            axs[0,0].set_xticks(np.linspace(0, len(timeseries_vis), num=round(len(timeseries_vis)/(24*pred_steps))+1))

            # Set all x labels empty 
            axs[0,0].set_xticklabels([])

            # Set labels only in the first and last day
            axs[0,0].set_xticklabels(["", "M", "Tu", "W", "Th", "F", "Sa", "Su", "M", "Tu", "W", "Th", "F", "Sa", "Su", 
                                    "M", "Tu", "W", "Th", "F", "Sa", "Su", "M", "Tu", "W", "Th", "F", "Sa", "Su", "M", "Tu"])

    # Draw dashed line in the top and bottom threshold
    axs[0,0].axhline(y=top_threshold, xmin=0, xmax=len(timeseries_vis),  color='red', linestyle='--', alpha=0.40)
    axs[0,0].axhline(y=bottom_threshold, xmin=0, xmax=len(timeseries_vis), color='red', linestyle='--', alpha=0.40)

    # Put the x labels on the left side 
    axs[0,0].tick_params(axis='x', labelsize=9, labelbottom=True, labeltop=False, bottom=True, labelright=True, labelleft=False)

    # Set x limits
    axs[0,0].set_xlim([0, timeseries_vis.shape[0]])

    ###### FIGURE 2: HISTOGRAM ######

    # Set background depends on the range of the CGM
    axs[0,1].axvspan(0, 54, facecolor='darkred', alpha=0.35)
    axs[0,1].axvspan(54, 70, facecolor='red', alpha=0.35)
    axs[0,1].axvspan(70, 180, facecolor='green', alpha=0.35)
    axs[0,1].axvspan(180, 250, facecolor='yellow', alpha=0.35)
    axs[0,1].axvspan(250, 500, facecolor='orange', alpha=0.35)

    # Plot a histogram than represents a normal distribution with the same mean and std than the timeseries. Plot only the edge and the bins are transparent
    axs[0,1].hist(np.random.normal(np.mean(timeseries_vis)-30, np.std(timeseries_vis)/2, len(timeseries_vis)),
                bins=50, color='black',  histtype='step', linewidth=1.5, alpha=0.75, label='The same amount of readings in a non-diabetic person') # Simulates a non-diabetic distribution, but needs to be updated

    # Plot the histogram of the timeseries
    axs[0,1].hist(timeseries_vis, bins=100, label='All your glucose readings', color='steelblue', alpha=1)

    # Set legend fontsize
    axs[0,1].legend(fontsize=10)

    # Set x and y labels
    axs[0,1].set_xlabel('CGM ' + '(' + unit + ')')
    axs[0,1].set_ylabel('Number of samples')

    # Move the y label a bit to the right 
    axs[0,1].yaxis.set_label_coords(-0.05, 0.5)

    # Set fontsize to ytick labels 
    axs[0,1].tick_params(axis='y', labelsize=10)

    # Set x limits
    axs[0,1].set_xlim([0, 500])

    # Delete [1,0]
    fig.delaxes(axs[1,0])

    ###### FIGURE 3: BAR DIAGRAM WITH THE SAMPLES ON EACH RANGE ######

    # Center the axis
    axs[1,1].set_position([0.35, 0.60, 0.400, 0.17])

    # Copmute the subranges in the CGM data 
    if unit == 'mg/dL':
        severely_hypo = timeseries_vis[timeseries_vis < 54]
        hypo = timeseries_vis[(timeseries_vis >= 54) & (timeseries_vis < 70)]
        normal = timeseries_vis[(timeseries_vis >= 70) & (timeseries_vis <= 180)]
        hyper = timeseries_vis[(timeseries_vis > 180) & (timeseries_vis <= 250)]
        severely_hyper = timeseries_vis[timeseries_vis > 250]

    elif unit == 'mmol/L':
        ############# revisar
        severely_hypo = timeseries_vis[timeseries_vis < 3.0]
        hypo = timeseries_vis[(timeseries_vis >= 3.0) & (timeseries_vis < 3.9)]
        normal = timeseries_vis[(timeseries_vis >= 3.9) & (timeseries_vis <= 10.0)]
        hyper = timeseries_vis[(timeseries_vis > 10.0) & (timeseries_vis <= 13.9)]
        severely_hyper = timeseries_vis[timeseries_vis > 13.9]

    # Create an stacked bar plot from severe hypo to severe hyper
    bar_width = 0.35
    labels = ['< 54', '54 - 70', '70 - 180', '180 - 250', '> 250']

    axs[1,1].bar(['Time in ranges\n(mg/dL)'],[severely_hypo.shape[0]], width = bar_width, color='darkred', alpha=0.5, align='center', label='< 54', edgecolor='black')
    axs[1,1].bar(['Time in ranges\n(mg/dL)'],[hypo.shape[0]], bottom=[severely_hypo.shape[0]], width = bar_width, color='red', alpha=0.5, align='center', label='54 - 70', edgecolor='black')
    axs[1,1].bar(['Time in ranges\n(mg/dL)'],[normal.shape[0]], bottom=[severely_hypo.shape[0] + hypo.shape[0]], width = bar_width, color='green', alpha=0.5, align='center', label='70 - 180', edgecolor='black')
    axs[1,1].bar(['Time in ranges\n(mg/dL)'],[hyper.shape[0]], bottom=[severely_hypo.shape[0] + hypo.shape[0] + normal.shape[0]], width = bar_width, color='yellow', alpha=0.5, align='center', label='180 - 250', edgecolor='black')
    axs[1,1].bar(['Time in ranges\n(mg/dL)'],[severely_hyper.shape[0]], bottom=[severely_hypo.shape[0] + hypo.shape[0] + normal.shape[0] + hyper.shape[0]], width = bar_width, color='orange', alpha=0.5, align='center', label='> 250', edgecolor='black')

    # Set x lim 
    axs[1,1].set_xlim([-0.5, 4])

    # Remove Y ticks and labels 
    axs[1,1].set_yticklabels([])
    axs[1,1].set_yticks([])

    # Remove the box around the plot
    axs[1,1].spines['top'].set_visible(False)  
    axs[1,1].spines['right'].set_visible(False)
    axs[1,1].spines['left'].set_visible(False)

    # Compute time in range (TiR), Time above the range and time below the range 
    # Get the number of total samples 
    total_samples = len(timeseries_vis)

    # Get the TiR (number of samples in normal glycaemic level)
    TiR_samples = len(timeseries_vis[(timeseries_vis >= bottom_threshold) & (timeseries_vis <= top_threshold)])

    # Get the number of samples in the range above normal range 
    above_range_samples = len(timeseries_vis[timeseries_vis > top_threshold])

    # Get the number of samples in the range below normal range
    below_range_samples = len(timeseries_vis[timeseries_vis < bottom_threshold])

    # Compute the percentage of TiR
    TiR_percentage = (TiR_samples / total_samples) * 100
    above_range_percentage = (above_range_samples / total_samples) * 100
    below_range_percentage = (below_range_samples / total_samples) * 100

    # Generate text with only 2 decimal positions 
    tir_text = "{:.2f}%".format(TiR_percentage)
    above_range_text = "{:.2f}%".format(above_range_percentage)
    below_range_text = "{:.2f}%".format(below_range_percentage)

    ###### "FIGURE" 5: ONLY TEXT REGARDING THE PATIENT ######

    # Extract datetimes from the timestamps 
    first_hour = str(cgm_timestamps[0].hour) + ":" + str(cgm_timestamps[0].minute)
    first_day = str(cgm_timestamps[0].day) + '-' + str(cgm_timestamps[0].month)
    first_year = str(cgm_timestamps[0].year)
    last_hour = str(cgm_timestamps[-1].hour) + ":" + str(cgm_timestamps[-1].minute)
    last_day = str(cgm_timestamps[-1].day) + '-' + str(cgm_timestamps[-1].month)
    last_year = str(cgm_timestamps[-1].year)

    axs[2,0].text(0, 0.6, '- Sensor: ' + sensor , fontsize=16)
    axs[2,0].text(0.1, 0.5, 'Time between consecutive glucose readings: ', fontsize=16, style='italic')
    axs[2,0].text(0.77, 0.5, str(sensor_sampling_period)+" minutes", fontsize=16, weight='bold', color='steelblue')

    # Dates 
    axs[2,0].text(0, 0.35, '- Data from: ', fontsize=16)
    axs[2,0].text(0.2, 0.35, first_day+'-'+first_year , fontsize=16, style='italic')
    axs[2,0].text(0.4, 0.35, 'to: ', fontsize=16)
    axs[2,0].text(0.47, 0.35, last_day+'-'+last_year, fontsize=16, style='italic')

    axs[2,0].text(0, 0.25, '- Number of blocks without sensor reading interruptions: ', fontsize=16)
    axs[2,0].text(0.91, 0.25, str(num_blocks), fontsize=16, weight='bold')

    axs[2,0].axis('off')

    ###### "FIGURE" 6: ONLY TEXT REGARDING THE PATIENT (AGAIN) ######

    # Max autocorrelation index
    acf = sm.tsa.acf(timeseries_vis, nlags=300, fft=False, alpha=None, missing='none')
    acf = acf[30:]
    peak_idx = np.argmax(acf) + 30

    # Set the text with the patient ID
    if unit == 'mg/dL':

        axs[2,1].text(0.1, 0.6, 'Hypoglycaemia readings (< 70 mg/dL): ', fontsize=12)
        axs[2,1].text(0.53, 0.6, str(hypo.shape[0]+severely_hypo.shape[0]), fontsize=12, weight='bold')
        axs[2,1].text(0.1, 0.5, 'Normal range readings(between 70 and 180 mg/dL): ', fontsize=12)
        axs[2,1].text(0.68, 0.5, str(normal.shape[0]), fontsize=12, weight='bold')
        axs[2,1].text(0.1, 0.4, 'Hyperglycaemia range readings (> 250 mg/dL): ', fontsize=12)
        axs[2,1].text(0.62, 0.4, str(hyper.shape[0]+severely_hyper.shape[0]), fontsize=12, weight='bold')

    elif unit == 'mmol/L':
        axs[2,1].text(0.1, 0.6, 'Hypoglycaemia readings (< 3.9 mmol/L): ', fontsize=12)
        axs[2,1].text(0.57, 0.6, str(hypo.shape[0]+severely_hypo.shape[0]), fontsize=12, weight='bold')
        axs[2,1].text(0.1, 0.5, 'Normal range readings(between 3.9 and 10.0 mmol/L): ', fontsize=12)
        axs[2,1].text(0.72, 0.5, str(normal.shape[0]), fontsize=12, weight='bold')
        axs[2,1].text(0.1, 0.4, 'Hyperglycaemia range readings (> 10.0 mmol/L): ', fontsize=12)
        axs[2,1].text(0.66, 0.4, str(hyper.shape[0]+severely_hyper.shape[0]), fontsize=12, weight='bold')

    axs[2,1].text(0.1, 0.25, 'The maximum ', fontsize=12)
    axs[2,1].text(0.27, 0.25, 'Autocorrelation index', fontsize=12, weight='bold', style='italic')
    axs[2,1].text(0.53, 0.25, 'of your data is', fontsize=12)
    axs[2,1].text(0.70, 0.25, str(peak_idx), fontsize=12, weight='bold')
    axs[2,1].text(0.1, 0.1, 'This means that the most useful information to train your AI-personalized model\n is contained every ', fontsize=12,)
    axs[2,1].text(0.31, 0.1, str((round(N*sensor_sampling_period/60)))+' hours', fontsize=12, weight='bold')
    axs[2,1].text(0.41, 0.1, '!', fontsize=12)
    axs[2,1].axis('off')

    # Remove the last subplots and center the ax[3,0]
    fig.delaxes(axs[3,1])

    ###### FIGURE 7: LAST 144 SAMPLES, PLUS THE PREDICTION ######

    # Store the last 144 samples of timestamps 
    prediction_timestamps = cgm_timestamps[-144:]

    # Extract the strings with the timestamps to print 
    first_daymonth = str(prediction_timestamps[0].day) + '-' + str(prediction_timestamps[0].month)
    first_hour_minute = str( prediction_timestamps[0].hour) + ':' + str(prediction_timestamps[0].minute)
    last_daymonth = str(prediction_timestamps[-1].day) + '-' + str(prediction_timestamps[-1].month)
    last_hour_minute = str( prediction_timestamps[-1].hour) + ':' + str(prediction_timestamps[-1].minute)

    # Plot the last 144 samples
    axs[3,0].plot(color = 'darkkhaki', alpha = 0.5)

    # Set x and y labels
    axs[3,0].set_ylabel('CGM ' + '(' + unit + ')')

    # Extract the last window of real data used to predict the next hour 
    real_data = timeseries_vis[-144:]

    # Set background depends on the range of the CGM
    prediction = np.array([60, 105, 120, 131])# This must be the sequence obtained by the model

    axs[3,0].plot(real_data, label = 'Last samples')
    axs[3,0].plot(np.linspace(N-1,N-1+round(pred_steps), round(pred_steps)), prediction, color = 'red', label = 'AI prediction', alpha=1)

    # Fill with the prediction with the RMSE values 
    axs[3,0].fill_between(np.linspace(N-1,N-1+round(pred_steps), round(pred_steps), dtype=int), prediction-pred_rmse, prediction+pred_rmse, color='red', alpha=0.2)

    # Set title 
    axs[3,0].set_title('60 minutes AI prediction using your last 36 hours of CGM data', fontsize=12)

    # Remove X ticks labels 
    axs[3,0].set_xticklabels([])

    # Set background of the first 144 samples of one colour depending on the range of the CGM
    axs[3,0].axhspan(0, 54, xmin=0, xmax=143/148, facecolor='red', alpha=0.25)
    axs[3,0].axhspan(54, 70, xmin=0, xmax=143/148, facecolor='yellow', alpha=0.25)
    axs[3,0].axhspan(70, 180, xmin=0, xmax=143/148, facecolor='green', alpha=0.25)
    axs[3,0].axhspan(180, 250, xmin=0, xmax=143/148, facecolor='yellow', alpha=0.25)
    axs[3,0].axhspan(250, 400, xmin=0, xmax=143/148, facecolor='red', alpha=0.25)

    # Set a different bakground for the AI prediction 
    axs[3,0].axhspan(0, 400, xmin=143/148, xmax=1, facecolor='teal', alpha=0.20)

    # Vertical line to separate real data from prediction 
    axs[3,0].axvline(x=143, ymin=0, ymax=400, color='darkcyan', linestyle='--', alpha=0.75)

    # Set horizontal line in the second part of the plot with the different CGM ranges 
    axs[3,0].axhline(y=54, xmin=143/148, xmax=1,  color='red', linestyle='--', alpha=0.40)
    axs[3,0].axhline(y=70, xmin=143/148, xmax=1, color='yellow', linestyle='--', alpha=0.40)
    axs[3,0].axhline(y=180, xmin=143/148, xmax=1, color='yellow', linestyle='--', alpha=0.40)
    axs[3,0].axhline(y=250, xmin=143/148, xmax=1, color='red', linestyle='--', alpha=0.40)

    # Set X labels 
    axs[3,0].set_xticklabels([first_daymonth+"\n"+first_hour_minute, "", "", "", "", "", "", last_daymonth+"\n"+last_hour_minute])

    # Set pad in x labels 
    axs[3,0].tick_params(axis='x', pad=10)

    # Remove white spaces in the x and y axis
    axs[3,0].set_xlim([0, 148])
    axs[3,0].set_ylim([0, 400])

    ###### FIGURE 7: FIGURE 6 ZOOM ######
    # Plot the last 144 samples
    axs[4,0].plot(color = 'k', alpha = 0.5)

    # Set x and y labels
    axs[4,0].set_ylabel('CGM ' + '(' + unit + ')')

    # Remove X ticks labels
    axs[4,0].set_xticklabels([])

    axs[4,0].axhspan(0, 54, xmin=0, xmax=9.75/14, facecolor='red', alpha=0.18)
    axs[4,0].axhspan(54, 70, xmin=0, xmax=9.75/14, facecolor='yellow', alpha=0.18)
    axs[4,0].axhspan(70, 180, xmin=0, xmax=9.75/14, facecolor='green', alpha=0.18)
    axs[4,0].axhspan(180, 250, xmin=0, xmax=9.75/14, facecolor='yellow', alpha=0.18)
    axs[4,0].axhspan(250, 400, xmin=0, xmax=9.75/14, facecolor='red', alpha=0.18)

    # Vertical line to separate real data from prediction 
    axs[4,0].axvline(x=10, ymin=0, ymax=400, color='darkcyan', linestyle='--', alpha=0.75)

    # Extract the last window of real data used to predict the next hour 
    real_data = timeseries_vis[-10:]

    # Samples to make zoom
    N_zoom = 10

    axs[4,0].plot(np.linspace(0, N_zoom-1, N_zoom, dtype=int), real_data, marker='o', label = 'Last samples')
    axs[4,0].plot(np.linspace(N_zoom, N_zoom+pred_steps, pred_steps, dtype=int), prediction, color = 'red', marker='o',label = 'AI prediction', alpha=1)

    # Fill with the prediction with the RMSE values 
    axs[4,0].fill_between(np.linspace(N_zoom, N_zoom+pred_steps, pred_steps, dtype=int), prediction-pred_rmse, prediction+pred_rmse, color='red', alpha=0.2)

    # Set x labels 
    axs[4,0].set_xticklabels([first_daymonth+"\n"+first_hour_minute, "", "", "", "", "", "", "", "", last_daymonth+"\n"+last_hour_minute])

    # Set pad in x labels 
    axs[4,0].tick_params(axis='x', pad=10)

    # Add text in the first and last sample with data and time 
    axs[4,0].text(12.5, 300, 'Your AI prediction\nfor the next hour', fontsize=12, weight='bold', horizontalalignment='center', verticalalignment='center')#, bbox=dict(facecolor='white', alpha=0.5)

    # Add text that indicates the user the prediction horizons
    axs[4,0].text(11.5, -60, "in 30'", fontsize=12, weight='bold')
    axs[4,0].text(13.5, -60, "in 60'", fontsize=12, weight='bold')

    # Set a different bakground for the AI prediction 
    axs[4,0].axhspan(0, 400, xmin=9.75/14, xmax=1, facecolor='teal', alpha=0.20)

    # Set horizontal line in the second part of the plot with the different CGM ranges 
    axs[4,0].axhline(y=54, xmin=9.75/14, xmax=1,  color='red', linestyle='--', alpha=0.40)
    axs[4,0].axhline(y=70, xmin=9.75/14, xmax=1, color='yellow', linestyle='--', alpha=0.40)
    axs[4,0].axhline(y=180, xmin=9.75/14, xmax=1, color='yellow', linestyle='--', alpha=0.40)
    axs[4,0].axhline(y=250, xmin=9.75/14, xmax=1, color='red', linestyle='--', alpha=0.40)

    # Set title 
    axs[4,0].set_title('Zoom in your last 2:30 hours', fontsize=12)

    # Set Y limit between 0 and 400
    axs[4,0].set_ylim([0, 400])

    # Set one tick per sample 
    axs[4,0].set_xticks(np.linspace(0, N_zoom+pred_steps, N_zoom+pred_steps, dtype=int))

    # Remove ax[4,1]
    fig.delaxes(axs[4,1])

    # Center and set size 
    axs[3,0].set_position([0.300, 0.265, 0.400, 0.15])
    axs[4,0].set_position([0.300, 0.075, 0.400, 0.15])

    plt.text(1, 0.83, 'Useful information about YOU to understand your personalized AI-based model', fontsize=16, horizontalalignment='center', verticalalignment='center', transform=axs[2,0].transAxes)

    # TiR texts for ax[1,0]
    plt.text(0.35, 0.55, 'TiR: ' + tir_text, fontsize=14, weight='bold', transform=axs[1,1].transAxes)
    plt.text(0.35, 0.45, 'Time above Range: ' + above_range_text, fontsize=14, weight='bold', transform=axs[1,1].transAxes)
    plt.text(0.35, 0.35, 'Time below Range: ' + below_range_text, fontsize=14, weight='bold', transform=axs[1,1].transAxes)
    plt.text(0.35, 0.05, 'Average Glucose: ' + str(round(cgm_mean, 2)) + ' ± ' + str(round(cgm_std, 2)) + ' ' + unit, fontsize=14, weight='bold', transform=axs[1,1].transAxes)

    plt.savefig('app_CGM_' + visualization_range +  '_analysis_and_prediction.svg', format='svg', dpi=1200)


def get_prediction_graphic(X : np.array, X_norm : np.array, predicted_points : int, X_times : np.array, rmse : List, unit : str, model : tf.keras.Model, 
                           N : int = 96, step : int =  1): 
    """
    This function generates and save two plots taking the last day available of a given 
    user, if it uploads enough and reliable data to generate a CGM prediction until the 
    next 30'. The first plot shows the last 96 samples of the user's data and the AI
    (one day), together with the prediction. The second subplot is a zoommed version of
    the first one

    Args: 
    -----
        X : un-normalized training vector to perform de-normalization
        X_norm : vector from which the last day of the user's data (96 samples) will be extracted and denormalized
        predicted_points : Number of samples to predict. Depends on the PH and the sensor sampling period
        X_times : Timestamps from which the timestamps of the last day are extracted
        rmse : List with the RMSE corresponding to the user's 4-fold cross-validation for each predicted instant. Its lengths should be equal to the number of predicted points
        unit : Unit of the CGM data. Can be 'mg/dL' or 'mmol/L'
        N : input window length. Default: 96 (see paper)
        step : step to generate the dataset to train de models. Default: 1 (see paper)
        
    """

    # Extract timestamps to plot to the user in the unzoommed and zoommed plot 
    # Unzommed plot
    first_daymonth = str(X_times[-1][0])[5:10]
    first_hour_minute = str(X_times[-1][0])[11:16]
    first_daymonth, first_hour_minute

    second_daymonth = str(X_times[-1][31])[5:10]
    second_hour_minute = str(X_times[-1][31])[11:16]

    third_daymonth = str(X_times[-1][63])[5:10]
    third_hour_minute = str(X_times[-1][63])[11:16]

    last_daymonth = str(X_times[-1][-1])[5:10]
    last_hour_minute = str(X_times[-1][-1])[11:16]
    last_daymonth, last_hour_minute

    # Zommed plot 
    nine_daymonth = str(X_times[-1][-2])[5:10]
    nine_hour_minute = str(X_times[-1][-2])[11:16]

    eight_daymonth = str(X_times[-1][-3])[5:10]
    eight_hour_minute = str(X_times[-1][-3])[11:16]

    seven_daymonth = str(X_times[-1][-4])[5:10]
    seven_hour_minute = str(X_times[-1][-4])[11:16]

    six_daymonth = str(X_times[-1][-5])[5:10]
    six_hour_minute = str(X_times[-1][-5])[11:16]

    five_daymonth = str(X_times[-1][-6])[5:10]
    five_hour_minute = str(X_times[-1][-6])[11:16]

    four_daymonth = str(X_times[-1][-7])[5:10]
    four_hour_minute = str(X_times[-1][-7])[11:16]

    three_daymonth = str(X_times[-1][-8])[5:10]
    three_hour_minute = str(X_times[-1][-8])[11:16]

    two_daymonth = str(X_times[-1][-9])[5:10]
    two_hour_minute = str(X_times[-1][-9])[11:16]

    one_daymonth = str(X_times[-1][-10])[5:10]
    one_hour_minute = str(X_times[-1][-10])[11:16]

    # Load the last day of the data
    last_day = X_norm[-96:,0,:]

    # Make the prediction
    prediction = model.predict(last_day[np.newaxis,:,:])

    # Denorm last_day and prediction (assuming min-max normalization)
    last_day_denorm = last_day*(np.max(X) - np.min(X)) + np.min(X)
    last_day_denorm = last_day_denorm[:,0]
    prediction_denorm = prediction*(np.max(X) - np.min(X)) + np.min(X)
    prediction_denorm = prediction_denorm[0]

    ######################## FIGURES OF PREDICTION (UNZOOMMED AND ZOOMMED) ############################
    fig, axs= plt.subplots(2, 1, figsize=(15,15))

    ####### FIGURE GENERAL PARAMS #########
    # Set San Serif font and Font size 
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['font.family'] = 'Arial'

    # Plot the last 144 samples
    axs[0].plot(color = 'darkkhaki', alpha = 0.5)

    # Set x and y labels
    axs[0].set_ylabel('CGM ' + '(' + unit + ')')

    axs[0].plot(last_day_denorm, label = 'Last samples', marker='o')
    axs[0].plot(np.linspace(N-1,N-1+round(predicted_points), round(predicted_points)), prediction_denorm, color = 'red', marker='o', label = 'AI prediction', alpha=1)

    # Fill with the prediction with the RMSE values 
    axs[0].fill_between(np.linspace(N-1,N-1+round(predicted_points), round(predicted_points), dtype=int), prediction_denorm-rmse, prediction_denorm+rmse, color='red', alpha=0.2)

    # Set title 
    axs[0].set_title('30 minutes AI prediction using your last 24 hours of CGM data', fontsize=12)

    # Remove X ticks labels 
    axs[0].set_xticklabels([])

    # Set background of the first 96 samples of one colour depending on the range of the CGM
    axs[0].axhspan(0, 54, xmin=0, xmax=95/98, facecolor='red', alpha=0.25)
    axs[0].axhspan(54, 70, xmin=0, xmax=95/98, facecolor='yellow', alpha=0.25)
    axs[0].axhspan(70, 180, xmin=0, xmax=95/98, facecolor='green', alpha=0.25)
    axs[0].axhspan(180, 250, xmin=0, xmax=95/98, facecolor='yellow', alpha=0.25)
    axs[0].axhspan(250, 400, xmin=0, xmax=95/98, facecolor='red', alpha=0.25)

    # Set a different bakground for the AI prediction 
    axs[0].axhspan(0, 400, xmin=95/98, xmax=1, facecolor='teal', alpha=0.20)

    # Vertical line to separate real data from prediction 
    axs[0].axvline(x=95, ymin=0, ymax=400, color='darkcyan', linestyle='--', alpha=0.75)

    # Set horizontal line in the second part of the plot with the different CGM ranges 
    axs[0].axhline(y=54, xmin=95/98, xmax=1,  color='red', linestyle='--', alpha=0.40)
    axs[0].axhline(y=70, xmin=95/98, xmax=1, color='yellow', linestyle='--', alpha=0.40)
    axs[0].axhline(y=180, xmin=95/98, xmax=1, color='yellow', linestyle='--', alpha=0.40)
    axs[0].axhline(y=250, xmin=95/98, xmax=1, color='red', linestyle='--', alpha=0.40)

    # Add x ticks positions
    axs[0].set_xticks(np.linspace(0, 95, 4, dtype=int))

    # Set X ticks labels 
    axs[0].set_xticklabels([first_daymonth+"\n"+first_hour_minute, second_daymonth+"\n"+second_hour_minute,third_daymonth+"\n"+third_hour_minute, last_daymonth+"\n"+last_hour_minute])

    # Set pad in x labels 
    axs[0].tick_params(axis='x', pad=10)

    # Remove white spaces in the x and y axis
    axs[0].set_xlim([0, 98])
    axs[0].set_ylim([0, 400])

    # Plot the last DAY (96 SAMPLES)
    axs[1].plot(color = 'k', alpha = 0.5)

    # Set x and y labels
    axs[1].set_ylabel('CGM ' + '(' + unit + ')')

    # Remove X ticks labels
    axs[1].set_xticklabels([])

    axs[1].axhspan(0, 54, xmin=0, xmax=9.65/12, facecolor='red', alpha=0.18)
    axs[1].axhspan(54, 70, xmin=0, xmax=9.65/12, facecolor='yellow', alpha=0.18)
    axs[1].axhspan(70, 180, xmin=0, xmax=9.65/12, facecolor='green', alpha=0.18)
    axs[1].axhspan(180, 250, xmin=0, xmax=9.65/12, facecolor='yellow', alpha=0.18)
    axs[1].axhspan(250, 400, xmin=0, xmax=9.65/12, facecolor='red', alpha=0.18)

    # Vertical line to separate real data from prediction 
    axs[1].axvline(x=10, ymin=0, ymax=400, color='darkcyan', linestyle='--', alpha=0.75)

    # Extract the last window of real data used to predict the next hour 
    # Samples to make zoom
    N_zoom = 10

    axs[1].plot(np.linspace(0, N_zoom-1, N_zoom, dtype=int), last_day_denorm[-10:], marker='o', label = 'Last samples')
    axs[1].plot(np.linspace(N_zoom, N_zoom+predicted_points, predicted_points, dtype=int), prediction_denorm, color = 'red', marker='o',label = 'AI prediction', alpha=1)

    # Fill with the prediction with the RMSE values 
    axs[1].fill_between(np.linspace(N_zoom, N_zoom+predicted_points, predicted_points, dtype=int), prediction_denorm-rmse, prediction_denorm+rmse, color='red', alpha=0.2)

    # Set x labels 
    axs[1].set_xticklabels([one_daymonth+"\n"+one_hour_minute, two_daymonth+"\n"+two_hour_minute, three_daymonth+"\n"+three_hour_minute,
                            four_daymonth+"\n"+four_hour_minute, five_daymonth+"\n"+five_hour_minute, six_daymonth+"\n"+six_hour_minute,
                            seven_daymonth+"\n"+seven_hour_minute,eight_daymonth+"\n"+eight_hour_minute,
                            nine_daymonth+"\n"+nine_hour_minute, last_daymonth+"\n"+last_hour_minute, "in 15'", "in 30'"])

    # Set pad in x labels 
    axs[1].tick_params(axis='x', pad=10)

    # Add text in the first and last sample with data and time 
    axs[1].text(11.25, 415, "Your AI prediction\nfor the next 30'", fontsize=12, weight='bold', horizontalalignment='center', verticalalignment='center')#, bbox=dict(facecolor='white', alpha=0.5)


    # Set a different bakground for the AI prediction 
    axs[1].axhspan(0, 400, xmin=9.65/12, xmax=1, facecolor='teal', alpha=0.20)

    # Set horizontal line in the second part of the plot with the different CGM ranges 
    axs[1].axhline(y=54, xmin=9.65/12, xmax=1,  color='red', linestyle='--', alpha=0.40)
    axs[1].axhline(y=70, xmin=9.65/12, xmax=1, color='yellow', linestyle='--', alpha=0.40)
    axs[1].axhline(y=180, xmin=9.65/12, xmax=1, color='yellow', linestyle='--', alpha=0.40)
    axs[1].axhline(y=250, xmin=9.65/12, xmax=1, color='red', linestyle='--', alpha=0.40)

    # Set title 
    axs[1].set_title("Zoom in your last 2h 30'", fontsize=12)

    # Set Y limit between 0 and 400
    axs[1].set_ylim([0, 400])

    # Set one tick per sample 
    axs[1].set_xticks(np.linspace(0, N_zoom+predicted_points, N_zoom+predicted_points, dtype=int))

    # Save figure
    plt.savefig('your_last_prediction.png', dpi=300, bbox_inches='tight') 