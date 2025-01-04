import argparse
import numpy as np
import pandas as pd
from fastdtw import fastdtw
import matplotlib.pyplot as plt

SUCCESS_DISTANCE_THRESHOLD = 5.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_path", type=str, default="./trajectories/reference_path.txt")
    parser.add_argument("--predicted_path", type=str, default="./trajectories/model4.txt")
    args = parser.parse_args()

    reference_path_df = pd.read_csv(args.reference_path, sep='	', header=None)
    reference_path_df.columns = ['VehicleName', 'TimeStamp', 'POS_X', 'POS_Y', 'POS_Z', 'Q_W', 'Q_X', 'Q_Y', 'Q_Z', 'Throttle', 'Steering', 'Brake', 'Gear', 'Handbrake', 'RPM', 'Speed', 'ImageFile']
    reference_path_df = reference_path_df.iloc[1: , :]

    model_path_df = pd.read_csv(args.predicted_path, sep='	', header=None)
    model_path_df.columns =  ['VehicleName', 'TimeStamp', 'POS_X', 'POS_Y', 'POS_Z', 'Q_W', 'Q_X', 'Q_Y', 'Q_Z', 'Throttle', 'Steering', 'Brake', 'Gear', 'Handbrake', 'RPM', 'Speed', 'ImageFile']
    model_path_df = model_path_df.iloc[1: , :]

    reference_path_df[['POS_X', 'POS_Y', 'POS_Z']] = reference_path_df[['POS_X', 'POS_Y', 'POS_Z']].apply(pd.to_numeric)
    model_path_df[['POS_X', 'POS_Y', 'POS_Z']] = model_path_df[['POS_X', 'POS_Y', 'POS_Z']].apply(pd.to_numeric)

    reference_path_df['-POS_X'] = reference_path_df['POS_X'].apply(lambda x: x*-1)
    reference_path_df['-POS_Y'] = reference_path_df['POS_Y'].apply(lambda x: x*-1)
    reference_path_df['-POS_Z'] = reference_path_df['POS_Z'].apply(lambda x: x*-1)

    model_path_df['-POS_X'] = model_path_df['POS_X'].apply(lambda x: x*-1)
    model_path_df['-POS_Y'] = model_path_df['POS_Y'].apply(lambda x: x*-1)
    model_path_df['-POS_Z'] = model_path_df['POS_Z'].apply(lambda x: x*-1)

    reference_path = get_locs(reference_path_df)
    model_path = get_locs(model_path_df)
    
    print(f"Reference Path Length: {path_length(reference_path).round(1)}")
    print(f"Model Path Length: {path_length(model_path).round(1)}")
    print(f"SR: {success_rate(model_path, reference_path, SUCCESS_DISTANCE_THRESHOLD)}")
    print(f"OSR: {oracle_success_rate(model_path, reference_path, SUCCESS_DISTANCE_THRESHOLD)}")
    print(f"NE: {euclidian_distance(np.array(model_path[-1]), np.array(reference_path[-1])).round(2)}")
    print(f"nDTW: {ndtw(model_path, reference_path, SUCCESS_DISTANCE_THRESHOLD).round(2)}")
    print(f"SDTW: {sdtw(model_path, reference_path, SUCCESS_DISTANCE_THRESHOLD).round(2)}")
    print(f"CLS: {cls(model_path, reference_path, SUCCESS_DISTANCE_THRESHOLD).round(2)}")
    
    plot_paths(reference_path, model_path)

# Returns locations in np array
def get_locs(df):
    first_z_loc = 0.0

    res = []
    for row in df.iterrows():
        res.append(np.array([row[1]['POS_X'], row[1]['POS_Y'], first_z_loc]))
    return np.array(res)

# Returns euclidian distance between point_a and point_b
def euclidian_distance(position_a, position_b) -> float:
    return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

# Returns Success Rate (SR) that indicates the navigation is considered successful 
# if the agent stops within success_distance_threshold of the destination
def success_rate(p_path, r_path, success_distance_threshold):
    distance_to_target = euclidian_distance(np.array(p_path[-1]), np.array(r_path[-1]))
    #print(distance_to_target)
    if distance_to_target <= success_distance_threshold:
        return 1.0
    else:
        return 0.0

# Returns Oracle Success Rate (OSR) where one navigation is considered oracle success
# if the distance between the destination and any point on the trajectory is less than success_distance_threshold
def oracle_success_rate(p_path, r_path, success_distance_threshold):
    for p in p_path:
        if euclidian_distance(p, r_path[-1]) <= success_distance_threshold:
            #print(p, r_path[-1])
            return 1.0
    return 0.0

# Returns normalized dynamic time warping metric score
def ndtw(p_path, r_path, success_distance_threshold):
    dtw_distance = fastdtw(p_path, r_path, dist=euclidian_distance)[0]
    nDTW = np.exp(-dtw_distance / (len(r_path) * success_distance_threshold))
    return nDTW

# Returns Success weighted by normalized Dynamic Time Warping 
# where one navigation is considered successful if one is returned
def sdtw(p_path, r_path, success_distance_threshold):
    return success_rate(p_path, r_path, success_distance_threshold) * ndtw(p_path, r_path, success_distance_threshold)

# Returns path length
def path_length(path):
    path_length = 0.0
    previous_position = path[0]
    for current_position in path[1:]:
        path_length += euclidian_distance(current_position, previous_position)    
        previous_position = current_position
    return path_length

# Returns path coverage score that indicates how well the reference path
# is covered by the predicted path
def path_coverage(p_path, r_path, success_distance_threshold):
    coverage = 0.0
    for r_loc in r_path:
        min_distance = float('inf')
        for p_loc in p_path:
            distance = euclidian_distance(p_loc, r_loc)
            if distance < min_distance:
                min_distance = distance
        coverage += np.exp(-min_distance / success_distance_threshold)
    return coverage / len(r_path)

# Returns the expected optimal length score given reference path’s coverage of predicted path
def epl(p_path, r_path, success_distance_threshold):
    return path_coverage(p_path, r_path, success_distance_threshold) * path_length(r_path)

# Returns length score of predicted path respect to reference path
def ls(p_path, r_path, success_distance_threshold):
    return epl(p_path, r_path, success_distance_threshold) / (epl(p_path, r_path, success_distance_threshold) + np.abs(epl(p_path, r_path, success_distance_threshold) - path_length(p_path)))

# Returns Coverage weighted by Length Score (CLS) indicates
# how closely predicted path conforms with the entire reference path 
def cls(p_path, r_path, success_distance_threshold):
    return path_coverage(p_path, r_path, success_distance_threshold) * ls(p_path, r_path, success_distance_threshold)

# Plots the reference and predicted paths
def plot_paths(reference_path, model_path):
    # Calculate path lengths
    reference_pl = path_length(reference_path)
    model_pl = path_length(model_path)
    
    reference_path = reference_path[:, [0, 1]]
    model_path = model_path[:, [0, 1]]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.plot(reference_path[:, 1], reference_path[:, 0], color='red', lw=2, label=f'Reference path (PL: {reference_pl:.2f})')
    ax.plot(model_path[:, 1], model_path[:, 0], color='blue', lw=2, label=f'RL path (PL: {model_pl:.2f})')
    
    # Set axis labels
    ax.set_xlabel('Y', fontsize=13) 
    ax.set_ylabel('X', fontsize=13) 
    ax.set_title('Improved RBG Trajectory', fontsize=13)
    
    ax.legend(fontsize=12)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()