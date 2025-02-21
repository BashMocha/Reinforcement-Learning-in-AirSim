import airsim
import numpy as np
import torch
import time
from collections import deque
from Deep_QNetwork import DQN
from utils import get_state_from_simulator, map_action_to_controls

# Ensure reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the number of actions (should match what was used during training)
action_size = 5

# Path to your saved model weights
MODEL_PATH = './trajectories/best_dqn_model4.pth'

# Waypoints (same as used during training, adjust if needed)
waypoints = [
    airsim.Vector3r(-3.8146975356312396e-08, 7.445784285664558e-05, -0.5857376456260681),  # Starting point
    airsim.Vector3r(21.836427688598633, -0.024445464834570885, -0.5837180614471436),       # White car
    airsim.Vector3r(51.68717575073242, -0.5642141103744507, -0.584981381893158),           # Red car
    airsim.Vector3r(80.388427734375, -1.1560953855514526, -0.5853434801101685),            # Near end of the street
    airsim.Vector3r(119.025634765625, -1.2211841344833374, -0.5852082371711731),           # End of the street
    
    airsim.Vector3r(128.579345703125, -12.327738761901855, -0.58383709192276),           
    airsim.Vector3r(129.1442108154297, -26.571720123291016, -0.5849749445915222),           
    airsim.Vector3r(128.38992309570312, -46.184783935546875, -0.5853114724159241),          
    airsim.Vector3r(128.2161407470703, -66.68011474609375, -0.5851855278015137),  
]

def load_model():
    # Initialize the model
    model = DQN(action_size).to(device)
    # Load the saved state dictionary
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

def main():
    # Connect to the AirSim simulator
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()

    print("Model loaded. Starting autonomous driving...")

    model = load_model()

    # Initialize the state stack with a deque (for stacking frames)
    state_stack = deque(maxlen=4)

    current_waypoint_index = 0  # Start with the first waypoint

    try:
        # Reset the environment
        client.reset()
        client.enableApiControl(True)
        time.sleep(0.1)
        car_controls = airsim.CarControls()
        client.startRecording()

        # Get the initial state
        state_image, relative_position = get_state_from_simulator(client, waypoints[current_waypoint_index])

        # Initialize state stack
        for _ in range(4):
            state_stack.append(state_image)
        stacked_state = np.array(state_stack)

        while True:
            if not client.isRecording():
                client.startRecording()

            collision_info = client.simGetCollisionInfo()
            if collision_info.has_collided:
                print("Collision detected. Stopping...")
                client.reset()
                client.enableApiControl(True)
                time.sleep(0.1)
                car_controls = airsim.CarControls()
                client.stopRecording()
                continue
                # Get the initial state
                #state_image, relative_position = get_state_from_simulator(client, waypoints[current_waypoint_index])


            # Convert state to tensor
            state_tensor = torch.FloatTensor(stacked_state).unsqueeze(0).to(device)
            relative_position_tensor = torch.FloatTensor(relative_position).unsqueeze(0).to(device)

            # Get action from the model
            with torch.no_grad():
                q_values = model(state_tensor, relative_position_tensor)
                action = torch.argmax(q_values).item()

            # Map action to controls
            car_controls = map_action_to_controls(action)

            # Send control to the car
            client.setCarControls(car_controls)

            # Get next state
            next_state_image, next_relative_position = get_state_from_simulator(client, waypoints[current_waypoint_index])

            # Update state stack
            state_stack.append(next_state_image)
            stacked_state = np.array(state_stack)

            # Update relative position
            relative_position = next_relative_position

            # Check if waypoint is reached
            car_state = client.getCarState()
            car_position = car_state.kinematics_estimated.position

            for w in waypoints:
                min_distance = float('-inf')
                distance = np.sqrt(
                    (car_position.x_val - w.x_val) ** 2 +
                    (car_position.y_val - w.y_val) ** 2
                )
                if min_distance < distance:
                    min_distance = distance

            print(min_distance)

            if min_distance <= 5.0:
                print("Finished")
                break
            
            """
            target_waypoint = waypoints[current_waypoint_index]
            distance = np.sqrt(
                (car_position.x_val - target_waypoint.x_val) ** 2 +
                (car_position.y_val - target_waypoint.y_val) ** 2
            )

            # Define a threshold to consider the waypoint "reached"
            threshold_distance = 5.0

            if distance <= threshold_distance:
                current_waypoint_index += 1
                print(f"Reached waypoint {current_waypoint_index}")
                if current_waypoint_index >= len(waypoints):
                    print("All waypoints reached. Mission accomplished!")
                    break  # Exit the loop when all waypoints are reached
                else:
                    print(f"Heading to waypoint {current_waypoint_index + 1}")
            """
            # Small delay to simulate real-time control
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Autonomous driving interrupted.")
    finally:
        # Reset controls and disable API control
        car_controls.brake = 1.0
        client.setCarControls(car_controls)
        client.enableApiControl(False)
        client.stopRecording()
        print("Autonomous driving stopped.")

if __name__ == '__main__':
    main()