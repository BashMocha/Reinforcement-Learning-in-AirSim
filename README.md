# Autonomous Driving in AirSim by Reinforcement Learning

> Autonomous driving for path tracking in AirSim by applying the Deep-Q Network algorithm. See the [documentation](https://github.com/BashMocha/Reinforcement-Learning-in-AirSim/blob/master/docs/Reinforcement_Learning_on_Autonomous_Vehicles.pdf) for a detailed explanation.
<br>

https://github.com/user-attachments/assets/263b3539-5bf9-4b35-b96e-7c6c5ffb3027

---

## Simulation Environment
[AirSim's](https://github.com/microsoft/AirSim/) neighborhood environment [(AirSimNH)](https://github.com/microsoft/AirSim/releases/tag/v1.8.1-windows) is utilized due to its simple structure, making it well-suited for path-tracking algorithms. The project has been tested on Windows 11, but not on Linux; therefore, running the scripts on a Windows operating system is recommended.

The simulation environment can be downloaded from our release page. To configure the environment, move the `settings.json` file to the `%userprofile%\Documents\AirSim` and run the `AirSimNH\WindowsNoEditor\AirsimNH.exe` from the environment folder. Once the simulation is running, a Python script containing the following lines can connect to the environment for API calls:

```python
    import airsim

    client = airsim.CarClient()
    client.confirmConnection()     
    client.enableApiControl(True)  
    car_controls = airsim.CarControls()
```

## Conda environment
Run the following commands to create and activate the Conda environment with the required dependencies
```
  conda env create -f environment.yml
  conda activate airsim
```

## Usage

#### Training
Once the simulation environment is running, execute the train.py script to connect to it. By default, the agent is trained for 1000 epochs, but this can be adjusted using the episodes parameter in the script.
```
  python train.py
```

Since this is a path-tracking application, the reference route is specified by the `waypoints` list in the script.
```python
waypoints = [
        # Starting Street
        airsim.Vector3r(-3.8146975356312396e-08, 7.445784285664558e-05, -0.5857376456260681),  # Starting point
        airsim.Vector3r(21.836427688598633, -0.024445464834570885, -0.5837180614471436),       # White car
        airsim.Vector3r(51.68717575073242, -0.5642141103744507, -0.584981381893158),           # Red car
        airsim.Vector3r(80.388427734375, -1.1560953855514526, -0.5853434801101685),            # Near end of the street
        ]
```

Run custom_collect_poses.py to control the car for position data collection. The collected position data will be stored in `%userprofile%\Documents\AirSim`.
```
  python custom_collect_poses.py
```

#### Transfer Learning
The network weights will be saved after training. Running load_model.py will load and execute the saved weights.
```
  python load_model.py
```

#### Path Evaluation
To evaluate the model's path-tracking performance, robotics navigation metrics (SR, OSR, NE, SDTW, NDTW, CLS) are provided in the evaluation.py script. Executing the script will generate a comparison plot of the resulting and reference trajectories.

See the [documentation](https://github.com/BashMocha/Reinforcement-Learning-in-AirSim/blob/master/docs/Reinforcement_Learning_on_Autonomous_Vehicles.pdf) for a detailed explanation of metrics.
```
  python evaluation.py
```
![trajectory-comp](https://github.com/user-attachments/assets/7101f733-4821-47eb-b740-276b05968c61)


## DQN Algorithm
A simple DQN is built to train the agents for the given path. RGB and GPS sensors are used as inputs to the network, and an action is obtained as the output. This approach enables the agent to track the path and avoid obstacles effectively.

See the [documentation](https://github.com/BashMocha/Reinforcement-Learning-in-AirSim/blob/master/docs/Reinforcement_Learning_on_Autonomous_Vehicles.pdf) for a detailed explanation of the utilized network.

![scheme-updated(1)](https://github.com/user-attachments/assets/e8e0eee5-dfb0-4d72-87fd-4fd66ae67a49)
