# Autonomous Driving in AirSim by Reinforcement Learning

> Autonomous driving for path tracking in AirSim by applying the Deep-Q Network algorithm. See the [documentation](https://github.com/BashMocha/Reinforcement-Learning-in-AirSim/blob/master/docs/Reinforcement_Learning_on_Autonomous_Vehicles.pdf) for a detailed explanation.
<br>

https://github.com/user-attachments/assets/263b3539-5bf9-4b35-b96e-7c6c5ffb3027

---

## Simulation Environment
AirSim's neighborhood environment is utilized due to its simple structure, making it well-suited for path-tracking algorithms. The project has been tested on Windows 11, but not on Linux; therefore, running the scripts on a Windows operating system is recommended.

The simulation environment can be downloaded from our release page. To configure the environment, move the `settings.json` file to the `%userprofile%\Documents\Airsim` and run the `AirsimNH.exe` from the environment folder. Once the simulation is running, a Python script containing the following lines can connect to the environment for API calls:

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

## DQN Algorithm
![scheme-updated(1)](https://github.com/user-attachments/assets/e8e0eee5-dfb0-4d72-87fd-4fd66ae67a49)



## TODO
- Update scripts
- Update README file
