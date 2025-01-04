import airsim
from time import sleep

def main():
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(False)
    
    try:
        count = 0
        while True:
            client.startRecording()
            car_position = client.simGetVehiclePose().position
            print("{0}th Car Position: {1}, {2}, {3}".format(count, car_position.x_val, car_position.y_val, car_position.z_val))
            sleep(3)
            count += 1
    except KeyboardInterrupt:
        print("Driving interrupted.")
        car_controls = airsim.CarControls()
        car_controls.brake = 1.0
        client.setCarControls(car_controls)
        
        client.stopRecording()
        client.enableApiControl(True)

if __name__ == '__main__':
    """16 19 37 41"""
    main()