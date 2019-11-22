The file "cooling_data.txt" contains data obtained from the default Simulink
simulation "Electric Vehicle Configured for HIL".

https://www.mathworks.com/help/physmod/sps/examples/electric-vehicle-configured-for-hil.html

The file contains temperatures from the inlet and outlet of the cooling system
as well as the temperature of the electric motor. The data columns are:

Time (s)     Inlet (C)     Outlet (C)     Motor (C)

The identification goal is to generate a predictor for the motor temperature
using the inlet and outlet temperatures. It should be noted that the vehicle
speed effects the airflow to the radiator, effecting the cooling dynamics.

