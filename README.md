# simpleBilliards



This script depends only on ```numpy``` and ```matplotlib```, which can be installed using pip with:

```
python -m pip install numpy matplotlib
```

With these two requirements installed, the script can be executed with ```python billiards.py```. No argument parser was implemented and options are set with parameters in the main function:

```
    DT = 5e-2 # Integration time step
    BOUNDS = 60 # Graph bounds (AU)
    ITVL_PYPLOT = 10 # Interval of each step in the animation (miliseconds)
    NUM_PARTICLES = 3 # number of particles
```


## Methods

It uses the leapfrog method for integrating position and velocity.
