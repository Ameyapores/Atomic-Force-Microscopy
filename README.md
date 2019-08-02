# Atomic-Force-Microscopy
Automation in predicting the Youngs Modulus
### Pre-requisite
- Python3.5+
- PyTorch 0.4.0
- Scipy 0.19.0
- pandas 0.24.0

### Pre-processing
Following pre-proceesing is undertaken to eliminate data from the text file that is not useful.
1) Remove unnecesary lines
2) Remove unnecesary columns
3) Take only first 250 data points
4) Normalize both the x and y values

### Directly predicting the Youngs modulus from (x,y) raw data
Neural network architecture: 
- Input layer:500 neurons
- hidden layer1:1000 neurons
- hidden layer2: 256 neurons
- Output layer: 1 neuron

### Finding the x
 <img src="images/Figure_1.png">

- The value of x is -2.703609e-07
- Youngs modulus from the fit: 690.5668434384045 
- Original youngs modulus: 682.199
