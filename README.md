# TB-MIA
Unofficial pytorch implementation of paper: Neural Network Inversion in Adversarial Setting via Background Knowledge Alignment

## Description
This this an <b>unofficial</b> pytorch implementation of paper:	Z. Yang, J. Zhang, E.-C. Chang, and Z. Liang, "Neural Network Inversion in Adversarial Setting via Background Knowledge Alignment."


The official code from the author can be found at:https://github.com/yziqi/adversarial-model-inversion

## Usage
The main branch contains the code for attacking the MNIST dataset. 

* main.py -- train the target network
* test.py -- test the target network
* attack.py -- perform model inversion attack
* turn_to_list.py -- code to make the dataset index
* the four txts -- dataset index files

<b>IMPORTANT: The code runs successfully on my machine. However, I might forgot to describe some details when writting this readme. So feel free to contact me via [email](mailto:zhangzp9970@outlook.com) or GitHub issues :)</b>

## Third-party libraries

* pytorch 1.8.1
* torchvision
* [easydl](https://github.com/thuml/easydl)
* tqdm
* numpy
* some standard python libs
* [mnist_png](https://github.com/myleott/mnist_png)

## Differents
I only test my code on the 'Generic' settings. I use 0,1,2,3,4 to train the classifier and 5,6,7,8,9 to train the inversion network.

Instead of the official MNIST dataset, I use [this project](https://github.com/myleott/mnist_png) to extract the image datas from MNIST databyte file into png images.


## License

Copyright © 2021 Zeping Zhang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.