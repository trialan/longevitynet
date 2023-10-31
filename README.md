# Deep Learning for longevity prediction
This project was inspired by an off-hand comment by Balaji Srinivasan in his
[podcast with Lex
Fridman](https://www.youtube.com/watch?v=VeH7qKZr0WI&ab_channel=LexFridman).

$$ F(\text{picture of face}, \text{age}) = \text{life expectancy} $$

The methodology is to collect a large dataset of pictures of faces and their dates, along with the dates of birth and death of the people in the picture. Then train a neural network to predict age of death.

Hardware requirements: the models can be trained on CPU, but will be faster with a CUDA backend. MPS backend (Apple Silicon) is also available as of torch version 2.2.0.dev20231013. I highly recommend using this if you are training on a MacBook.

# Performance
At present, the best model trained with this repo has a <u>MAE of 6.3 years</u>. For comparison, the best age prediction models have an [MAE of 3.7 years](https://paperswithcode.com/sota/age-estimation-on-utkface).
