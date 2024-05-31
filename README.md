# Deep learning for longevity prediction
This project was inspired by an off-hand comment by Balaji Srinivasan in his
[podcast with Lex
Fridman](https://www.youtube.com/watch?v=VeH7qKZr0WI&ab_channel=LexFridman).

$$ F(\text{picture of face}, \text{age}) = \text{life expectancy} $$

The methodology is to collect a large dataset of pictures of faces and their dates, along with the dates of birth and death of the people in the picture. Then train a neural network to predict age of death.

Hardware requirements: the models can be trained on CPU, but will be faster with a CUDA backend. MPS backend (Apple Silicon) is also available as of torch version 2.2.0.dev20231013. I highly recommend using this if you are training on a MacBook.

# Performance
At present, the best model trained with this repo has a <u>MAE of 6.3 years</u>. For comparison, the best age prediction models have an [MAE of 3.7 years](https://paperswithcode.com/sota/age-estimation-on-utkface). Just predicting that everyone will live for the mean life expectancy has an MAE of 15+ years, which is a good baseline to compare with. So it's not great, but seems like a convincing proof-of-concept.

Below you can see a video of how the model performs on the test set.

https://github.com/trialan/longevitynet/assets/16582240/d2f43a79-178a-4aa5-924e-750ca353a973
# Dataset
The current dataset, dataset_v4, is composed of 8297 training examples and 3500 test and validation examples. I cleaned it manually to remove any image with multiple people in it (with a bit of help from haar cascades and some NNs to count faces). One limitation of dataset_v4 is the gender and age imbalances: it skews older, and is composed of 92% men. I hope to address this in a future update. It is possible to downsaple dataset_v4 and get a balanced dataset, on which out-sample MAE is about 9 years.
