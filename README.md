# Longevity prediction
This project was inspired by an off-hand comment by Balaji Srinivasan in his
[podcast with Lex
Fridman](https://www.youtube.com/watch?v=VeH7qKZr0WI&ab_channel=LexFridman).

$$ F(\text{picture of face}, \text{age}) = \text{life expectancy} $$

The methodology is to collect a large dataset of pictures of faces and their dates, along with the
dates of birth and death of the people in the picture. Then train a neural network to predict age of death.

# Performance
A ResNet-50 gets to sub-1 year accuracy for life expectancy on the test set.

| Model | Best test loss | Estimated precision (years) | Git hash |
|---------|---------|---------|---------|
| ResNet-50 (last block un-frozen)   | 0.007870   | 0.08   | 5e0fb47a6c00118495dca9ba6   |
# Known limitations
1. The dataset is heavily skewed towards older people, so I'm not sure how well it
performs on pictures of younger people.

2. When generating the dataset I simplified myself the work so all dates are just
years (I didn't bother with the month or day of the year, so at best this can
only ever be accurate to the year, this is easy to change however).

3. Some pictures in the dataset (I estimate <2%) have more than one person in
   them, which is "corrupt" data.

4. Some of the pictures are in black and white, others are colored. I don't
   think this is good because black and white pictures are older / might leak
some sort of information.

5. I've done very little optimization: there is a lot of room for performance
   improvements: before doing this we need to adress point (2). I also think
that before doing more performance boosts it makes sense to put this behind a
frontend to share with the world and see if there's any demand for this.

*Other possible problems*: I haven't thought hard about data leakage, so maybe
something is off here.

# High level technical overview
1. [Wikidata's API](query.wikidata.org) to generate the dataset (plus a bit of
   scraping): dataset_v2 has ~5000 examples. I also have dataset_v3 with about
14k examples, but haven't used it yet.
2. I used pre-trained models as the initialisation of my neural nets,
ensembling different pre-trained models improves performance.
3. The target variable is a min-max scaled delta-life expectancy (subtracted
   the mean life expectancy, this is a big improvement on just predicting
min-max scaled life expectancy).
4. The loss function is mean squared error
5. I use the function in ```interpretability.py``` to interpret the MSELoss as
an accuracy in years. This seems broadly legit, but is an approximation (don't
use test data to do the scaling-unscaling, also MSELoss obviously quite
sensitive to outliers).

The models are small enough that you can train on CPU, but I recommend running
on a GPU (I did my training on a Quadro M4000, takes about 10 minutes for 15
epochs which is more than enough).

*Some details on dataset cleaning*: a lot of the results returned by the wikidata
API (see exact query in ```dataset_generation/wikidata.py```) had pictures that
were stamps of the person, not actual pictures. The way this came up is that
when running the dataset generation / scraping code, the year of death and the
year of the picture would be the same. It seems important to remove these data
before using a new dataset, but I'm not sure what the performance hit would be
from them. A ton of the wikidata results had pictures without a figure caption
(so there is no picture date to scrape), so my dataset collection script is
pretty inefficient because I didn't use SPARQL to filter these. A last thing I
did is look through the data for rows with crazy life expectancies and spotted
a couple bad apples this way.


