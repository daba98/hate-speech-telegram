# Hate Speech Detection in Telegram - Python
There is more and more german hate speech in Telegram, that should be automatically detected. This automatic detection can be done by machine learning, however, one needs a lot of annotated data in order to train those machine learning models. For german hate speech there are only very few datasets for this and most of them don't contain Telegram messages, but Twitter posts.
Therefore, we tried to train eleven classifiers on different Twitter datasets and tested whether they can be also used to detect hate speech in Telegram messages. In particular, we used the following datasets:
- Germeval 2018
- Germeval 2019
- Hasoc 2019
- Hasoc 2020
- a special dataset containing COVID-19 related hate speech

One problem is that Twitter post have a character limit, while Telegram messages don't. To tackle this issue, we split long Telegram messages into small chunks and classify each chunk individually. The label of the whole message is simply the most hateful label of its chunks.
![classification_pipeline](https://user-images.githubusercontent.com/109282684/181725728-69665491-4f4b-4c18-aaae-f4731d2be3ed.png)

In order to evaluate the classifiers, we created a test dataset containing Telegram messages from different hate speech topics. For reasons of comparison, we also classified those messages with the [Perspective API](https://perspectiveapi.com/) from Google. For a more detailed description of the work, see the [work summary paper](paper.pdf), or the paper [Introducing an Abusive Language Classification Framework for Telegram to Investigate the German Hater Community](https://ojs.aaai.org/index.php/ICWSM/article/view/19364), that also contains part of this work.

Note that this repo does not contain the binary files of the models, because of their huge size.
