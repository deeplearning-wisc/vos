# VOS

This is the source code for the illustration in Figure. 1.  the paper [***VOS: Learning What You Don’t Know by Virtual Outlier Synthesis***](https://openreview.net/forum?id=TW7d65uYu5M) 
by Xuefeng Du, Zhaoning Wang, Mu Cai, and Yixuan Li.


To train the classification model on 3-class Gaussian data using the standard cross entroy, run the following command:
```
python main.py --train_scheme ce --train_epoch 3000
```
At the save time, you will see the resulting image in folders  `./results`.


To see, run the following command:
```
python main.py --train_scheme vos --train_epoch 6000
```
At the save time, you will see the resulting image in folders  `./results`.
