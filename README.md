# Final Report Code for Human Action Recognition

---

Mingcong Chen 27/3/2019

---

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

##  1. Environments

**Pytorch** : 1.0

**Python** : 3.6.5

**tensorboardX** ```pip install tensorboardX```

**tensorflow** : 1.7.0 ```pip install tensorflow==1.7.0```

##  2. Files Description

**joints** : folder of joints data files

**runs** : folder for saving tensorboard data

**lable.txt** : lable of actions

**network.py** : main function

## 3. How to use 

```bash
python3 network.py

(20, 10, 2)
15860
torch.Size([15860])
torch.Size([15860, 20, 3])
Choose network:
1.Linear Network
2.Full Connection Network
3.CNN
Enter index to choose network:
```

Then enter the index of network, it will start to train.
