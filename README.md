# IO-aware-Factorization-Machines

This is our implementation for the paper:

IO-aware Factorization Machines for User Response Prediction  
Zhenhao Hu, Chao Peng, Cheng He

This is our implementation is based on [Attentional Factorization Machines](https://github.com/hexiangnan/attentional_factorization_machine).

## Environments
+ Tensorflow (version: 1.10.0)
+ numpy
+ sklearn

## Dataset
We use the same input format as the [LibFM toolkit](http://www.libfm.org/). 
In this instruction, we use MovieLens. 
The MovieLens data has been used for personalized tag recommendation, which contains 668,953 tag applications of users on movies. 
We convert each tag application (user ID, movie ID and tag) to a feature vector using one-hot encoding and obtain 90,445 binary features. 
The following examples are based on this dataset and it will be referred as ml-tag wherever in the files' name or inside the code. 
We provided the results of our pre-training(We skipped 'bigml-tag_256.data-00000-of-00001' because it is too big).
When the dataset is ready, the current directory should be like this:

- code   
   + AFM.py
   + FM.py
   + LoadData.py
- data
   + ml-tag
      + ml-tag.train.libfm
      + ml-tag.validation.libfm
      + ml-tag.test.libfm
- pretrain
   + fm_ml-tag_16
      + checkpoint
      + ml-tag_16.data-00000-of-00001
      + ml-tag_16.index
      + ml-tag_16.meta
## Quick Example with Optimal parameters
Use the following command to train the model with the optimal parameters:


```
# step into the code folder  
cd code

# train FM model and save as pretrain file  
python FM.py --dataset ml-tag --epoch 100 --pretrain -1 --batch_size 4096 --hidden_factor 256 --lr 0.01 --keep 0.8

# train IOFM model using the pretrained weights from FM  
python IOFM.py --dataset ml-tag --epoch 100 --pretrain 1 --batch_size 4096 --hidden_factor [8,256] --keep [1.0,0.8] --temp 10 --lr 0.05 --lamda_attention 1.0 --lamda_attention1 0.001
```

The instruction of commands has been clearly stated in the codes (see the parse_args function).

The current implementation supports regression classification, which optimizes RMSE.