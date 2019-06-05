# footprint_pred_height_and_weight





#  Introduce:

###  	the dataset is consist of  about two thousand of human's pressure footprint, devide by ID number. The weight and the height are provided in ZY122.xlsx. We could use footprint to pred human's height and weight.

###  Two kinds of pred models are used.  One is vgg16, another is resnet 34. Details abouts the models are below.

| model_nam\ACC | height  | weight  |
| ------------- | ------- | ------- |
| vgg16         | 96.927% | 93.268% |
| resnet34      | 97.690% | 93.321% |



# 	First step:  prepare the data

## - extract data from the 压力图ZY2未删减.rar,

`` python processing.py``

#  Second step: train

```    parser.add_argument('--n_epoch', nargs='?', type=int, default=50, 
    parser.add_argument('--n_epoch', nargs='?', type=int, default=50,  
  				    help='# of the 	epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16, 
                    help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-4, 
                    help='Learning Rate')
    parser.add_argument('--gpu',nargs='*', type=int, default=1)
    parser.add_argument('--model',nargs='?',type=str,default='vgg16')
    parser.add_argument('--use_pred',nargs='?',type=str,default=None)    ``
```

` python work.py --model resnet34    `





