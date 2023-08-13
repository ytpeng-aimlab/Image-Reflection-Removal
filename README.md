# Image-Reflection-Removal-based-on-Knowledge-distilling

## Prepare Datasets
#### Training datasets
* Synthetic: 13697 image pairs from [Zhang](https://drive.google.com/drive/folders/1NYGL3wQ2pRkwfLMcV2zxXDV8JRSoVxwA)
* Real: 200 image pairs from [IBCLN](https://drive.google.com/file/d/1YWkm80jWsjX6XwLTHOsa8zK3pSRalyCg/view) and 398 image pairs from [ERRNet](https://github.com/Vandermode/ERRNet)

Create and put the training images into the followimg folder

```
./dataset/Reflection/train
+--- input
+--- gt
```

#### Testing datasets
* SIR dataset: three sub-datasets (Solid, Postcard, Wild) from [SIR dataset](https://sir2data.github.io/).  
* Real dataset: 20 real testing images from [Berkeley real dataset](https://github.com/ceciliavision/perceptual-reflection-removal).
* Nature dataset: 20 real testing images from [IBCLN](https://drive.google.com/file/d/1YWkm80jWsjX6XwLTHOsa8zK3pSRalyCg/view).


#### Start training
```shell
$ CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 28500 main.py
```

#### Start testing
```shell
$ python -m torch.distributed.launch --master_port 28500 main.py --mode "test"
```
