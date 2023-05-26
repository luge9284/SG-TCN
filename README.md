# SG-TCN:Semantic Guidance for temporal action segmentation
This repo provides training & inference code for IJCNN 2022 paper: [SG-TCN:Semantic Guidance for temporal action segmentation](https://ieeexplore.ieee.org/abstract/document/9891932).
## Enviroment
Pytorch == 1.4.0, torchvision == 0.5.0, python == 3.6, CUDA=10.2
## Datasets
In our paper, we use 50Salads, GTEA and Breakfast datasets for training and evaluation following the baseline model [MS-TCN](https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html). You can download the data folder [here](https://mega.nz/file/O6wXlSTS#wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8), which contains the video features and the ground truth labels.(~30GB)
## Train the model
Run

    python main.py --action=train --dataset=50salads/gtea/breakfast --split=1/2/3/4/5
## Evaluate the model
Run

    python main.py --action=predict --dataset=50salads/gtea/breakfast --split=1/2/3/4/5
## Citation:
If you find our repo useful, please cite

    @inproceedings{zhang2022sg,

        title={SG-TCN: Semantic Guidance Temporal Convolutional Network for Action Segmentation},
  
        author={Zhang, Yunlu and Ren, Keyan and Zhang, Chun and Yan, Tong},
  
        booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
  
        pages={1--8},
  
        year={2022},
  
        organization={IEEE}
  
    }

If you have any questions with our code, please don’t hesitate to let us know .
