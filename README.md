# Visible-Infrared-Person-Re-Identification-Baseline
 a baseline(ResNet50) code for visible-infrared person Reid.
**update:** add vit.
 
 continuous update...
 
 ## Requierment
 **torch, PIL, numpy**
 ```
 pip install torch
 pip install pillow
 pip install numpy
 ```
 ## Datasets
 **RegDB:**
 
 *Nguyen D T, Hong H G, Kim K W, et al. Person recognition system based on a combination of body images from visible light and thermal cameras[J]. Sensors, 2017, 17(3): 605.*
 
 **SYSU-MM01:**
 
 *Wu A, Zheng W S, Yu H X, et al. RGB-infrared cross-modality person re-identification[C]//Proceedings of the IEEE international conference on computer vision. 2017: 5380-5389.*
 
 If you want to train the SYSU-MM01, run the ``pre_process_sysu.py`` firstly and then run the ``train.py``.
 
 ## Train
 Set line 27 of ``train.py`` to your dataset path and run it.
 
 ## Footer
 This project is being continuously updated.
 
 Leave a star in GitHub if you found this helpful.
