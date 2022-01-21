# Implementation of "Modal Uncertainty Estimation for Medical Image Diagnosis".

This is a pytorch reimplementation of the MUE model proposed in [this paper](https://link.springer.com/chapter/10.1007/978-3-030-87735-4_1).

To train the model, have a look at the functions in `/pipeline/train_lib.py` and configuration examples provided in `/pipeline/configs`.
It should straightforward to create a `train_main.py` file (preferably in `/pipeline`) using the functions therein and your own associated .gin configuration files.
Then run the following command from the project root directory
```
python -m PATH_TO_TRAIN_MAIN
```
to train the model.


For evaluation, refer to the functions defined in `/pipeline/eval_lib.py`, and create your `eval_main.py` file similarly.  
Example for loading the evaluation results and compute the _Generalized Energy Distance_ metric can be found in `plot_LIDC.ipynb`.


If you find this research helpful, consider citing our paper:
```
@incollection{qiu2021modal,
  title={Modal Uncertainty Estimation for Medical Imaging Based Diagnosis},
  author={Qiu, Di and Lui, Lok Ming},
  booktitle={Uncertainty for Safe Utilization of Machine Learning in Medical Imaging, and Perinatal Imaging, Placental and Preterm Image Analysis},
  pages={3--13},
  year={2021},
  publisher={Springer}
}
```
 
