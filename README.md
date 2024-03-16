# SoundingBodies

## Supplemental Video

https://github.com/facebookresearch/SoundingBodies/assets/17986358/1213c073-d096-42d0-9a38-ac8ba68d01f2

## Data
The [Sounding Bodies dataset](https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/SoundingBodies/dataset/index.html) is hosted on AWS S3.
We recommend using the AWS command line interface (see [AWS CLI installation instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)).

To download the dataset run:
```
aws s3 cp --recursive --no-sign-request s3://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15/SoundingBodies/ SoundingBodies/
```
or use `sync` to avoid transferring existing files:
```
aws s3 sync --no-sign-request s3://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15/SoundingBodies/ SoundingBodies/
```
The dataset takes around 680GB of space. If necessary, in `configs/config_main.py` adjust `data_dir` and `mic_loc_file` to point to your download location. 

NOTE: The published datased does not include speech data from subject7 and has no data from subject8. With respect to data used in the paper, this brings the total capture time from 4.4 hours to 3.6 hours. Below we provide pretrained model and updated evaluation numbers for the published dataset.

## Code
Third-party dependencies: 
- tqdm
- numpy
- gitpython
- mmcv
- torch
- torchaudio


To train the network, run:
```
python train.py --config configs/config_main.py
```
To evaluate the performance of the model, in `configs/config_main.py` change `test_info_file` to desired test set: `./data_info/test/nonspeech_data.json` for non-speech data, and `./data_info/test/speech_data.json` for speech data, and run:
```
python evaluate.py --config configs/config_main.py --test_epoch best-accumulated_loss --out_name test 
```
To save the output `.wav` files add `--save` option, for example: 
```
python evaluate.py --config configs/config_main.py --test_epoch epoch-100 --out_name test --save
```

## Pretrained model
We provide the model trained on the published training set in `./checkpoint/neurips/pretrained/`. To evaluate the model, run: 
```
python evaluate.py --config configs/config_pretrained.py --test_epoch best-accumulated_loss --out_name neurips_evaluation 
```
The updated evaluation metrics are:

NON-SPEECH
```
SDR:                3.052
amplitude (x10^3):  0.832
phase:              0.314
```
SPEECH
```
SDR:                9.635
amplitude (x10^3):  0.701
phase:              0.464
```
NOTE: For the speech metrics reported in the paper, speech audio data was erroneously amplified by 10, resulting in the amplitude error being multiplied by 10, and the phase error being higher due to more silence/noise segments passing the energy threshold. 

## Citation

If you use this code or the dataset, please cite

```
@inproceedings{xu2023soundingbodies,
  title={Sounding Bodies: Modeling 3D Spatial Sound of Humans Using Body Pose and Audio},
  author={Xu, Xudong and Markovic, Dejan and Sandakly, Jacob and Keebler, Todd and Krenn, Steven and Richard, Alexander},
  booktitle={Conference on Neural Information Processing Systems},
  year={2023}
}
```

## License

The code and dataset are released under [CC-BY-NC 4.0 license](https://github.com/facebookresearch/SoundingBodies//blob/main/LICENSE).
