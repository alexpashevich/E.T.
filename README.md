# Episodic Transformers (E.T.)

[<b>Episodic Transformer for Vision-and-Language Navigation</b>](https://arxiv.org/abs/2105.06453)  
[Alexander Pashevich](https://thoth.inrialpes.fr/people/apashevi/), [Cordelia Schmid](https://www.di.ens.fr/willow/people_webpages/cordelia/), [Chen Sun](https://chensun.me/)

**Episodic Transformer (E.T.)** is a novel attention-based architecture for vision-and-language navigation. E.T. is based on a multimodal transformer that encodes language inputs and the full episode history of visual observations and actions.
This code reproduces the results obtained with E.T. on [ALFRED benchmark](https://arxiv.org/abs/1912.01734). To learn more about the benchmark and the original code, please refer to [ALFRED repository](https://github.com/askforalfred/alfred).

![](files/overview.png)

## Quickstart

Clone repo:
```bash
$ git clone https://github.com/alexpashevich/E.T..git ET
$ export ET_ROOT=$(pwd)/ET
$ export ET_LOGS=$ET_ROOT/logs
$ export ET_DATA=$ET_ROOT/data
$ export PYTHONPATH=$PYTHONPATH:$ET_ROOT
```

Install requirements:
```bash
$ virtualenv -p $(which python3.7) et_env
$ source et_env/bin/activate

$ cd $ET_ROOT
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Downloading data and checkpoints

Download [ALFRED dataset](https://github.com/askforalfred/alfred):
```bash
$ cd $ET_DATA
$ sh download_data.sh json_feat
```

Copy pretrained checkpoints:
```bash
$ wget http://pascal.inrialpes.fr/data2/apashevi/et_checkpoints.zip
$ unzip et_checkpoints.zip
$ mv pretrained $ET_LOGS/
```

Render PNG images and create an LMDB dataset with natural language annotations:
```bash
$ python -m alfred.gen.render_trajs
$ python -m alfred.data.create_lmdb with args.visual_checkpoint=$ET_LOGS/pretrained/fasterrcnn_model.pth args.data_output=lmdb_human args.vocab_path=$ET_ROOT/files/human.vocab
```
Note #1: For rendering, you may need to configure `args.x_display` to correspond to an X server number running on your machine.  
Note #2: We do not use JPG images from the `full` dataset as they would differ from the images rendered during evaluation due to the JPG compression.  

## Pretrained models evaluation

Evaluate an E.T. agent trained on human data only:
```bash
$ python -m alfred.eval.eval_agent with eval.exp=pretrained eval.checkpoint=et_human_pretrained.pth eval.object_predictor=$ET_LOGS/pretrained/maskrcnn_model.pth exp.num_workers=5 eval.eval_range=None exp.data.valid=lmdb_human
```
Note: make sure that your LMDB database is called exactly `lmdb_human` as the word embedding won't be loaded otherwise.

Evaluate an E.T. agent trained on human and synthetic data:
```bash
$ python -m alfred.eval.eval_agent with eval.exp=pretrained eval.checkpoint=et_human_synth_pretrained.pth eval.object_predictor=$ET_LOGS/pretrained/maskrcnn_model.pth exp.num_workers=5 eval.eval_range=None exp.data.valid=lmdb_human
```
Note: For evaluation, you may need to configure `eval.x_display` to correspond to an X server number running on your machine.

## E.T. with human data only

Train an E.T. agent:
```bash
$ python -m alfred.model.train with exp.model=transformer exp.name=et_s1 exp.data.train=lmdb_human train.seed=1
```

Evaluate the trained E.T. agent:
```bash
$ python -m alfred.eval.eval_agent with eval.exp=et_s1 eval.object_predictor=$ET_LOGS/pretrained/maskrcnn_model.pth exp.num_workers=5
```
Note: you may need to train up to 5 agents using different random seeds to reproduce the results of the paper.

## E.T. with language pretraining

Language encoder pretraining with the translation objective:
```bash
$ python -m alfred.model.train with exp.model=speaker exp.name=translator exp.data.train=lmdb_human
```

Train an E.T. agent with the language pretraining:
```bash
$ python -m alfred.model.train with exp.model=transformer exp.name=et_synth_s1 exp.data.train=lmdb_human train.seed=1 exp.pretrained_path=translator
```

Evaluate the trained E.T. agent:
```bash
$ python -m alfred.eval.eval_agent with eval.exp=et_synth_s1 eval.object_predictor=$ET_LOGS/pretrained/maskrcnn_model.pth exp.num_workers=5
```
Note: you may need to train up to 5 agents using different random seeds to reproduce the results of the paper.

## E.T. with joint training

You can also generate more synthetic trajectories using [generate\_trajs.py](alfred/gen/generate_trajs.py), create an LMDB and jointly train a model on it.
Please refer to the [original ALFRED code](https://github.com/askforalfred/alfred/tree/master/gen) to know more the data generation. The steps to reproduce the results are the following:
1. Generate 45K trajectories with `alfred.gen.generate_trajs`.
2. Create a synthetic LMDB dataset called `lmdb_synth_45K` using `args.visual_checkpoint=$ET_LOGS/pretrained/fasterrcnn_model.pth` and `args.vocab_path=$ET_ROOT/files/synth.vocab`.
3. Train an E.T. agent using `exp.data.train=lmdb_human,lmdb_synth_45K`.

## Citation

If you find this repository useful, please cite our work:
```
@misc{pashevich2021episodic,
  title ={{Episodic Transformer for Vision-and-Language Navigation}},
  author={Alexander Pashevich and Cordelia Schmid and Chen Sun},
  year={2021},
  eprint={2105.06453},
  archivePrefix={arXiv},
}
```
