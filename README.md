pandora
==========

A (language-independent) Tagger-Lemmatizer for Latin & the Vernacular


### Install

For now, installation needs to be done by pulling the repository and installing the required libraries yourself.
Currently, Pandora relies to either Keras (+TensorFlow) or Pytorch as backends. In order to run Pandora with
the Pytorch backend, you should go to [pytorch.org](https://www.pytorch.org) and follow the installation
instructions.

#### Environment free

*Note* : if you have CUDA installed, you should do `pip install -r requirements-gpu.txt` instead

```bash
git clone https://github.com/hipster-philology/pandora.git
cd pandora
pip install -r requirements.txt
```

#### Virtualenv

**For CUDA-Ready machines owner**:

```bash
git clone https://github.com/hipster-philology/pandora.git
cd pandora
virtualenv env
source env/bin/activate
pip install -r requirements-gpu.txt
```

**For the others**:

```bash
git clone https://github.com/hipster-philology/pandora.git
cd pandora
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

### Scripts

*Note* : with Virtualenv install, do not forget to do `source env/bin/activate`.

#### main.py

`train.py` allows you to train your own models :

```bash
python train.py --help
python train.py config.txt --dev /path/to/dev/resources --train /path/to/train/resources --test /path/to/test/resources
python train.py config.txt --dev /path/to/dev/resources --train /path/to/train/resources --test /path/to/test/resources --nb_epochs 1
python train.py path/to/model/config.txt --load --dev /path/to/dev/resources --train /path/to/train/resources --test /path/to/test/resources
```

#### unseen.py

`tagger.py` allows you to annotate a string or folder

```bash
python tagger.py --help
python tagger.py path/to/model/dir --string --input "Cur in theatrum, Cato severe, venisti?"
python tagger.py path/to/model/dir --input /path/to/dir/to/annotate/ --output /path/to/output/dir/
python tagger.py path/to/model/dir --tokenized_input --input /path/to/dir/to/annotate/ --output /path/to/output/dir/
```
