pandora
==========

A (language-independent) Tagger-Lemmatizer for Latin & the Vernacular


The tagging technology behind Pandora is described in the following papers:
- Kestemont, M., De Pauw, G., Van Nie, R. & Daelemans, W., ‘Lemmatisation for Variation-Rich Languages Using Deep Learning’. Forthcoming in: *DSH – Digital Scholarship in the Humanities*. [https://academic.oup.com/dsh/article/doi/10.1093/llc/fqw034/2669790/Lemmatization-for-variation-rich-languages-using](paper)
- Kestemont, M. & J. de Gussem, ‘Integrated Sequence Tagging for Medieval Latin Using Deep Representation Learning’, Journal of Data Mining & Digital Humanities (2017), pp. 17. Special Issue on Computer-Aided Processing of Intertextuality in Ancient Languages, ed. M. Buechler and L. Mellerin [https://jdmdh.episciences.org/3835/pdf](pdf).

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

Note that we do not officially support the Theano backend for keras (anymore), because the Theano development will halt after the 1.0 release ([https://groups.google.com/forum/#!topic/theano-users/7Poq8BZutbY](announcement)).

### Examples

The repository includes sample configurations (see `config_example` folder), and
 is shipped with a small test data-set of Old French epic texts from the Geste 
 corpus (https://github.com/Jean-Baptiste-Camps/Geste).
 
To launch training on this corpus, do
```bash
python3 train.py config_geste.txt --train data/geste/train --dev data/geste/dev --test data/geste/test
```
  

