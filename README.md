<a name="readme-top"></a>

<br />
<div align="center">
  <h3 align="center">Dilbert & Boss - Dialogue Generator Trasnformer</h3>

  <p align="center">
The main purpose of this repository  is the generation of dialogues, the training and practical implementation of the Transformer architecture and the prediction script. Through the use of the Python language and the Tensor Flow and Karas libraries.
This is the fourth phase of the project following the construction of the dataset. the previous three in order are
</p>
<ol>
<li>[dilbert-cascade-classifier by GuiZ88](project https://github.com/GuiZ88/dilbert-cascade-classifier)</li>
<li>[dilbert-boss-cascade-classifier by GuiZ88](project https://github.com/GuiZ88/dilbert-text-extractor)</li>
<li>[dilbert-text-extractor by GuiZ88](project https://github.com/GuiZ88/dilbert-text-extractor)</li>
</ol>

</div>

<!-- ABOUT THE PROJECT -->
## About The Project

After building the 'dataset.txt' we proceed to train the models for both Dilbert and the Boss.

```sh
python3 training_dilber.py
```

```sh
python3 training_boss.py
```

In each file it is possible to parameterise the number of epochs and further parameters of the Transformer on the basis of the dataset obtained.
The 'dataset.txt' file is a portion of it. After the training the two h5 files are produced: 'model_boss.h5' and 'model_dilbert.h5'

The prediction generates a dialogue of three keystroke exchanges like a classic comic strip.

```sh
python3 predict.py
```

```txt
Boss:  are you trying to kill us? 
Dilbert: thats a real thing .

Boss: it might be all of those .
Dilbert: but seriously , we can finish the project ?

Boss: well , the bad data to support that claim .
Dilbert: well , we have been getting some inconsistent results when users try to log in . sometimes it works , for a little bit .
```
<u><b>The result of the prediction is strongly influenced by the dataset and on the basis of the latter by the epochs.</b></u>

<p align="right">(<a href="#readme-top">back to top</a>)</p>