<center align="center">
<h1 align="center"><font size="+4">Arabic Dialect Classification</font></h1>
</center>

<h1 color="green"><b>Abstract</b></h1>
<p>Auto-Complete is a feature that provides relevant suggestions based on input by the user. It works best in domains with a limited number of possible words.
</p>

<h1 color="green"><b>Tried Methods</b></h1>
<ol>
<li>TF-IDF + Linear SVM</li>
<li>Keras Embedding Layer + LSTM</li>
<li>Arabert Transformer Embedding + LSTM</li>
<li>Fine-tuning Arabert Transformer</li>
</ol>

<h1 color="green"><b>Training Dataset</b></h1>
<p>1.5 billion words Arabic Corpus dataset was used for this specific task. In this <a href="https://arxiv.org/ftp/arxiv/papers/1611/1611.04033.pdf">paper</a>, the researcher has chosen ten sources to be used in the corpus. Several news websites were tested before selecting the source that will be used. The fame of the website, and the news source, or the number of readers were not the criterion for selection. There were other criteria and technical reasons for selecting the news resources used in building the corpus.

Out of all these ten sources, only two were chosen to be used due to limited resources. Almasralyoum which was used to train all our models and youm7 which was only used in fine tuning the transformer along with Almasralyoum.
</p>

<h1 color="green"><b>Application</b></h1>
<p>The SVM model and Fine-tuned Transformer were the models used to build an API using a new easy web framework which is FastAPI. In order to try out the application, please follow the following instructions:</p>

1. Install all Python libraries that the notebooks depend on:

```python
pip install -r requirements.txt
```

2. Download files needed from these links: <a href="https://drive.google.com/file/d/1dvtXwdMghOQNC0lP_qPT4OXZz_XN3ebC/view">TF-IDF</a>  <a href="https://drive.google.com/file/d/1qoiulklaR5co2z3YiG4ZuibhtRIQ5UvZ/view">SVM-Model</a>  <a href="https://drive.google.com/file/d/1yOaqkUGAamXc15xy3oW16_aQijEBRZ5_/view">Finetuned-Transformer-Weights</a>

3. Clone the Arabert repo:
```python
git clone https://github.com/aub-mind/arabert.git
```

4. Run the server:

```python
python App_FastAPI.py -t [Tfidf Path] -ml [SVM Model Path] -dl [Transformer Weights Path]
```

5. Navigate to your local host `http://localhost:8000/docs`

6. Assign the text that needs to be classified


<h1 color="green"><b>Demo</b></h1>
<img src="images/1.PNG" alt="Simply Easy Learning" >
<img src="images/2.PNG" alt="Simply Easy Learning" >
<img src="images/3.PNG" alt="Simply Easy Learning" >
<img src="images/4.PNG" alt="Simply Easy Learning" >
