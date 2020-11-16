# Installation Guide

1. Make sure `python 3.7` installed. Run `sudo apt install python3.6` (ubuntu os)
2. Clone this repo `git clone https://github.com/DoraMatching/DoraMatching-API-ML.git`
3. `cd DoraMatching-API-ML`
4. Install virtualenv: `pip3 install --user virtualenv`
5. Set env _python36_: `virtualenv --python=/usr/bin/python3.7 python37`
6. Active _python36_: `source python37/bin/activate`
7. Install dependencies:
```shell
pip3 install flask flask_cors requests nltk bs4 pandas matplotlib regex scipy sklearn underthesea
```
8. Copy Vietnamese stopwords to `corpora stopwords` folder
```shell
cp ./vietnamese_dash $HOME/nltk_data/corpora/stopwords/vietnamese_dash
```
9. Run the ML API `python3 app.py`
10. Test 