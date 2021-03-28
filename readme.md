# Installation Guide

1. Make sure `python 3.7` installed. Run `sudo apt install python3.7` (ubuntu os)
2. Clone this repo `git clone https://github.com/DoraMatching/DoraMatching-API-ML.git`
3. `cd DoraMatching-API-ML`
4. Install virtualenv: `pip3 install --user virtualenv`
5. Set env _python37_: `virtualenv --python=/usr/bin/python3.7 python37`
6. Active _python37_: `source python37/bin/activate`
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

## For Raspbian OS

```shell
sudo apt-get install libatlas-base-dev
```

## For Vagrant box

Recommend: cpus>=2, memory>=2GB

```ruby
# Vagranfile
# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
    config.vm.box = "tranphuquy19/doramatching-api-ml"
    config.vm.box_version = "1.0.0"
    config.vm.network "private_network", ip: "172.16.10.107"
    config.vm.hostname = "ml.dora"
  
    config.vm.provider "virtualbox" do |vb|
        vb.name = "ml.dora"
        vb.cpus = 2
        vb.memory = "2048"
    end
end
```
Init VM: `vagrant up` then `vagrant ssh`

Start API Server:

```shell
cd cd DoraMatching-API-ML/
source python38/bin/activate # active env
python3 app.py # start server
```
