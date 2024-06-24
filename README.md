<p align="center">
  <img src="./docs/img/logo.png" alt="drawing" width="70%"/>
</p>

Cognixlib is a python package and library designed to help with the offline or online analysis of EEG and Eye-Gaze signals. It emerged as a deliverable of the [CogniX](http://www.cognix.gr) project. It provides a scripting API that helps in the manipulation, filtering or general transformation of various kind of biophysical signals, currently specialized for EEG and Eye-Gaze. It also provides a node-based API created using the scripting API and [cognixcore](https://github.com/CogniX-Up/cognixcore). The original commit history can be found in a [fork](https://github.com/HeftyCoder/Ryven/tree/cognix). After initial development it was moved ton this standalone repository. Much of the library is still being tested and improved on, so you can expect breaking changes.

### Installation

PyPi installation is in the works. Python Version should at least be 3.11.

Before installing from source, make sure to also install [cognixcore](https://github.com/CogniX-Up/cognixcore)

Install from sources - preferably after creating and activating a [python virtual environment](https://docs.python.org/3/library/venv.html): 
```
git clone https://github.com/CogniX-Up/cognixlib.git
cd cognixcore
pip install .
```

### Usage

You can utilize this package as a scripting API or as a nodes-based API. Since it is quite experimental, breaking changes might be introduced. For a guide on how to get started and what you can actually do, visit the [docs](https://cognix-up.github.io/cognixlib/)

### Features

The main features include

- A variety of utilities, usually in the form of classes, that make it easier to manipulate time signals.
- Segment a signal based on time-markers or windows with overlaps.
- Train, test and deploy machine learning algorithms through an easy-to-setup API.
- Utilize all the above in a scripting manner or through a nodes library designed to work with [cognixcore](https://github.com/CogniX-Up/cognixcore) and [cognix-editor](https://github.com/CogniX-Up/cognix-editor)

### Licensing

cognixlib is licensed under the [GPL License](https://github.com/CogniX-Up/cognixlib/blob/main/LICENSE).