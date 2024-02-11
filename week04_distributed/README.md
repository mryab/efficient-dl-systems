# Week 4: Intro to distributed ML

* Lecture: [pdf version](./slides.pdf), [odp version](./slides.odp)
* Seminar & homework: [here](./practice.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mryab/efficient-dl-systems/blob/main/week04_distributed/practice.ipynb)
* [Video](https://disk.yandex.ru/i/925ihyugBYxJDg) (both lecture and seminar)
 
__Note:__ while you *can* do the entire thing in Colab, we recommend using a node with at least 4 cpus to better "feel" the difference :)

Most modern PCs already have 4+ cores, but you can also find free cloud alternatives [here](https://www.dataschool.io/cloud-services-for-jupyter-notebook/).
The easiest version is to use Kaggle kernels.

__Note 2:__ The practice notebook was tested in Linux and MacOS. Running in Windows may cause problems due to inability to fork processes. When in doubt, try [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) or docker[(kitematic)](https://kitematic.com/). Run linux inside a VM will also do the trick.

__More stuff:__
* [Numba parallel](https://numba.pydata.org/numba-doc/dev/user/parallel.html) - a way to develop threaded parallel code in python without GIL
* [joblib](https://joblib.readthedocs.io/) - a library of multiprocessing primitives similar to mp.Pool, but with some extra conveniences
* BytePS paper - https://www.usenix.org/system/files/osdi20-jiang.pdf
* Alternative lecture: Parameter servers from CMU 10-605 - [here](https://www.youtube.com/watch?v=N241lmq5mqk)
* Alternative seminar: python multiprocessing - [playlist](https://www.youtube.com/watch?v=RR4SoktDQAw&list=PL5tcWHG-UPH3SX16DI6EP1FlEibgxkg_6)