
This week's menu:
* Slides: [pdf version](https://disk.yandex.ru/i/sn3j7dVvehkMdQ), [odp version](https://disk.yandex.ru/i/Lrgenf4Q_qOveQ)
* Lecture: Basic parallelism & parameter servers (in Russian) - [here](https://disk.yandex.ru/i/iKjF7KyUvMsztQ)
* Seminar: multiprocessing basics (in Russian) - [here](https://disk.yandex.ru/i/0Tth1vu8w47xdg)
* Alternative lecture: Parameter servers from CMU 10-605 - [here](https://www.youtube.com/watch?v=N241lmq5mqk)
* Alternative seminar: python multiprocessing - [playlist](https://www.youtube.com/watch?v=RR4SoktDQAw&list=PL5tcWHG-UPH3SX16DI6EP1FlEibgxkg_6)


Practice: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mryab/efficient-dl-systems/blob/2022/week02_distributed/practice.ipynb)
 
__Note:__ while you *can* do the entire thing in colab, we recommend using a node with at least 4 cpus to better "feel" the difference :)

Most modern PCs already have 4+ cores, but you can also find free cloud alternatives [here](https://www.dataschool.io/cloud-services-for-jupyter-notebook/).
The easiest version is to use kaggle kernels.

__Note 2:__ The practice notebook was tested in Linux and MacOS. Running in Windows may cause problems due to inability to fork processes. When in doubt, try docker[(kitematic)](https://kitematic.com/) or run a linux VM. [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) might also do the trick.

__More stuff:__
* [numba parallel](https://numba.pydata.org/numba-doc/dev/user/parallel.html) - a way to develop threaded parallel code in python without GIL
* [joblib](https://joblib.readthedocs.io/) - a library of multiprocessing primitives similar to mp.Pool, but with some extra conveniences
* BytePS paper - https://www.usenix.org/system/files/osdi20-jiang.pdf
