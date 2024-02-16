# ActIVO: An Active perception framework for skill transfer through Iterative Visual Observations 

The research project "CARPOS - Coachable Robot for Fruit Picking Operations" is implemented in the framework of H.F.R.I call "Basic Research Financing (Horizontal support of all Sciences)" under the National Recovery and Resilience Plan "Greece 2.0" funded by the European Union – NextGenerationEU(H.F.R.I. Project Number: 16523).

https://carpos.hmu.gr/ 

<p align="center">
  <img src="./doc/Greece_2.jpg" height="275" />
  <img src="./doc/elidek_logo.png" height="275" />
</p>

This package implements an active perception controller for minimizing the uncertainty during visual observations of an action performed by a human hand. 

A UR5e robot is considered with a realsense d415 camera attached to its end-effector.


<p align="center">
  <img src="./doc/example_ellipses_crop.png" height="275" />
  <img src="./doc/control_descr.png" height="275" />
</p>


## 1. Usage

For running the active perception controller:

    ```
    python3 ActiVO_main.py

    ```

For post-processing and skill learning, based on the recorded data:

```
python3 ActiVO_learning.py

```
