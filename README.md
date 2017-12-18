The final project for the course AST 381: Planetary Astrophysics (Fall 2017) at UT Austin.

`abstract/` and `proposal/` are folders used to track the history of this project. All of the code for this project is in `src/`. To run the code, add `src/` to the main MATLAB directory, then run `main.m`. Specifically:
```
cd <path>/src/
matlab <--no-splash> <--no-desktop> -r 'main';
```
This requires VLFeat 0.9.20 to be installed and built, and for the resulting folder to be renamed as 'vlfeat' and inside the source folder. VLFeat is available for free to download from their website [here](http://www.vlfeat.org/).

`report/` contains the main report for this project. It uses AASTeX 6.1 template, so this must be available for use for any TeX build attempting to compile it. The compiled PDF is also made available.

`img/` contains all of the images used and made by the program, as well as those used in the report or previous sections. 