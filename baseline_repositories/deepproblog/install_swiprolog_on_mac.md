1. verify with homebrew if swi-prolog exists 
```brew search swi-prolog```

2. install 
```brew install swi-prolog```

3. verify version 
```swipl --version```

4. Problem: the installed version is `SWI-Prolog version 9.2.8 for arm64-darwin` but we need not newer than 9.0.0. Older versions can be downloaded [here](https://www.swi-prolog.org/download/stable?show=all).

5. Tried some things that didn't work ... (brew extract, use brew tab [here](https://cmichel.medium.com/how-to-install-an-old-package-version-with-brew-cc1c567dd088))

6. Found instructions to build from source here (https://github.com/SWI-Prolog/swipl-devel/blob/master/CMAKE.md)

6.1 ```brew install cmake gmp libffi```

6.2 Download the version we need: ```curl -LO https://www.swi-prolog.org/download/stable/src/swipl-8.4.3.tar.gz```

6.3 unpack ```tar -xzvf swipl-8.4.3.tar.gz``` 

6.4 go in directory ```cd swipl-8.4.3```

6.5 make directory for build and go there ```mkdir build``` and ```cd build```

Follow instructions 
6.6 ```cmake -G Ninja ..```

6.7 ```ninja```

6.8 ```ninja install```

6.9 ```ctest-j 8```

7. uninstall the newer version ```brew uninstall swi-prolog```

Then, when calling `swipl --version` I get this `SWI-Prolog version 8.4.3 for arm64-darwin` which is the version we want to have. 