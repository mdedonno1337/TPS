@echo off

echo Compiling ...
echo.
echo.

del /Q build
del TPSCy.c
del TPSCy.pyd

C:\Python27\Scripts\cython.exe TPSCy.pyx
C:\Python27\python.exe setup.py build_ext --inplace --compiler=mingw32


echo.
echo.
echo.
echo End
echo.
echo.
echo.

