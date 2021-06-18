![Chama](documentation/figures/logo.png)
=========================================

![build](https://github.com/sandialabs/chama/workflows/build/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/sandialabs/chama/badge.svg?branch=main)](https://coveralls.io/github/sandialabs/chama?branch=main)
[![Documentation Status](https://readthedocs.org/projects/chama/badge/?version=latest)](http://chama.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/chama)](https://pepy.tech/project/chama)

Continuous or regularly scheduled monitoring has the potential to quickly 
identify changes in the environment. However, even with low-cost sensors, only 
a limited number of sensors can be used. 
The physical placement of these sensors and the sensor technology used can have 
a large impact on the performance of a monitoring strategy.  

Chama is a Python package which includes mixed-integer, stochastic 
programming formulations to determine sensor locations and technology that maximize 
the effectiveness of the detection program. 
The software was developed to design sensor networks for water distribution networks and airborne pollutants, 
but the methods are general and 
can be applied to a wide range of applications.

For more information, go to http://chama.readthedocs.io

Citing Chama
-----------------

To cite Chama, use the following reference:

* Klise, K.A., Nicholson, B., and Laird, C.D. (2017). Sensor Placement Optimization using Chama, Sandia Report SAND2017-11472, Sandia National Laboratories.

License
------------

Revised BSD.  See the LICENSE.txt file.

Organization
------------

Directories
  * chama - Python package
  * ci - Travis CI requirements
  * documentation - User manual

Contact
-------
   * Katherine Klise, Sandia National Laboratories, kaklise@sandia.gov
   
Sandia National Laboratories is a multimission laboratory managed and operated by National Technology and 
Engineering Solutions of Sandia, LLC., a wholly owned subsidiary of Honeywell International, Inc., for the 
U.S. Department of Energy's National Nuclear Security Administration under contract DE-NA-0003525.
