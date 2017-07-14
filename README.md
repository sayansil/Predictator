# Predictator

Data can belong to various classes. Given two independant attributes of a dependant variable, we can plot that data point in a 2D-plane. The 3rd attribute having no concrete/well-defined dependency on the previous two attribute is known as the class/type of data point.
#### For example:
The two independant attributes of humans could be 1.Age and 2.Years of Programming experience.
We collect data of such 100 sample humans and plot them on a 2D plane with the corresponding x and y axis.
Now, we divide our 100 data samples into n=2 categories: 1.People who know Scala  2. People who dont know scala

We can now show the class/category of each data type on our previous plot with a different color?

But thats not it. As the headline suggests, we take this set of data points, and the set of their individual classes and analyzing it (using the kNN approach), we could tell the most likely class a data point would belong to at any given point in it.
As the data conclusive analysis is done on the basis of trend in the dataset provided, more the number of data-points, more the accuracy of the algorithm.

For a further reading on the kNN method, visit [here.](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What things you need to install the software and how to install them:

* A python environment with the following modules loaded:
* numpy
* random
* scipy
* matplotlib
* sklearn
* time
* os

#### I would personally suggest downloading Anaconda if you are hearing these names for the first time, as it comes with all such useful modules downloaded as a package.
Download Anaconda package from [here.](https://www.continuum.io/downloads)

### Installing

Still not sure about it. Project is under construction.
But feel free to tweak around stuff.

## Built With

* [Spyder](https://github.com/spyder-ide) IDE for Python
* [Anaconda](https://docs.continuum.io/anaconda/) packages
* A tiny bit of [love](https://www.google.co.in/search?q=love&oq=love&aqs=chrome..69i57j0l5.1607j0j4&sourceid=chrome&ie=UTF-8#q=what+is+love?) 

## Versioning

For the versions available, see the [tags on this repository](https://github.com/sayan2207/Class-Predictor/tags). 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Prof. JP Onnela, Department of Biostatistics, Harvard T.H. Chan School of Public Health

## Inspiration

* [Tamoghna Chowdhury](https://github.com/tamchow)
* [Kuntal Ghosh](https://github.com/kuntaltattu)
