Coresets
--------
This library contains the implementation coreset generation for k-Means and (Bayesian) Gaussian mixture models. 
It also offers the extended versions of the corresponding algorithms that support weighted data sets.

To get started, take a look at: 
>examples/intro.ipynb


Setup
-------
```
pip install -r requirements \
python setup.py build_ext --inplace
```

Running tests
-------------
In project root run:
```
python -m pytest tests/ 
```


References
---------
The implementation of the library is based on the following works:
>Bachem, O., Lucic, M., & Krause, A. (2017). Practical coreset constructions for machine learning. arXiv preprint arXiv:1703.06476.

> Bachem, O., Lucic, M., & Krause, A. (2017). Scalable and distributed clustering via lightweight coresets. arXiv preprint arXiv:1702.08248.

>Lucic, M., Faulkner, M., Krause, A., & Feldman, D. (2018). Training Gaussian Mixture Models at Scale via Coresets. Journal of Machine Learning Research, 18, Art-No.

> Borsos, Z., Bachem, O., & Krause, A. Variational Inference for DPGMM with Coresets. (2017). Advances in Approximate Bayesian Inference
 
 